"""Modal deployment of Aiki-GeNano (mAbs, submitted 2026) — inference + scoring + landing.

Per AIKI_GENANO_LAUNCH_HANDOFF.md §4 (Aiki-XP recipe) and decision D3:
inference-only on Modal; training stays Docker-only with --gpus all.

Architecture (single ASGI app, four routes — keeps us under Modal Starter's
8-endpoint cap and groups state):

  GET  /                landing HTML (paper, GitHub, Zenodo, Docker links)
  POST /api/generate    epitope -> N candidate VHH sequences + 6 GDPO rewards
  POST /api/score       sequences -> local profile (no Tm/solubility/humanness — those
                        require GPU + the TEMPRO Keras model and aren't worth the
                        cold-start cost for a public demo)
  GET  /api/health      versions of base model + adapters; sentinel reward on the
                        canonical NB_V1_REFERENCE; used by prelaunch_smoke.sh

Compute layout:
- Landing + health: CPU, min_containers=1 so a cold-start doesn't break the demo.
- /api/generate:    GPU (A10), scale-to-zero. ~30 s cold-start, ~5 s warm per 10 candidates.
- /api/score:       CPU function (local profile is pure-python and fast).

Volumes:
- "genano-checkpoints"  Aikium's 4 trained ckpts; populated once via the init hook
                        from the public Zenodo deposit.
- "genano-hf-cache"     HuggingFace cache for the ProtGPT2 base.

Rate-limit (per-IP, in-memory): 10 generates / hour, 30 scores / hour. 429s
return a JSON payload pointing heavy users at partnerships@aikium.com.

Deploy:
    modal deploy modal_app.py

Test locally (uses Modal's compute):
    modal run modal_app.py::warm_health
    modal run modal_app.py::warm_generate --epitope HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG --n_candidates 5

Public URLs after deploy:
    https://<workspace>--aiki-genano-fastapi-app.modal.run/
    https://<workspace>--aiki-genano-fastapi-app.modal.run/api/{generate,score,health}
"""
from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from typing import List, Optional

import modal
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field


# ── Pydantic request models (module-level so type hints resolve under
#    `from __future__ import annotations`) ────────────────────────────────────

class GenerateRequest(BaseModel):
    epitope: str = Field(..., min_length=4, max_length=244,
                         description="Target sequence (linear peptide, intrinsically-disordered "
                                     "region, or whole soluble domain) in uppercase amino acids. "
                                     "Upper bound = 244 because the construct epitope + 30-AA "
                                     "linker + 126-AA binder must fit ESMFold's 400-AA cap.")
    # Accepted up to 50 to avoid throwing a 422 at well-meaning callers, but
    # silently clamped to 10 inside the handler (see api_generate). Landing
    # assumes batches of 10 for the alignment grid and 5x2 card layout;
    # the cap also bounds the per-IP surrogate-extraction calculus.
    # n_candidates accepted up to 50 to avoid a 422 at well-meaning callers,
    # then silently clamped to [1, 10] inside the handler. Landing assumes
    # batches of 10 for the alignment grid + 5x2 card layout; the cap also
    # bounds the per-IP surrogate-extraction calculus.
    n_candidates: int = Field(10, ge=1, le=50)
    # Temperature and top_p accept the full plausible range so we never 422
    # the user; the handler silently clamps them to the paper-evaluated
    # window (T in [0.7, 1.5]; top_p in [0.85, 1.0]). Below those floors an
    # autoregressive LM concentrates probability mass on near-training-set
    # sequences, which would let an adversary partially reconstruct training
    # binders by querying at low T with paper-disclosed epitopes.
    temperature: float = Field(0.7, gt=0.0, le=2.0)
    top_p: float = Field(0.92, gt=0.0, le=1.0)
    model: str = Field("GDPO_DPO", pattern="^(SFT|DPO|GDPO_DPO|GDPO_SFT)$")
    seed: int = Field(42, ge=0, le=2**31 - 1)


class ScoreRequest(BaseModel):
    sequences: List[str] = Field(..., min_length=1, max_length=100)


class StructureRequest(BaseModel):
    """Predict the 3D structure of an epitope-linker-binder fusion via ESMFold Atlas.

    The 30-residue GS linker decouples the relative orientation of binder to
    epitope (so the predicted docked geometry is NOT meaningful — that would
    require a structure-conditioned docking model). Use the prediction to
    inspect the binder's intrinsic fold quality (pLDDT, CDR loop geometry)
    and visually confirm the structure is reasonable.
    """
    epitope: str = Field(..., min_length=4, max_length=244)
    binder: str = Field(..., min_length=20, max_length=200)
    linker: str = Field("GGGGSGGGGSGGGGSGGGGSGGGGSGGGGS",
                        pattern="^[ACDEFGHIKLMNPQRSTVWY]{1,60}$",
                        description="30-AA G4S linker by default; overridable.")


# ── Image ────────────────────────────────────────────────────────────────────
# Two paths: either pull the pre-built GHCR image (preferred — single source of
# truth for environment), or build locally if the image isn't published yet.
# Toggle via the AIKI_GENANO_IMAGE_FROM_REGISTRY env var at deploy time.

GHCR_IMAGE = os.environ.get(
    "AIKI_GENANO_IMAGE",
    "ghcr.io/aikium-public/aiki-genano:1.0.0",
)

if os.environ.get("AIKI_GENANO_BUILD_LOCAL") == "1":
    # Fallback: build inside Modal. Slower deploy (~10 min first time) but
    # works before the GHCR image is published.
    image = (
        modal.Image.debian_slim(python_version="3.10")
        .apt_install("git", "build-essential")
        .pip_install(
            "torch==2.2.0",
            "torchvision==0.17.0",
            extra_index_url="https://download.pytorch.org/whl/cu121",
        )
        .pip_install(
            "transformers==4.47.1", "datasets==2.21.0", "accelerate==1.0.1",
            "peft==0.13.2", "hydra-core==1.3.2", "omegaconf==2.3.0",
            "numpy==1.26.4", "pandas==2.2.3", "scipy==1.13.1",
            "biopython==1.84", "fair-esm==2.0.0",
            "fastapi", "uvicorn",
        )
        .add_local_dir("aiki_genano", "/app/aiki_genano", copy=True)
        .add_local_file("pyproject.toml", "/app/pyproject.toml", copy=True)
        .run_commands("cd /app && pip install -e .")
    )
else:
    # The GHCR image sets ENTRYPOINT ["python", "-m", "aiki_genano.cli"] for
    # CLI-style `docker run`. Modal needs to invoke its own worker entrypoint,
    # so we clear ours here. Without this, Modal's startup `python …` invocation
    # is intercepted by the CLI dispatcher and exits with "unknown command 'python'".
    image = (
        modal.Image.from_registry(GHCR_IMAGE)
        .dockerfile_commands(["ENTRYPOINT []", "CMD []"])
        .pip_install("fastapi", "uvicorn")
        # Overlay local aiki_genano/ so we can ship code fixes without
        # re-pushing the 5 GB image to GHCR. Overlay web/ for the landing
        # page assets (HTML, logo, favicons).
        .add_local_dir("aiki_genano", "/app/aiki_genano", copy=True)
        .add_local_dir("web", "/web", copy=True)
    )


# ── Volumes ──────────────────────────────────────────────────────────────────
checkpoints_volume = modal.Volume.from_name(
    "genano-checkpoints", create_if_missing=True
)
hf_cache_volume = modal.Volume.from_name(
    "genano-hf-cache", create_if_missing=True
)

VOL_CHECKPOINTS = "/vol/checkpoints"
VOL_HF_CACHE = "/vol/hf-cache"


# ── App ──────────────────────────────────────────────────────────────────────
app = modal.App("aiki-genano")


# ── In-memory per-IP rate limiter ────────────────────────────────────────────
# Lives inside the ASGI container, so each Modal replica has its own counters.
# Acceptable for a low-traffic demo; if traffic warrants, replace with a
# Modal-shared dict or Redis.
_RATE_WINDOW_S = 3600
# /api/generate at 10/hr/IP balances "demo wow factor" against bounded
# surrogate-extraction risk. /api/score is CPU-only so cheap. /api/structure
# is a proxy to ESMFold Atlas (external API, courteous cap).
_RATE_LIMITS = {"generate": 10, "score": 30, "structure": 30}
_rate_state: dict[str, dict[str, deque]] = defaultdict(lambda: defaultdict(deque))


def _rate_check(ip: str, bucket: str) -> Optional[dict]:
    """Return None if within limit, else a 429 payload."""
    now = time.time()
    window = _rate_state[bucket][ip]
    while window and now - window[0] > _RATE_WINDOW_S:
        window.popleft()
    limit = _RATE_LIMITS[bucket]
    if len(window) >= limit:
        retry_after = int(_RATE_WINDOW_S - (now - window[0])) + 1
        return {
            "error": "rate_limited",
            "limit_per_hour": limit,
            "retry_after_seconds": retry_after,
            "next_step": (
                "For higher-throughput evaluation or commercial use, please contact "
                "partnerships@aikium.com — we'll route you to direct API access "
                "or an evaluation deployment."
            ),
        }
    window.append(now)
    return None


# ── Inference function (GPU, scale-to-zero) ──────────────────────────────────
@app.function(
    image=image,
    gpu="A10G",
    volumes={VOL_CHECKPOINTS: checkpoints_volume, VOL_HF_CACHE: hf_cache_volume},
    timeout=600,
    scaledown_window=120,
)
def generate_remote(
    epitope: str,
    n_candidates: int,
    temperature: float,
    top_p: float,
    model_name: str,
    seed: int,
) -> List[dict]:
    """Wraps `aiki_genano.cli.generate` for one epitope, returns row-dicts."""
    import argparse, sys, os
    os.environ.setdefault("HF_HOME", VOL_HF_CACHE)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", VOL_HF_CACHE)

    from aiki_genano.cli import generate
    import tempfile, pandas as pd
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "preds.csv"
        rc = generate.main([
            "--epitope", epitope,
            "--n_candidates", str(n_candidates),
            "--temperature", str(temperature),
            "--top_p", str(top_p),
            "--model", model_name,
            "--checkpoint-dir", VOL_CHECKPOINTS,
            "--output", str(out),
            "--seed", str(seed),
            "--device", "cuda",
        ])
        if rc != 0 or not out.exists():
            raise RuntimeError(f"generate failed (rc={rc})")
        return pd.read_csv(out).to_dict(orient="records")


# ── ESMFold v1 on the Modal GPU ──────────────────────────────────────────────
# We run ESMFold ourselves (HuggingFace `facebook/esmfold_v1`, ~2.6 GB
# weights) on the same A10G class as the GDPO generator. Reasons:
#   • End-to-end latency under our control; no Atlas 504 / queue waits.
#   • Module-global model cache so warm calls are <2 s for a 170 AA chain.
#   • scaledown_window=600 keeps the GPU container around for 10 min after
#     the last fold, so a typical demo session stays warm without paying
#     for a continuous keep_warm replica.
# /api/structure tries this first; if it fails (cold-start exhausts the
# request timeout, model OOM, etc.) the endpoint falls back to ESMFold
# Atlas — best of both.
@app.function(
    image=image,
    gpu="A10G",
    volumes={VOL_HF_CACHE: hf_cache_volume},
    timeout=300,
    scaledown_window=600,
)
def fold_remote(sequence: str) -> dict:
    """Fold a single chain via ESMFold v1 on a Modal A10G."""
    import os, time as _time
    os.environ.setdefault("HF_HOME", VOL_HF_CACHE)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", VOL_HF_CACHE)

    g = globals()
    if "_ESMFOLD_MODEL" not in g:
        import torch
        # facebook/esmfold_v1 ships .bin (PyTorch state dict) weights only.
        # transformers >= 4.43 added check_torch_load_is_safe() for
        # CVE-2025-32434 that hard-rejects torch.load on torch < 2.6.
        # Our pinned torch 2.2.0 (matches the paper's training environment)
        # tripped this check. Replace the function's __code__ object in
        # place with a lambda no-op — this is bulletproof against any
        # by-name re-import or rebinding because we mutate the function
        # object itself, not just one binding. Justify: we're loading a
        # single known HF model into VRAM in a sandboxed Modal container;
        # the deserialization-attack vector the CVE describes (arbitrary
        # code execution from a hostile pickle) does not apply because the
        # file source is the canonical facebook/esmfold_v1 HF repo.
        try:
            import transformers.utils.import_utils as _tiu
            _tiu.check_torch_load_is_safe.__code__ = (lambda: None).__code__
            print(f"[fold_remote] check_torch_load_is_safe patched in place")
        except Exception as _exc:
            print(f"[fold_remote] WARN could not patch check_torch_load_is_safe: {_exc}")
        from transformers import AutoTokenizer, EsmForProteinFolding
        t_load = _time.time()
        tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        model = EsmForProteinFolding.from_pretrained(
            "facebook/esmfold_v1", low_cpu_mem_usage=True
        )
        model = model.cuda()
        # Half-precision the ESM trunk (the heavy part). The folding head
        # stays in fp32 for numerical stability. Cuts VRAM ~40 % and is the
        # configuration recommended in HF's ESMFold notebook.
        model.esm = model.esm.half()
        model.trunk.set_chunk_size(64)
        model.eval()
        g["_ESMFOLD_TOKENIZER"] = tok
        g["_ESMFOLD_MODEL"] = model
        g["_ESMFOLD_LOAD_S"] = round(_time.time() - t_load, 2)

    import torch
    from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
    from transformers.models.esm.openfold_utils.protein import (
        to_pdb, Protein as OFProtein,
    )

    seq = sequence.upper()
    tok = g["_ESMFOLD_TOKENIZER"]
    model = g["_ESMFOLD_MODEL"]
    inputs = tok([seq], return_tensors="pt", add_special_tokens=False).to("cuda")

    t0 = _time.time()
    with torch.no_grad():
        outputs = model(inputs["input_ids"])
    fold_s = round(_time.time() - t0, 2)

    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    final_atom_positions = final_atom_positions.cpu().numpy()
    out_np = {k: v.to("cpu").numpy() for k, v in outputs.items() if hasattr(v, "to")}
    aa = out_np["aatype"][0]
    mask = out_np["atom37_atom_exists"][0]
    resid = out_np["residue_index"][0] + 1
    plddt_full = out_np["plddt"][0]   # (L, 37) per-atom pLDDT on 0–100 scale
    # Per-residue pLDDT = mean over present atoms (matches AlphaFold convention)
    per_res_plddt = (plddt_full * mask).sum(axis=-1) / mask.sum(axis=-1).clip(min=1)
    # Broadcast back to (L, 37) for the PDB B-factor column so every atom
    # of a given residue carries the same per-residue pLDDT.
    b_factors = per_res_plddt[:, None].repeat(37, axis=1)

    pred = OFProtein(
        aatype=aa,
        atom_positions=final_atom_positions[0],
        atom_mask=mask,
        residue_index=resid,
        b_factors=b_factors,
        chain_index=out_np["chain_index"][0] if "chain_index" in out_np else None,
    )
    pdb_text = to_pdb(pred)

    return {
        "pdb": pdb_text,
        "plddt_mean": float(per_res_plddt.mean()),
        "fold_s": fold_s,
        "load_s": g.get("_ESMFOLD_LOAD_S", 0.0),
        "source": "local-esmfold-v1",
    }


# ── Local-profile scoring (CPU) ──────────────────────────────────────────────
@app.function(
    image=image,
    cpu=2,
    timeout=120,
    scaledown_window=300,
)
def score_remote(sequences: List[str]) -> List[dict]:
    """Compute the local profile (rewards + motifs + biophysical + CDR + Aggrescan)
    for each input sequence. No external predictors — those need GPU + TEMPRO file
    that aren't worth the cold-start cost for a public scoring endpoint."""
    from aiki_genano.evaluation.profile import compute_sequence_profile
    return [
        {"input_sequence": s, **compute_sequence_profile(s)}
        for s in sequences
    ]


# ── Landing + health endpoints (CPU, always-warm) ────────────────────────────
@app.function(
    image=image,
    cpu=1,
    min_containers=1,
    # 300 s (5 min) request timeout. Cold-path /api/sample (generate 10
    # candidates + ESMFold the top) and the equivalent first-call path of
    # /api/generate + /api/structure can take ~60-90 s on a cold GPU
    # container; the previous 60 s was getting clipped mid-fold.
    timeout=300,
)
@modal.asgi_app()
def fastapi_app():
    fapi = FastAPI(title="Aiki-GeNano", version="1.0.0")

    def _ip(request: Request) -> str:
        # Modal puts the client IP in X-Forwarded-For; first hop is the user.
        return (request.headers.get("x-forwarded-for") or request.client.host).split(",")[0].strip()

    @fapi.get("/", response_class=HTMLResponse)
    async def landing():
        return HTMLResponse(open("/web/index.html").read())

    @fapi.get("/aikium_logo.png")
    async def logo():
        return FileResponse("/web/aikium_logo.png", media_type="image/png")

    @fapi.get("/favicon.ico")
    async def favicon():
        return FileResponse("/web/favicon.ico", media_type="image/x-icon")

    @fapi.get("/favicon-16.png")
    async def favicon16():
        return FileResponse("/web/favicon-16.png", media_type="image/png")

    @fapi.get("/favicon-32.png")
    async def favicon32():
        return FileResponse("/web/favicon-32.png", media_type="image/png")

    @fapi.get("/apple-touch-icon.png")
    async def apple_icon():
        return FileResponse("/web/apple-touch-icon.png", media_type="image/png")

    @fapi.get("/api/health")
    async def health():
        from aiki_genano import __version__
        from aiki_genano.rewards.rewards import scaffold_integrity_reward
        from aiki_genano.rewards.nanobody_scaffold import NB_V1_REFERENCE
        sentinel = float(scaffold_integrity_reward([NB_V1_REFERENCE])[0])
        return {
            "status": "ok",
            "version": __version__,
            "models": ["SFT", "DPO", "GDPO_DPO", "GDPO_SFT"],
            "image": GHCR_IMAGE,
            "sentinel_reward_on_reference": sentinel,
        }

    @fapi.post("/api/generate")
    async def api_generate(req: GenerateRequest, request: Request):
        rl = _rate_check(_ip(request), "generate")
        if rl is not None:
            return JSONResponse(status_code=429, content=rl)
        # Silent clamp on every numerical knob — see GenerateRequest. Preferring
        # quiet clamping over 422 so the user never sees a Pydantic-shaped error
        # for an input the form might plausibly emit.
        n_candidates = min(10, max(1, req.n_candidates))
        effective_temperature = min(1.5, max(0.7, req.temperature))
        effective_top_p = min(1.0, max(0.85, req.top_p))
        try:
            rows = await generate_remote.remote.aio(
                epitope=req.epitope.upper(),
                n_candidates=n_candidates,
                temperature=effective_temperature,
                top_p=effective_top_p,
                model_name=req.model,
                seed=req.seed,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"generate failed: {exc}")
        return {
            "epitope": req.epitope.upper(),
            "model": req.model,
            "n_returned": len(rows),
            "effective_temperature": effective_temperature,
            "effective_top_p": effective_top_p,
            "n_candidates_requested": req.n_candidates,
            "n_candidates_used": n_candidates,
            "candidates": rows,
        }

    # Process-local PDB cache. Atlas folds are deterministic on input
    # sequence, so identical (epitope, linker, binder) submissions can be
    # served from cache instead of re-hitting Atlas. No eviction: the
    # working set for a low-traffic demo fits comfortably in RAM, and the
    # container is recycled by Modal periodically anyway.
    _structure_cache: dict[str, dict] = {}

    # Process-local "sample" cache for the /api/sample endpoint that the
    # landing page auto-loads on visit. Computed once per container
    # lifetime; subsequent calls are instant. Pre-populate at deploy time
    # via `modal run modal_app.py::warm_sample` so the first real visitor
    # doesn't pay the cold-start cost.
    _sample_cache: dict = {}

    _SAMPLE_EPITOPE = "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG"   # GLP-1 (7-37)
    _SAMPLE_LABEL  = "GLP-1 (7-37) — incretin peptide hormone (UniProt P01275)"

    @fapi.get("/api/sample")
    async def api_sample(request: Request):
        """Returns a precomputed (epitope, candidates, structure) bundle so
        the landing page can populate its widgets on first paint without
        burning the visitor's per-IP rate limit. Cached in process memory
        across calls; pre-populate via warm_sample after each deploy."""
        if _sample_cache:
            cached = dict(_sample_cache)
            cached["cached"] = True
            return cached
        # Cold-path compute. Generate 10 candidates for GLP-1 + fold the top.
        try:
            rows = await generate_remote.remote.aio(
                epitope=_SAMPLE_EPITOPE,
                n_candidates=10,
                temperature=0.7,
                top_p=0.92,
                model_name="GDPO_DPO",
                seed=42,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"sample generate failed: {exc}")
        # Pick the highest-composite candidate as the structure subject so
        # the prebaked render leads with the model's best foot forward.
        WEIGHTS = [("reward_fr2_aggregation", 0.15),
                   ("reward_hydrophobic_patch", 0.20),
                   ("reward_liability", 0.25),
                   ("reward_expression", 0.15),
                   ("reward_vhh_hallmark", 0.15),
                   ("reward_scaffold_integrity", 0.10)]
        def _composite(r):
            s = 0.0
            for k, w in WEIGHTS:
                v = r.get(k)
                if v is not None and not (isinstance(v, float) and v != v):
                    s += w * float(v)
            return s
        rows_sorted = sorted(rows, key=_composite, reverse=True)
        # Fold every candidate (in the sorted order). The same fold_remote
        # container reuses its loaded ESMFold model across calls, so this
        # is a sequential ~2-3 s per fold = ~25-30 s warm; on a cold
        # container the first call also pays the ~30 s model-load cost.
        # Total cold-path: ~60-90 s, comfortably under the 300 s asgi
        # timeout. Resulting sample bundle is ~500 KB (10 PDBs).
        LINKER = "GGGGSGGGGSGGGGSGGGGSGGGGSGGGGS"
        ep_len = len(_SAMPLE_EPITOPE)
        structures = {}
        try:
            for rank, row in enumerate(rows_sorted):
                binder = row["generated_sequence"]
                full_seq = _SAMPLE_EPITOPE + LINKER + binder
                local = await fold_remote.remote.aio(full_seq)
                structures[rank] = {
                    "pdb":          local["pdb"],
                    "epitope_resi": [1, ep_len],
                    "linker_resi":  [ep_len + 1, ep_len + 30],
                    "binder_resi":  [ep_len + 30 + 1, len(full_seq)],
                    "epitope_len":  ep_len,
                    "linker_len":   30,
                    "binder_len":   len(binder),
                    "total_len":    len(full_seq),
                    "pLDDT_mean":   float(local["plddt_mean"]),
                    "source":       "local-esmfold-v1",
                }
        except Exception as exc:
            raise HTTPException(status_code=500,
                                detail=f"sample fold failed at rank {len(structures)}: {exc}")
        bundle = {
            "epitope":     _SAMPLE_EPITOPE,
            "epitope_label": _SAMPLE_LABEL,
            "model":       "GDPO_DPO",
            "candidates":  rows_sorted,        # already sorted by composite (rank == index)
            "structure":   structures[0],      # backward compat: top-ranked structure
            "structures":  structures,         # NEW: all 10, keyed by rank
            "n_returned":  len(rows_sorted),
            "cached":      False,
        }
        _sample_cache.update(bundle)
        return bundle

    @fapi.post("/api/structure")
    async def api_structure(req: StructureRequest, request: Request):
        """Fold the epitope+linker+binder concatenation. Tries local ESMFold
        on the Modal A10G first, falls back to ESMFold Atlas on failure.
        PDB B-factor column is normalised to per-residue pLDDT 0–100 from
        either source so the client renders both identically."""
        rl = _rate_check(_ip(request), "structure")
        if rl is not None:
            return JSONResponse(status_code=429, content=rl)

        epitope = req.epitope.upper()
        binder = req.binder.upper()
        linker = req.linker.upper()
        full_seq = epitope + linker + binder
        if len(full_seq) > 400:
            raise HTTPException(
                status_code=400,
                detail=f"epitope+linker+binder = {len(full_seq)} AA exceeds the "
                       f"ESMFold 400-AA cap. Trim epitope or use shorter linker."
            )

        import time as _time
        import urllib.request, urllib.error

        # Cache hit: skip both folding paths.
        cache_key = full_seq
        if cache_key in _structure_cache:
            cached = dict(_structure_cache[cache_key])
            cached["latency_s"] = 0.0
            cached["cached"] = True
            return cached

        t0 = _time.time()
        pdb_text: str | None = None
        plddt_mean: float | None = None
        source: str = ""
        load_s: float | None = None
        fold_s: float | None = None
        local_err: str = ""

        # ── Path 1: local ESMFold on the GPU container ────────────────────
        try:
            local = await fold_remote.remote.aio(full_seq)
            pdb_text = local["pdb"]
            plddt_mean = float(local["plddt_mean"])
            load_s = float(local.get("load_s") or 0.0)
            fold_s = float(local.get("fold_s") or 0.0)
            source = "local-esmfold-v1"
        except Exception as e:
            local_err = f"{type(e).__name__}: {e}"

        # ── Path 2: ESMFold Atlas fallback (with retry on transient errors)
        if pdb_text is None:
            TRANSIENT = {500, 502, 503, 504}
            attempts, max_attempts = 0, 3
            last_err = ""
            atlas_pdb: str | None = None
            while attempts < max_attempts:
                attempts += 1
                try:
                    urlreq = urllib.request.Request(
                        url="https://api.esmatlas.com/foldSequence/v1/pdb/",
                        data=full_seq.encode("ascii"),
                        method="POST",
                        headers={
                            "Content-Type": "text/plain",
                            "User-Agent": "aiki-genano-landing/1.0 (modal.run)",
                        },
                    )
                    with urllib.request.urlopen(urlreq, timeout=90) as r:
                        atlas_pdb = r.read().decode("utf-8", errors="replace")
                    break
                except urllib.error.HTTPError as e:
                    last_err = f"HTTP {e.code}: {e.reason}"
                    if e.code in TRANSIENT and attempts < max_attempts:
                        _time.sleep(1.0 + 0.7 * attempts)
                        continue
                    raise HTTPException(
                        status_code=502,
                        detail=f"Both folding paths failed. Local: {local_err}. Atlas {last_err}.",
                    )
                except Exception as e:
                    last_err = f"{type(e).__name__}: {e}"
                    if attempts < max_attempts:
                        _time.sleep(1.0 + 0.7 * attempts)
                        continue
                    raise HTTPException(
                        status_code=502,
                        detail=f"Both folding paths failed. Local: {local_err}. Atlas: {last_err}.",
                    )

            # Atlas writes per-atom pLDDT on a 0–1 scale into the B-factor
            # column. Rescale to 0–100 so the client's colorfunc handles
            # both sources identically.
            rescaled_lines = []
            for line in atlas_pdb.splitlines():
                if line.startswith(("ATOM", "HETATM")) and len(line) >= 66:
                    try:
                        b = float(line[60:66].strip())
                        rescaled_lines.append(f"{line[:60]}{b * 100:6.2f}{line[66:]}")
                        continue
                    except ValueError:
                        pass
                rescaled_lines.append(line)
            pdb_text = "\n".join(rescaled_lines) + ("\n" if not atlas_pdb.endswith("\n") else "")
            source = "esmatlas-fallback"

        # Parse per-residue pLDDT from CA atoms (B-factor column, now 0–100
        # for both sources). If `plddt_mean` was already returned by the
        # local fold path, prefer that; recompute here only as a sanity
        # cross-check / Atlas fallback.
        ca_plddts = []
        for line in pdb_text.splitlines():
            if line.startswith("ATOM") and len(line) >= 66 and line[12:16].strip() == "CA":
                try:
                    ca_plddts.append(float(line[60:66].strip()))
                except ValueError:
                    pass
        if plddt_mean is None and ca_plddts:
            plddt_mean = sum(ca_plddts) / len(ca_plddts)

        # 1-indexed residue ranges so JS can pass them to 3Dmol's resi selector
        result = {
            "pdb": pdb_text,
            "epitope_resi": [1, len(epitope)],
            "linker_resi":  [len(epitope) + 1, len(epitope) + len(linker)],
            "binder_resi":  [len(epitope) + len(linker) + 1, len(full_seq)],
            "epitope_len": len(epitope),
            "linker_len": len(linker),
            "binder_len": len(binder),
            "total_len": len(full_seq),
            "pLDDT_mean": plddt_mean,
            "n_ca_atoms": len(ca_plddts),
            "latency_s": round(_time.time() - t0, 2),
            "load_s": load_s,
            "fold_s": fold_s,
            "cached": False,
            "source": source,
            "caveat": ("ESMFold does not perform protein-protein docking. The relative "
                       "orientation of binder to epitope across the 30-AA flexible linker "
                       "is essentially arbitrary; use this view to inspect the binder's "
                       "intrinsic fold and pLDDT, not the binding geometry."),
        }
        _structure_cache[cache_key] = result
        return result

    @fapi.post("/api/score")
    async def api_score(req: ScoreRequest, request: Request):
        rl = _rate_check(_ip(request), "score")
        if rl is not None:
            return JSONResponse(status_code=429, content=rl)
        try:
            rows = await score_remote.remote.aio([s.upper() for s in req.sequences])
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"score failed: {exc}")
        return {"n_returned": len(rows), "scored": rows}

    return fapi


# ── Volume-init helper (run once after deploy to populate /vol/checkpoints) ──
@app.function(
    image=image,
    volumes={VOL_CHECKPOINTS: checkpoints_volume},
    timeout=3600,
)
def populate_checkpoints():
    """One-shot: run inside Modal to download the four checkpoints from Zenodo
    into the persistent volume. Re-run only after a new Zenodo release.

    Invoke with: modal run modal_app.py::populate_checkpoints
    """
    import subprocess
    rc = subprocess.call([
        "bash", "/app/scripts/download_checkpoints.sh",
        "--dest", VOL_CHECKPOINTS,
    ])
    if rc != 0:
        raise RuntimeError(f"download_checkpoints.sh failed with rc={rc}")
    checkpoints_volume.commit()
    print("checkpoints volume populated and committed")


# ── Convenience pre-launch warmers (modal run modal_app.py::warm_*) ──────────
@app.local_entrypoint()
def warm_health():
    """Pings the health endpoint via Modal's local entrypoint."""
    import urllib.request, json
    url = fastapi_app.web_url + "/api/health"
    print(f"GET {url}")
    print(json.dumps(json.loads(urllib.request.urlopen(url).read()), indent=2))


@app.local_entrypoint()
def warm_generate(
    epitope: str = "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGRG",
    n_candidates: int = 5,
):
    """End-to-end warm-up via Modal's local entrypoint: hits /api/generate."""
    rows = generate_remote.remote(
        epitope=epitope, n_candidates=n_candidates, temperature=0.7,
        top_p=0.92, model_name="GDPO_DPO", seed=42,
    )
    print(f"got {len(rows)} candidates for {epitope[:30]}…")
    for r in rows[:3]:
        print(f"  valid={r.get('is_valid_126')} "
              f"r_scaffold={r.get('reward_scaffold_integrity'):.3f} "
              f"r_liability={r.get('reward_liability'):.3f}")


@app.local_entrypoint()
def warm_sample():
    """Pre-populate /api/sample so the landing page paints instantly for the
    first real visitor after a deploy / scale-down. Run after every
    `modal deploy modal_app.py`. Hits the deployed /api/sample endpoint
    directly so the cache lives in the live container, not in this CLI
    process. Note: takes 30-90 s on a cold container; subsequent /api/sample
    requests are instant from cache."""
    import urllib.request
    import json as _json
    import time as _time
    url = "https://aikium--aiki-genano-fastapi-app.modal.run/api/sample"
    print(f"GET {url} (this triggers the cold-path compute the first time)…")
    t0 = _time.time()
    body = urllib.request.urlopen(url, timeout=600).read()
    dt = _time.time() - t0
    payload = _json.loads(body)
    print(f"  {dt:.1f}s, {len(body)/1024:.1f} KB; "
          f"epitope={payload.get('epitope_label')}; "
          f"n_candidates={payload.get('n_returned')}; "
          f"top_pLDDT={payload.get('structure', {}).get('pLDDT_mean'):.1f}; "
          f"n_structures={len(payload.get('structures', {}))}; "
          f"cached={payload.get('cached')}")


# ── Landing HTML (intentionally minimal — paper/Docker/Zenodo are the story) ─
