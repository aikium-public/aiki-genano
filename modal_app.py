"""Modal deployment of Aiki-GeNano (mAbs 2026) — inference + scoring + landing.

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
    modal run modal_app.py::warm_generate --epitope MNYPLTLEMDLENLEDLFWELDRLDNYNDTSLVENHL --n_candidates 5

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
    image = modal.Image.from_registry(GHCR_IMAGE).pip_install("fastapi", "uvicorn")


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
_RATE_LIMITS = {"generate": 10, "score": 30}
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
    timeout=60,
)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel, Field

    fapi = FastAPI(title="Aiki-GeNano", version="1.0.0")

    class GenerateRequest(BaseModel):
        epitope: str = Field(..., min_length=4, max_length=200,
                             description="Target epitope peptide sequence (uppercase AAs).")
        n_candidates: int = Field(10, ge=1, le=50)
        temperature: float = Field(0.7, gt=0.0, le=2.0)
        top_p: float = Field(0.92, gt=0.0, le=1.0)
        model: str = Field("GDPO_DPO",
                           pattern="^(SFT|DPO|GDPO_DPO|GDPO_SFT)$")
        seed: int = Field(42, ge=0, le=2**31 - 1)

    class ScoreRequest(BaseModel):
        sequences: List[str] = Field(..., min_length=1, max_length=100)

    def _ip(request: Request) -> str:
        # Modal puts the client IP in X-Forwarded-For; first hop is the user.
        return (request.headers.get("x-forwarded-for") or request.client.host).split(",")[0].strip()

    @fapi.get("/", response_class=HTMLResponse)
    async def landing():
        return _LANDING_HTML

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
        try:
            rows = generate_remote.remote(
                epitope=req.epitope.upper(),
                n_candidates=req.n_candidates,
                temperature=req.temperature,
                top_p=req.top_p,
                model_name=req.model,
                seed=req.seed,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"generate failed: {exc}")
        return {"epitope": req.epitope.upper(), "model": req.model,
                "n_returned": len(rows), "candidates": rows}

    @fapi.post("/api/score")
    async def api_score(req: ScoreRequest, request: Request):
        rl = _rate_check(_ip(request), "score")
        if rl is not None:
            return JSONResponse(status_code=429, content=rl)
        try:
            rows = score_remote.remote([s.upper() for s in req.sequences])
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
    epitope: str = "MNYPLTLEMDLENLEDLFWELDRLDNYNDTSLVENHL",
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


# ── Landing HTML (intentionally minimal — paper/Docker/Zenodo are the story) ─
_LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Aiki-GeNano — preference-optimized nanobody design</title>
<style>
  body { font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
         max-width: 760px; margin: 4rem auto; padding: 0 1.25rem; color: #181818; }
  h1 { font-size: 2rem; margin: 0 0 .25rem; }
  h2 { font-size: 1.1rem; margin: 2rem 0 .5rem; color: #333; }
  p, li { color: #2a2a2a; }
  code { background: #f3f3f3; padding: .12em .4em; border-radius: 3px; font-size: .94em; }
  pre  { background: #f3f3f3; padding: .9em 1em; border-radius: 4px; overflow-x: auto; }
  a { color: #0d4c8a; }
  .pill { display: inline-block; padding: .2em .6em; margin-right: .35em; margin-bottom: .35em;
          background: #eef3f8; border-radius: 12px; font-size: .82em; }
  .footer { color: #888; font-size: .85em; margin-top: 3rem; border-top: 1px solid #eee; padding-top: 1rem; }
</style>
</head>
<body>
  <h1>Aiki-GeNano</h1>
  <p><em>Preference-optimized generation of developable nanobodies across 65 epitope targets.</em>
     mAbs (2026), Aikium Inc.</p>
  <div>
    <span class="pill">paper: <a href="https://github.com/aikium-public/aiki-genano#paper">bioRxiv</a></span>
    <span class="pill">code: <a href="https://github.com/aikium-public/aiki-genano">GitHub</a></span>
    <span class="pill">data + weights: <a href="https://zenodo.org/records/PLACEHOLDER">Zenodo</a></span>
    <span class="pill">image: <code>ghcr.io/aikium-public/aiki-genano:1.0.0</code></span>
  </div>

  <h2>Try it</h2>
  <p>Generate ten candidate nanobody sequences for a target epitope (POST JSON):</p>
<pre>curl -X POST $URL/api/generate \\
  -H "Content-Type: application/json" \\
  -d '{
    "epitope": "MNYPLTLEMDLENLEDLFWELDRLDNYNDTSLVENHL",
    "n_candidates": 10,
    "model": "GDPO_DPO"
  }'</pre>

  <p>Score sequences you already have (returns the same six GDPO rewards
     used during training, plus motif/biophysical/CDR descriptors):</p>
<pre>curl -X POST $URL/api/score \\
  -H "Content-Type: application/json" \\
  -d '{"sequences": ["QVQLVESGGGS..."]}'</pre>

  <h2>Limits</h2>
  <p>10 generations/hour and 30 scores/hour per IP. For higher throughput or
     commercial evaluation, please contact <a href="mailto:partnerships@aikium.com">partnerships@aikium.com</a>.
     The endpoint is GPU-backed and scales to zero — first call after an idle
     period takes ~30&nbsp;seconds.</p>

  <h2>Reproducibility</h2>
  <p>For the full paper pipeline (SFT → DPO → GDPO training, property prediction
     with TEMPRO + NetSolP + Sapiens, and the analysis notebook that produces
     every figure), use the Docker image with <code>--gpus all</code>:</p>
<pre>docker run --rm --gpus all \\
  -v $PWD/checkpoints:/app/checkpoints \\
  -v $PWD/output:/app/output \\
  ghcr.io/aikium-public/aiki-genano:1.0.0 \\
  predict --sequences /app/output/preds.csv --with-properties \\
          --output /app/output/profiled.csv</pre>

  <p class="footer">
    Each upstream model and dataset retains its own licence. Users deploying
    Aiki-GeNano, its outputs, or derived sequences in their own workflows are
    responsible for complying with the respective upstream terms.
  </p>
</body>
</html>"""
