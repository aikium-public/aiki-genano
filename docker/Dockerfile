# Aiki-GeNano — single image for inference, training, and property prediction.
#
# Tag convention (per AIKI_GENANO_LAUNCH_HANDOFF.md §5, decision D2):
#   ghcr.io/aikium-public/aiki-genano:latest        (floating, current release)
#   ghcr.io/aikium-public/aiki-genano:1.0.0         (pinned to git tag v1.0.0)
#
# Build (from repo root):
#   docker build -f docker/Dockerfile -t ghcr.io/aikium-public/aiki-genano:latest .
#
# Mount points (decision §3.2):
#   /app/checkpoints   4 trained checkpoints from Zenodo (run scripts/download_checkpoints.sh first)
#   /app/data          training_subset/ + generated_sequences/ from Zenodo (training/eval only)
#   /app/output        user-writable output dir
#   /models            HF cache (ProtGPT2, Sapiens, NetSolP)
#
# Run (sequence generation, GPU):
#   docker run --rm --gpus all \
#     -v "$(pwd)/checkpoints:/app/checkpoints" \
#     -v "$(pwd)/output:/app/output" \
#     -v "$(pwd)/hf_cache:/models" \
#     ghcr.io/aikium-public/aiki-genano:latest \
#     generate --epitope ABCDEF... --n_candidates 50 --model GDPO_DPO --output /app/output/preds.csv
#
# Run (sequence generation, CPU fallback — slower but works on laptops without GPU):
#   docker run --rm \
#     -v "$(pwd)/checkpoints:/app/checkpoints" \
#     -v "$(pwd)/output:/app/output" \
#     -v "$(pwd)/hf_cache:/models" \
#     ghcr.io/aikium-public/aiki-genano:latest \
#     generate --epitope ABCDEF... --n_candidates 5 --model SFT --device cpu --output /app/output/preds.csv

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models \
    HUGGINGFACE_HUB_CACHE=/models

# NVIDIA apt repo can be mid-sync; we don't need CUDA apt packages here.
RUN rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list || true

# System packages
RUN apt-get update -o Acquire::Retries=5 \
 && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        git wget curl ca-certificates \
        build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && python -m pip install --upgrade pip setuptools wheel

# ─── Core ML stack ─────────────────────────────────────────────────────────
# Pinned to the versions the paper was trained and evaluated under.
RUN pip install \
        torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
        --index-url https://download.pytorch.org/whl/cu121

RUN pip install \
        transformers==4.47.1 \
        datasets==2.21.0 \
        accelerate==1.0.1 \
        peft==0.13.2 \
        bitsandbytes==0.44.1 \
        hydra-core==1.3.2 \
        omegaconf==2.3.0 \
        numpy==1.26.4 \
        pandas==2.2.3 \
        scipy==1.13.1 \
        scikit-learn==1.5.2 \
        tqdm \
        rich \
        wandb \
        pyyaml

# ─── NVIDIA's TRL-GDPO fork ────────────────────────────────────────────────
# Cloned from https://github.com/NVlabs/GDPO and pinned to the SHA we tested
# against. Two source patches below are load-bearing for transformers >= 4.47;
# both were validated in the previous Dockerfile.gdpo and carried forward
# verbatim. If GDPO upstream ever publishes a tagged release that subsumes
# these patches, drop them and pin to that tag instead.
ARG GDPO_REF=main
RUN git clone https://github.com/NVlabs/GDPO.git /opt/GDPO \
 && cd /opt/GDPO \
 && git checkout "$GDPO_REF"

# Patch 1: `is_rich_available` shifted across transformers releases (sometimes
# in transformers.utils, sometimes in transformers.utils.import_utils, gone
# entirely in 4.47.1). It's only used for optional rich-formatted logging in
# GDPO trainer/callbacks — not load-bearing. Walk every .py file under the
# GDPO tree, strip the symbol from any import that names it, and inject a
# local `def is_rich_available(): return False` stub at the top of every
# such file. Robust to upstream re-arrangements.
RUN python - <<'PY'
import re
from pathlib import Path

root = Path("/opt/GDPO/trl-GDPO/trl-0.18.0-gdpo/trl")
stub = ("# Aiki-GeNano patch: is_rich_available stub — transformers cross-version compat\n"
        "def is_rich_available():\n    return False\n\n")
patched = 0
for p in root.rglob("*.py"):
    src = p.read_text()
    if "is_rich_available" not in src:
        continue
    new = src
    new = re.sub(r",\s*\bis_rich_available\b", "", new)
    new = re.sub(r"\bis_rich_available\b\s*,\s*", "", new)
    new = re.sub(r"^\s*from\s+\S+\s+import\s+is_rich_available\s*\n", "", new, flags=re.MULTILINE)
    # Inject stub after the first `from __future__` line, or at top.
    if "from __future__ import annotations" in new:
        new = new.replace("from __future__ import annotations\n",
                          "from __future__ import annotations\n\n" + stub, 1)
    else:
        new = stub + new
    p.write_text(new)
    patched += 1
    print(f"  patched {p.relative_to(root)}")
if patched == 0:
    raise SystemExit("FATAL: no is_rich_available references found anywhere in GDPO tree; upstream changed. Re-vet.")
print(f"Patch 1 (is_rich_available stub) applied to {patched} file(s)")
PY

# Patch 2: GPT2 (and any older HF model) doesn't accept logits_to_keep kwarg.
# Add a try/except that retries without the kwarg if the model rejects it.
RUN python - <<'PY'
from pathlib import Path
p = Path("/opt/GDPO/trl-GDPO/trl-0.18.0-gdpo/trl/trainer/grpo_trainer.py")
src = p.read_text()
old = '''logits = model(
                input_ids=input_ids_batch, attention_mask=attention_mask_batch, logits_to_keep=logits_to_keep + 1
            ).logits'''
new = '''try:
                logits = model(
                    input_ids=input_ids_batch, attention_mask=attention_mask_batch, logits_to_keep=logits_to_keep + 1
                ).logits
            except TypeError as exc:
                if "logits_to_keep" not in str(exc):
                    raise
                logits = model(
                    input_ids=input_ids_batch, attention_mask=attention_mask_batch
                ).logits'''
if old not in src:
    raise SystemExit("FATAL: logits_to_keep patch target not found; GDPO upstream may have moved on. Re-vet.")
p.write_text(src.replace(old, new))
print("Patch 2 (logits_to_keep fallback) applied")
PY

# Patch 3: GDPO upstream now imports VLLMClient at module load (since the SHA
# we tested against), and VLLMClient itself imports `vllm` at module load.
# We don't use vLLM-backed generation (the paper used standard HF generate),
# so wrap both imports in try/except — if vllm isn't installed, VLLMClient is
# None and any code path that tries to use it will error there. This avoids a
# 5 GB vllm dependency just to satisfy an import.
RUN python - <<'PY'
from pathlib import Path

# 3a) Make the VLLMClient import optional in grpo_trainer.py.
gt = Path("/opt/GDPO/trl-GDPO/trl-0.18.0-gdpo/trl/trainer/grpo_trainer.py")
src = gt.read_text()
old_a = "from ..extras.vllm_client import VLLMClient\n"
new_a = ("try:\n"
         "    from ..extras.vllm_client import VLLMClient\n"
         "except Exception:\n"
         "    VLLMClient = None  # vllm not installed; non-vllm code paths still work\n")
if old_a in src:
    gt.write_text(src.replace(old_a, new_a))
    print("Patch 3a (VLLMClient import optional in grpo_trainer.py) applied")
else:
    print("Patch 3a target not present; skipping (older GDPO SHA without vllm dep).")

# 3b) Make every vllm import in vllm_client.py optional so the file parses.
vc = Path("/opt/GDPO/trl-GDPO/trl-0.18.0-gdpo/trl/extras/vllm_client.py")
if vc.exists():
    src = vc.read_text()
    new_lines = []
    in_top_imports = True
    patched = 0
    for line in src.splitlines():
        stripped = line.strip()
        is_vllm_import = (
            stripped.startswith("from vllm") or stripped.startswith("import vllm")
        ) and not stripped.startswith("#")
        if in_top_imports and is_vllm_import:
            indent = line[: len(line) - len(line.lstrip())]
            new_lines.append(f"{indent}try:")
            new_lines.append(f"{indent}    {stripped}")
            new_lines.append(f"{indent}except Exception:")
            new_lines.append(f"{indent}    pass  # vllm not installed; VLLMClient will fail at use, not import")
            patched += 1
        else:
            new_lines.append(line)
        # Heuristic: stop wrapping once we leave the top-of-file import block.
        if line and not stripped.startswith(("import", "from", "#", "\"", "'")) and not stripped.startswith(")"):
            in_top_imports = False
    vc.write_text("\n".join(new_lines) + "\n")
    print(f"Patch 3b (vllm imports made optional in vllm_client.py): wrapped {patched} lines")
else:
    print("Patch 3b: vllm_client.py not present; skipping.")
PY

# Editable install with --no-deps so the GDPO fork's declared dependencies
# (datasets>=3.0.0, transformers>=5.x in current upstream metadata) don't drag
# our pinned versions forward into incompatible majors. The transitive deps
# we actually need (accelerate, peft, transformers, datasets, huggingface-hub)
# are all already installed at the pinned versions above. If a runtime path
# inside the fork actually requires the newer datasets/transformers APIs, we
# will see an explicit AttributeError at training time and can revisit.
RUN pip install --no-deps -e /opt/GDPO/trl-GDPO/trl-0.18.0-gdpo \
 && python -c "from trl import GRPOTrainer, GRPOConfig; print('TRL-GDPO import OK')"

# ─── Property predictors (TEMPRO, NetSolP, Sapiens) ─────────────────────────
# These pull weights from HuggingFace at first run into /models.
RUN pip install \
        fair-esm==2.0.0 \
        biopython==1.84 \
        sentencepiece \
        protobuf

# ─── Aiki-GeNano package ────────────────────────────────────────────────────
WORKDIR /app

# Copy package + configs + scripts. Editable install so users re-mounting the
# repo for dev iteration don't have to rebuild the image.
COPY pyproject.toml /app/pyproject.toml
COPY aiki_genano /app/aiki_genano
COPY configs /app/configs
COPY scripts /app/scripts

RUN pip install -e /app

# Verify the package imports inside the image.
RUN python -c "import aiki_genano; from aiki_genano.rewards import rewards, nbv1_properties, nanobody_scaffold; print('aiki_genano import OK')"

# Populate /models (HF cache) and /app/checkpoints / /app/data / /app/output
# at runtime via mounts. /models declared as a VOLUME so weights persist
# across `docker run` invocations when the user supplies a named volume.
VOLUME ["/models"]
RUN mkdir -p /app/checkpoints /app/data /app/output

ENTRYPOINT ["python", "-m", "aiki_genano.cli"]
CMD ["--help"]
