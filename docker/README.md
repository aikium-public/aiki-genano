# Aiki-GeNano Docker image

Single image for sequence generation, property prediction, and re-training. Replaces the prior `:inference` / `:full` split (decision D2 — see `data/_internal_planning/PLANNING_DOCKER_MODAL.md`).

## Tags

| Tag | Purpose |
|---|---|
| `ghcr.io/aikium-public/aiki-genano:latest` | Floating tag, current release. |
| `ghcr.io/aikium-public/aiki-genano:1.0.0` | Pinned to git tag `v1.0.0`. Use this for reproducibility. |

## Mount points

| Path | Contents | Required for |
|---|---|---|
| `/app/checkpoints` | Four trained checkpoints (SFT, DPO, GDPO_DPO, GDPO_SFT). NDA-gated; request via `partnerships@aikium.com`. | `generate`, `predict` |
| `/app/data` | User-supplied training data and/or sequence CSVs. The Aikium screening corpus and the generated-sequence amino-acid strings are not redistributed (figure-data tables only are available at the Zenodo deposit, [10.5281/zenodo.19757842](https://doi.org/10.5281/zenodo.19757842)). | `train`, `predict --recompute` |
| `/app/output` | User-writable output directory for predictions and generated FASTAs. | `generate`, `predict` |
| `/models` | HuggingFace cache (ProtGPT2, Sapiens, NetSolP-ESM weights). Persist across runs by mounting a named volume. | `predict --with-properties`, first `generate` run |

## Quickstart

### 1. Obtain the trained checkpoints (NDA)

```bash
git clone https://github.com/aikium-public/aiki-genano.git
cd aiki-genano
# Place ./checkpoints/{SFT,DPO,GDPO_DPO,GDPO_SFT}/ once you have access.
```

The four trained checkpoints (SFT merged + DPO/GDPO_DPO/GDPO_SFT LoRA adapters, ~3.4 GB combined) are NDA-gated; request access via `partnerships@aikium.com`. Without local checkpoints, use the Modal demo at [https://aikium--aiki-genano-fastapi-app.modal.run](https://aikium--aiki-genano-fastapi-app.modal.run).

### 2. Generate nanobody candidates for an epitope (GPU)

```bash
docker run --rm --gpus all \
    -v "$(pwd)/checkpoints:/app/checkpoints" \
    -v "$(pwd)/output:/app/output" \
    -v "$(pwd)/hf_cache:/models" \
    ghcr.io/aikium-public/aiki-genano:latest \
    generate \
        --epitope MNYPLTLEMDLENLEDLFWELDRLDNYNDTSLVENHL \
        --n_candidates 50 \
        --temperature 0.7 \
        --model GDPO_DPO \
        --output /app/output/preds.csv
```

Output `/app/output/preds.csv` columns: `epitope, generated_sequence, gen_length, is_valid_126, model, seed, temperature, reward_scaffold_integrity, reward_liability, reward_hydrophobic_patch, reward_fr2_aggregation, reward_vhh_hallmark, reward_expression`.

### 3. Generate without a GPU (laptop / Docker for Windows without CUDA passthrough)

```bash
docker run --rm \
    -v "$(pwd)/checkpoints:/app/checkpoints" \
    -v "$(pwd)/output:/app/output" \
    -v "$(pwd)/hf_cache:/models" \
    ghcr.io/aikium-public/aiki-genano:latest \
    generate \
        --epitope MNYPLTLEMDLENLEDLFWELDRLDNYNDTSLVENHL \
        --n_candidates 5 \
        --model SFT \
        --device cpu \
        --output /app/output/preds.csv
```

CPU fallback works because PyTorch picks the device at runtime; the CUDA libs in the image become dead weight but do not block execution. Expect ~5 minutes per 5 candidates instead of ~5 seconds on GPU.

### 4. Full property profile (TEMPRO + NetSolP + Sapiens, requires GPU)

```bash
docker run --rm --gpus all \
    -v "$(pwd)/checkpoints:/app/checkpoints" \
    -v "$(pwd)/output:/app/output" \
    -v "$(pwd)/hf_cache:/models" \
    ghcr.io/aikium-public/aiki-genano:latest \
    predict \
        --sequences /app/output/preds.csv \
        --with-properties \
        --output /app/output/preds_profiled.csv
```

The first `predict --with-properties` run downloads ~2 GB of foundation-model weights into `/models`; subsequent runs are warm.

### 5. Re-training on the Zenodo subset

```bash
docker run --rm --gpus all \
    -v "$(pwd)/checkpoints:/app/checkpoints" \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/output:/app/output" \
    -v "$(pwd)/hf_cache:/models" \
    ghcr.io/aikium-public/aiki-genano:latest \
    train --stage sft --config configs/sft_10k.yaml
```

(SFT and DPO each take a few GPU-hours on an A100; GDPO is ~30 min for the 2,000-step run reported in the paper. Subset training is for verification, not for matching paper headline numbers — those require the proprietary 65-target dataset.)

## What is NOT in the image

- Foundation-model weights (ProtGPT2, Sapiens, NetSolP-ESM). Download at runtime to `/models` on first invocation.
- Aikium-trained checkpoints (SFT, DPO, GDPO adapters). Pull from Zenodo via `scripts/download_checkpoints.sh`.
- Training data. The Zenodo subset is mounted at `/app/data` only when needed.

This keeps the image at ~14 GB and respects upstream redistribution licences.

## Building locally

```bash
# from the repo root
docker build -f docker/Dockerfile -t ghcr.io/aikium-public/aiki-genano:latest .
docker tag   ghcr.io/aikium-public/aiki-genano:latest \
             ghcr.io/aikium-public/aiki-genano:1.0.0
```

## Pushing to GHCR (release-time only)

```bash
echo "$GITHUB_PAT" | docker login ghcr.io -u <github-username> --password-stdin
docker push ghcr.io/aikium-public/aiki-genano:latest
docker push ghcr.io/aikium-public/aiki-genano:1.0.0

# First push lands the package as Private; flip to Public:
gh api --method PATCH /orgs/aikium-public/packages/container/aiki-genano/visibility \
       -f visibility=public
gh api /orgs/aikium-public/packages/container/aiki-genano --jq '.visibility'   # → "public"
```
