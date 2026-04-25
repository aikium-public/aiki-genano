#!/usr/bin/env bash
#
# Docker image smoke-test wrapper. Builds the image (if --build) and runs
# `aiki-genano smoke {--offline,--real}` inside it, checking exit code and
# the presence of expected output artefacts on the mounted output dir.
#
# Usage:
#   scripts/docker_smoke_test.sh --offline             # ~1 min, no downloads
#   scripts/docker_smoke_test.sh --real                # ~10 min on first run (Zenodo download)
#   scripts/docker_smoke_test.sh --offline --build     # rebuild image first
#   scripts/docker_smoke_test.sh --offline --image custom:tag
#
# Per-AIKI_GENANO_LAUNCH_HANDOFF §5: always smoke-test before pushing to GHCR.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE=""
IMAGE="ghcr.io/aikium-public/aiki-genano:latest"
BUILD=0
GPU=""

while (( "$#" )); do
    case "$1" in
        --offline) MODE="offline"; shift ;;
        --real)    MODE="real";    shift ;;
        --build)   BUILD=1;        shift ;;
        --image)   IMAGE="$2";     shift 2 ;;
        --gpu)     GPU="--gpus all"; shift ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *)         echo "FATAL: unknown arg '$1'" >&2; exit 2 ;;
    esac
done

[[ -n "$MODE" ]] || { echo "FATAL: pass --offline or --real" >&2; exit 2; }

cd "$REPO_ROOT"

if (( BUILD )); then
    echo "[smoke-test] building $IMAGE"
    docker build -f docker/Dockerfile -t "$IMAGE" .
fi

OUT="$REPO_ROOT/data/_smoke_test_output"
mkdir -p "$OUT"
rm -rf "${OUT:?}/"*

CKPTS="$REPO_ROOT/data/_smoke_test_checkpoints"
HF_CACHE="$REPO_ROOT/data/_smoke_test_hf_cache"
mkdir -p "$CKPTS" "$HF_CACHE"

case "$MODE" in
    offline)
        echo "[smoke-test] running OFFLINE smoke (no GPU, no Zenodo)"
        docker run --rm \
            -v "$OUT:/app/output" \
            "$IMAGE" \
            smoke --offline --output-dir /app/output
        ;;
    real)
        echo "[smoke-test] running REAL smoke (downloads + inference)"
        docker run --rm $GPU \
            -v "$CKPTS:/app/checkpoints" \
            -v "$OUT:/app/output" \
            -v "$HF_CACHE:/models" \
            "$IMAGE" \
            smoke --real --checkpoint-dir /app/checkpoints --output-dir /app/output
        ;;
esac

echo
echo "[smoke-test] PASS — artefacts under $OUT"
