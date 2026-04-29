#!/usr/bin/env bash
#
# Build, smoke-test, and push the Aiki-GeNano image to GHCR.
#
# Per AIKI_GENANO_LAUNCH_HANDOFF.md §5: never push without a successful
# pre-push smoke test. Per anti-pattern #6: never force-push a public image
# after the LinkedIn post.
#
# Required env:
#   GITHUB_PAT       PAT with write:packages, read:packages
#   GITHUB_USER      GitHub username for `docker login`
#
# Usage:
#   scripts/build_and_push.sh                 # build + offline smoke + push :latest + :1.0.0
#   scripts/build_and_push.sh --skip-build    # use existing local image (must already be built)
#   scripts/build_and_push.sh --skip-push     # build + smoke only, no push
#   scripts/build_and_push.sh --version 1.0.1 # override version tag (default: 1.0.0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REGISTRY="ghcr.io/aikium-public/aiki-genano"
VERSION="1.0.0"
SKIP_BUILD=0
SKIP_PUSH=0

while (( "$#" )); do
    case "$1" in
        --skip-build) SKIP_BUILD=1; shift ;;
        --skip-push)  SKIP_PUSH=1;  shift ;;
        --version)    VERSION="$2"; shift 2 ;;
        -h|--help)    sed -n '2,18p' "$0"; exit 0 ;;
        *)            echo "FATAL: unknown arg '$1'" >&2; exit 2 ;;
    esac
done

cd "$REPO_ROOT"

echo "[build_and_push] target: $REGISTRY:{latest, $VERSION}"

if (( ! SKIP_BUILD )); then
    echo "[build_and_push] building image…"
    docker build -f docker/Dockerfile -t "$REGISTRY:latest" .
    docker tag "$REGISTRY:latest" "$REGISTRY:$VERSION"
fi

echo "[build_and_push] running offline smoke inside the image…"
bash scripts/docker_smoke_test.sh --offline --image "$REGISTRY:latest"

if (( SKIP_PUSH )); then
    echo "[build_and_push] --skip-push set; stopping after smoke."
    exit 0
fi

: "${GITHUB_PAT:?Missing GITHUB_PAT env var}"
: "${GITHUB_USER:?Missing GITHUB_USER env var}"

echo "[build_and_push] docker login ghcr.io as $GITHUB_USER"
echo "$GITHUB_PAT" | docker login ghcr.io -u "$GITHUB_USER" --password-stdin

echo "[build_and_push] pushing $REGISTRY:latest"
docker push "$REGISTRY:latest"
echo "[build_and_push] pushing $REGISTRY:$VERSION"
docker push "$REGISTRY:$VERSION"

# 2026-04-28: alongside the canonical $VERSION tag, also publish a
# post-sanitization alias so callers can pin to the cleaned image. Skipped
# unless ALSO_TAG is set (override with `ALSO_TAG=1.0.2` or similar).
if [ -n "${ALSO_TAG:-}" ]; then
    echo "[build_and_push] tagging + pushing $REGISTRY:$ALSO_TAG"
    docker tag "$REGISTRY:latest" "$REGISTRY:$ALSO_TAG"
    docker push "$REGISTRY:$ALSO_TAG"
fi

echo
echo "[build_and_push] PUSH OK. Now flip package visibility to Public if not already:"
echo "  gh api --method PATCH /orgs/aikium-public/packages/container/aiki-genano/visibility -f visibility=public"
echo "  gh api /orgs/aikium-public/packages/container/aiki-genano --jq '.visibility'   # → \"public\""
