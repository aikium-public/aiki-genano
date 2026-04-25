#!/usr/bin/env bash
#
# Pull Aiki-GeNano model checkpoints from Zenodo.
#
# Manifest-driven (scripts/zenodo_manifest.json) per decision D4 in
# data/_internal_planning/PLANNING_DOCKER_MODAL.md — every URL is verified
# against zenodo.org once during manifest creation, and every download is
# sha256-checked here to catch silent corruption / DOI swaps.
#
# Usage:
#   scripts/download_checkpoints.sh                     # all 4 checkpoints into ./checkpoints/
#   scripts/download_checkpoints.sh --dest /custom/dir  # alternate destination
#   scripts/download_checkpoints.sh --only SFT,GDPO_DPO # subset
#
# Requires: bash, curl, sha256sum, tar, jq.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MANIFEST="${SCRIPT_DIR}/zenodo_manifest.json"

DEST="${REPO_ROOT}/checkpoints"
ONLY=""

while (( "$#" )); do
    case "$1" in
        --dest)  DEST="$2"; shift 2 ;;
        --only)  ONLY="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,18p' "$0"
            exit 0
            ;;
        *)
            echo "FATAL: unknown arg '$1'" >&2
            exit 2
            ;;
    esac
done

for tool in curl sha256sum tar jq; do
    command -v "$tool" >/dev/null 2>&1 || { echo "FATAL: required tool '$tool' not found"; exit 2; }
done
[[ -f "$MANIFEST" ]] || { echo "FATAL: manifest not found at $MANIFEST"; exit 2; }

record_doi="$(jq -r .record_doi "$MANIFEST")"
record_url="$(jq -r .record_url "$MANIFEST")"
echo "[manifest] Zenodo record: $record_doi ($record_url)"

if [[ "$record_doi" == *PLACEHOLDER* ]]; then
    cat >&2 <<EOF
FATAL: zenodo_manifest.json still has PLACEHOLDER DOI.
       The Zenodo deposit hasn't been published yet. Bump the manifest after
       upload (record_doi, record_url, per-checkpoint url + sha256) and re-run.
EOF
    exit 3
fi

mkdir -p "$DEST"

mapfile -t names < <(jq -r '.checkpoints | keys_unsorted[]' "$MANIFEST")
if [[ -n "$ONLY" ]]; then
    IFS=',' read -ra requested <<< "$ONLY"
    filtered=()
    for n in "${requested[@]}"; do
        ok=0
        for k in "${names[@]}"; do [[ "$k" == "$n" ]] && ok=1 && break; done
        [[ $ok -eq 1 ]] || { echo "FATAL: --only listed unknown checkpoint '$n'"; exit 2; }
        filtered+=("$n")
    done
    names=("${filtered[@]}")
fi

for name in "${names[@]}"; do
    url="$(jq -r ".checkpoints.\"$name\".url"    "$MANIFEST")"
    sha="$(jq -r ".checkpoints.\"$name\".sha256" "$MANIFEST")"
    sub="$(jq -r ".checkpoints.\"$name\".subdir" "$MANIFEST")"
    sz="$( jq -r ".checkpoints.\"$name\".approx_size_mb" "$MANIFEST")"

    out_tar="${DEST}/${name}.tar.gz"
    out_dir="${DEST}/${name}"

    echo
    echo "[$name] (~${sz} MB) → $out_dir"

    if [[ -d "$out_dir" && -n "$(ls -A "$out_dir" 2>/dev/null)" ]]; then
        echo "  already present, skipping (delete to re-download)"
        continue
    fi

    echo "  downloading $url"
    curl --fail --location --progress-bar -o "$out_tar" "$url"

    actual="$(sha256sum "$out_tar" | awk '{print $1}')"
    if [[ "$actual" != "$sha" ]]; then
        echo "FATAL: sha256 mismatch for $name" >&2
        echo "  expected: $sha" >&2
        echo "  actual:   $actual" >&2
        echo "  Either Zenodo corruption, MITM, or the manifest is stale. Investigate before proceeding." >&2
        rm -f "$out_tar"
        exit 4
    fi
    echo "  sha256 verified"

    mkdir -p "$out_dir"
    tar -xzf "$out_tar" -C "$out_dir" --strip-components=0
    rm -f "$out_tar"
done

echo
echo "All requested checkpoints downloaded to $DEST"
