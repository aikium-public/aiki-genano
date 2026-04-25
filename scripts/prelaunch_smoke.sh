#!/usr/bin/env bash
#
# Pre-launch smoke test for the deployed Modal app. Hits every public route,
# asserts payload shape, prints PASS/FAIL. Per AIKI_GENANO_LAUNCH_HANDOFF
# anti-pattern #7: don't claim "ready" without running this.
#
# Usage:
#   scripts/prelaunch_smoke.sh                                            # auto-detects URL via `modal app list`
#   scripts/prelaunch_smoke.sh --url https://...modal.run                 # explicit URL
#   scripts/prelaunch_smoke.sh --url https://...modal.run --skip-generate # omit GPU call (cold-start cost)

set -euo pipefail

URL=""
SKIP_GENERATE=0

while (( "$#" )); do
    case "$1" in
        --url)            URL="$2"; shift 2 ;;
        --skip-generate)  SKIP_GENERATE=1; shift ;;
        -h|--help)        sed -n '2,12p' "$0"; exit 0 ;;
        *)                echo "FATAL: unknown arg '$1'" >&2; exit 2 ;;
    esac
done

if [[ -z "$URL" ]]; then
    URL="$(modal app list --json 2>/dev/null \
        | python3 -c 'import sys,json;print(next((a["web_url"] for a in json.load(sys.stdin) if a["name"]=="aiki-genano"), ""))')" \
        || true
fi

[[ -n "$URL" ]] || { echo "FATAL: could not determine app URL; pass --url" >&2; exit 2; }
URL="${URL%/}"
echo "[prelaunch] target: $URL"

PASS=0; FAIL=0
report() { if [[ "$1" == "PASS" ]]; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi; echo "  [$1] $2"; }

# ── 1. Landing returns HTML ──────────────────────────────────────────────────
body="$(curl -fsSL "$URL/" || true)"
[[ "$body" == *"Aiki-GeNano"* ]] && report PASS "GET /"   || report FAIL "GET /"

# ── 2. /api/health returns expected fields ───────────────────────────────────
hjson="$(curl -fsSL "$URL/api/health" || true)"
ok="$(printf '%s' "$hjson" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(int(all(k in d for k in ("status","version","models","sentinel_reward_on_reference"))))' 2>/dev/null || echo 0)"
[[ "$ok" == "1" ]] && report PASS "GET /api/health" || report FAIL "GET /api/health: $hjson"

# ── 3. /api/score returns rows for one sequence ──────────────────────────────
ref="QVQLVESGGGSVQAGGSLRLSCTASGGSEYSYSTFSLGWFRQAPGQEREAVAAIASMGGLTYYADSVKGRFTISRDNAKNTVTLQMNNLKPVDTAIYYCAAVRGYFMRLPSWGQGTQVTVSGGGGS"
sjson="$(curl -fsSL -X POST "$URL/api/score" -H 'Content-Type: application/json' \
    -d "{\"sequences\":[\"$ref\"]}" || true)"
ok="$(printf '%s' "$sjson" | python3 -c 'import json,sys;d=json.load(sys.stdin);r=d.get("scored",[]);print(int(d.get("n_returned")==1 and r and "reward_scaffold_integrity" in r[0]))' 2>/dev/null || echo 0)"
[[ "$ok" == "1" ]] && report PASS "POST /api/score" || report FAIL "POST /api/score: $sjson"

# ── 4. /api/generate (GPU, expensive — opt out with --skip-generate) ─────────
if (( SKIP_GENERATE )); then
    report PASS "POST /api/generate (skipped)"
else
    gjson="$(curl -fsSL -X POST "$URL/api/generate" -H 'Content-Type: application/json' \
        -d '{"epitope":"MNYPLTLEMDLENLEDLFWELDRLDNYNDTSLVENHL","n_candidates":3,"model":"GDPO_DPO","seed":42}' || true)"
    ok="$(printf '%s' "$gjson" | python3 -c 'import json,sys;d=json.load(sys.stdin);c=d.get("candidates",[]);print(int(d.get("n_returned")==3 and len(c)==3 and "reward_scaffold_integrity" in c[0]))' 2>/dev/null || echo 0)"
    [[ "$ok" == "1" ]] && report PASS "POST /api/generate" || report FAIL "POST /api/generate: $gjson"
fi

echo
echo "[prelaunch] $PASS pass, $FAIL fail"
exit "$(( FAIL > 0 ))"
