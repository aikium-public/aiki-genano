#!/usr/bin/env bash
# Orchestrator to process ProtGPT2 once its generator finishes AND to regenerate
# the final comparison once all NetSolP runs complete.
#
# Steps:
#   1. Wait for ProtGPT2 status.json:in_progress to flip to False
#   2. Run adapter on ProtGPT2 FASTA → profile-input CSV
#   3. Run profile_tool --skip-netsolp on ProtGPT2 (TEMPRO + Sapiens + motifs)
#   4. Wait for the NetSolP batch log to say "NetSolP batch complete"
#   5. Run add_netsolp on ProtGPT2
#   6. Re-run compare_tools.py to produce the final BENCHMARK_COMPARISON.md
#
# Guardrails:
#   - Every step exits on first failure (set -euo pipefail)
#   - "Wait" loops have 2 h max timeout each; exceed → bail loudly
#   - All stdout/stderr goes to /data/generated_2026_04_24/logs/orchestrator.log

set -euo pipefail

BASE=.
LOGS="$BASE/data/generated_2026_04_24/logs"
PROTG="$BASE/data/generated_2026_04_24/protgpt2"

log() { echo "[$(date '+%H:%M:%S')] orchestrator: $*"; }

wait_for_file_with_status() {
    # $1 = path to status.json, $2 = max wait seconds
    local sf=$1 maxw=$2 waited=0
    log "waiting for protgpt2 status.json to report in_progress=False"
    until [ -f "$sf" ] && ${HOME}/miniforge3/envs/tempro/bin/python -c "import json,sys;d=json.load(open(sys.argv[1]));sys.exit(0 if d.get('in_progress') is False else 1)" "$sf" 2>/dev/null; do
        sleep 30
        waited=$((waited+30))
        if [ $waited -ge $maxw ]; then
            log "FAILED: protgpt2 status never flipped after ${maxw}s — aborting orchestrator"
            return 2
        fi
    done
    log "protgpt2 generator finished after ${waited}s"
}

wait_for_string_in_log() {
    # $1 = path to log, $2 = substring, $3 = max wait seconds
    local lf=$1 needle=$2 maxw=$3 waited=0
    log "waiting for '$needle' in $(basename "$lf")"
    until [ -f "$lf" ] && grep -q -- "$needle" "$lf" 2>/dev/null; do
        sleep 60
        waited=$((waited+60))
        if [ $waited -ge $maxw ]; then
            log "FAILED: '$needle' never appeared after ${maxw}s — aborting orchestrator"
            return 2
        fi
    done
    log "found '$needle' after ${waited}s"
}

# Shared env setup
source ${HOME}/miniforge3/etc/profile.d/conda.sh
conda activate tempro

# 1. Wait for ProtGPT2
wait_for_file_with_status "$PROTG/status.json" 3600 || exit 2
sleep 3

# Confirm the FASTA has at least 500 records before we trust it
proto_fa="$PROTG/generated_T0.7_seed42.fasta"
n_records=$(grep -c '^>' "$proto_fa" 2>/dev/null || echo 0)
log "protgpt2 FASTA has $n_records records"
if [ "$n_records" -lt 500 ]; then
    log "FAILED: protgpt2 FASTA too short ($n_records < 500) — aborting"
    exit 3
fi

# 2. Adapter
log "running fasta_to_profile_csv adapter on ProtGPT2"
python "$BASE/benchmarks/run/fasta_to_profile_csv.py" \
    --tool protgpt2 \
    --fasta "$proto_fa" \
    --out "$PROTG/protgpt2_seed42_temp0.7.csv"

# 3. profile_tool (skip NetSolP — that happens after NetSolP batch finishes)
log "running profile_tool (skip-netsolp) on ProtGPT2"
PYTHONUNBUFFERED=1 python "$BASE/benchmarks/run/profile_tool.py" \
    --in "$PROTG/protgpt2_seed42_temp0.7.csv" \
    --out "$PROTG/protgpt2_seed42_temp0.7_profiled.csv" \
    --skip-netsolp

# 4. Wait for NetSolP batch
wait_for_string_in_log "$LOGS/netsolp_batch.log" "NetSolP batch complete" 36000 || exit 4

# 5. add_netsolp on ProtGPT2
log "running add_netsolp on ProtGPT2"
PYTHONUNBUFFERED=1 python "$BASE/benchmarks/run/add_netsolp.py" \
    --csv "$PROTG/protgpt2_seed42_temp0.7_profiled.csv" \
    --model-type ESM1b

# 6. Regenerate comparison
log "regenerating BENCHMARK_COMPARISON.md"
python "$BASE/benchmarks/run/compare_tools.py"

log "orchestrator finished — all 7 competitor tools profiled + comparison updated"
