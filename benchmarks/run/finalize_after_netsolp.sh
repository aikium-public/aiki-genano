#!/usr/bin/env bash
# Waits for the NetSolP batch to finish, then adds NetSolP for ProtGPT2 and
# regenerates the final BENCHMARK_COMPARISON.md.
#
# Runs in background; reports to logs/finalize.log.
set -euo pipefail

BASE=.
LOGS="$BASE/data/generated_2026_04_24/logs"
BATCH_LOG="$LOGS/netsolp_batch.log"

log() { echo "[$(date '+%H:%M:%S')] finalize: $*"; }

# Wait for batch log to say "NetSolP batch complete" (max 10 h)
log "waiting for 'NetSolP batch complete' in $BATCH_LOG"
waited=0
until [ -f "$BATCH_LOG" ] && grep -q 'NetSolP batch complete' "$BATCH_LOG" 2>/dev/null; do
    sleep 120
    waited=$((waited+120))
    if [ $waited -ge 36000 ]; then
        log "FAILED: NetSolP batch never completed after 10 h"
        exit 2
    fi
done
log "batch finished after ${waited}s of waiting"

source ${HOME}/miniforge3/etc/profile.d/conda.sh
conda activate tempro

# Run NetSolP for ProtGPT2 (profiled CSV already exists from earlier run)
log "running add_netsolp on ProtGPT2"
PYTHONUNBUFFERED=1 python "$BASE/benchmarks/run/add_netsolp.py" \
    --csv "$BASE/data/generated_2026_04_24/protgpt2/protgpt2_seed42_temp0.7_profiled.csv" \
    --model-type ESM1b

# Regenerate comparison
log "regenerating BENCHMARK_COMPARISON.md"
python "$BASE/benchmarks/run/compare_tools.py"

log "finalize complete — BENCHMARK_COMPARISON.md updated with NetSolP columns for all tools including ProtGPT2"
