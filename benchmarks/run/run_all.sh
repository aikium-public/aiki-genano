#!/usr/bin/env bash
# Orchestrator. Runs every generator fastest -> slowest, idempotent per tool.
#   bash run_all.sh          # full run, N=100
#   N=16 bash run_all.sh     # quick sanity run

set -u
cd "$(dirname "$0")"

PY=python
TARGETS=./data/2026_04_23__genano__targets.csv
OUT=./data/generated_2026_04_24
N=${N:-100}
T=${T:-0.7}
SEED=${SEED:-42}
STOP_ON_ERROR=${STOP_ON_ERROR:-0}

export HF_HUB_DISABLE_TELEMETRY=1
export HF_TOKEN=${HF_TOKEN:-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}

mkdir -p "$OUT/logs"

run_step() {
    local name="$1"; shift
    local log="$OUT/logs/${name}.log"
    echo "[orch] ==== $name  (log: $log) ====" | tee -a "$OUT/logs/orchestrator.log"
    local t0=$(date +%s)
    "$@" >"$log" 2>&1
    local rc=$?
    local dt=$(( $(date +%s) - t0 ))
    echo "[orch] $name rc=$rc ${dt}s" | tee -a "$OUT/logs/orchestrator.log"
    if [ "$rc" != 0 ] && [ "$STOP_ON_ERROR" = "1" ]; then
        echo "[orch] STOP_ON_ERROR=1 -> aborting" | tee -a "$OUT/logs/orchestrator.log"
        exit $rc
    fi
    return $rc
}

run_step nanobert    "$PY" run_nanobert.py    --targets "$TARGETS" --out-dir "$OUT/nanobert"    --n "$N" --T "$T" --seed "$SEED"
run_step iglm        "$PY" run_iglm.py        --targets "$TARGETS" --out-dir "$OUT/iglm"        --n "$N" --T "$T" --seed "$SEED"
run_step protgpt2    "$PY" run_protgpt2.py    --targets "$TARGETS" --out-dir "$OUT/protgpt2"    --n "$N" --T "$T" --seed "$SEED"
run_step pepmlm      "$PY" run_pepmlm.py      --targets "$TARGETS" --out-dir "$OUT/pepmlm"      --n "$N" --T "$T" --seed "$SEED"
run_step nanoabllama "$PY" run_nanoabllama.py --targets "$TARGETS" --out-dir "$OUT/nanoabllama" --n "$N" --T "$T" --seed "$SEED"

run_step esmfold_prefold "$PY" prefold_esmfold.py --targets "$TARGETS" --out-dir "$OUT/esmfold_prefolds"

BACKBONE=${IGGM_REPO:-/opt/IgGM}/examples/pdb.files.native/8q95_B_NA_A.pdb
run_step proteindpo "$PY" run_proteindpo.py --targets "$TARGETS" --out-dir "$OUT/proteindpo" \
    --backbone-pdb "$BACKBONE" --chain H --n "$N" --T "$T" --seed "$SEED"

run_step iggm "$PY" run_iggm.py --targets "$TARGETS" --out-dir "$OUT/iggm" \
    --prefold-dir "$OUT/esmfold_prefolds" --n "$N" --seed "$SEED"

echo "[orch] ==== DONE ====" | tee -a "$OUT/logs/orchestrator.log"
