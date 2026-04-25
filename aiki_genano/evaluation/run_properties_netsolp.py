"""
Run NetSolP (solubility + usability) on one seed/temp profiled CSV per model.

Follows the same approach as run_netsolp.py — converts to FASTA, calls
predict.py via subprocess, merges results back into the profiled CSV.

Reads:  {STAT_DIR}/{model}/properties/{model}_seed{s}_temp{t}_profiled.csv
Writes: same file in-place (adds netsolp_solubility, netsolp_usability)

Usage (inside Docker):
    # All 4 models, seed=42, temp=0.7
    python -m src.binder_design.protgpt2_dpo.analysis.run_properties_netsolp

    # Single model
    python -m src.binder_design.protgpt2_dpo.analysis.run_properties_netsolp --model SFT

    # Different seed/temp
    python -m src.binder_design.protgpt2_dpo.analysis.run_properties_netsolp --seed 123 --temp 0.9
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# =========================================================================
# Configuration
# =========================================================================

STAT_DIR   = Path("/app/output/statistical")
ALL_MODELS = ["SFT", "DPO", "GDPO", "GDPO_SFT"]
SEQ_COL    = "generated_sequence"

NETSOLP_DIR  = Path("/opt/netsolp")
PREDICT_PY   = NETSOLP_DIR / "predict.py"
MODELS_DIR   = NETSOLP_DIR / "models"
MODEL_TYPE   = "ESM1b"       # ESM1b: more accurate; ESM12: fastest


# =========================================================================
# Helpers (same pattern as run_netsolp.py)
# =========================================================================

def csv_to_fasta(df: pd.DataFrame, fasta_path: Path) -> None:
    """Write generated_sequence column to a FASTA file."""
    with open(fasta_path, "w") as f:
        for idx, row in df.iterrows():
            f.write(f">seq_{idx}\n{row[SEQ_COL]}\n")


def call_predict(fasta_path: Path, output_path: Path) -> None:
    """Call NetSolP predict.py via subprocess, exactly like run_netsolp.py."""
    cmd = [
        sys.executable, str(PREDICT_PY),
        "--FASTA_PATH",      str(fasta_path),
        "--OUTPUT_PATH",     str(output_path),
        "--MODELS_PATH",     str(MODELS_DIR),
        "--MODEL_TYPE",      MODEL_TYPE,
        "--PREDICTION_TYPE", "SU",
    ]
    print(f"    Running: {' '.join(cmd[-8:])}")
    subprocess.run(cmd, check=True, cwd=str(NETSOLP_DIR))


def merge_back(profiled_csv: Path, netsolp_raw: Path) -> None:
    """Merge NetSolP predictions back into the profiled CSV in-place."""
    df    = pd.read_csv(profiled_csv)
    preds = pd.read_csv(netsolp_raw)

    df["netsolp_solubility"]   = preds["predicted_solubility"].values
    df["netsolp_usability"]    = preds["predicted_usability"].values
    df["netsolp_model_type"]   = MODEL_TYPE
    df.to_csv(profiled_csv, index=False)

    sol = df["netsolp_solubility"]
    usa = df["netsolp_usability"]
    print(
        f"    sol={sol.mean():.4f} ± {sol.std():.4f}  "
        f"usa={usa.mean():.4f} ± {usa.std():.4f}"
    )


# =========================================================================
# Per-model runner
# =========================================================================

def run_model(model_name: str, seed: int, temp: float) -> None:
    props_dir    = STAT_DIR / model_name / "properties"
    profiled_csv = props_dir / f"{model_name}_seed{seed}_temp{temp}_profiled.csv"

    print(f"\n{'=' * 60}")
    print(f"MODEL: {model_name}  |  seed={seed}  temp={temp}")
    print(f"{'=' * 60}")

    if not profiled_csv.exists():
        print(f"  [SKIP] profiled CSV not found: {profiled_csv}")
        return

    # Check if already done (skip only if same model type was used)
    df_check = pd.read_csv(profiled_csv, nrows=1)
    if (
        "netsolp_solubility" in df_check.columns
        and "netsolp_usability" in df_check.columns
        and df_check.get("netsolp_model_type", pd.Series([None]))[0] == MODEL_TYPE
    ):
        print(f"  [skip] already computed with {MODEL_TYPE} — skipping.")
        return

    if not PREDICT_PY.exists():
        print(f"  [SKIP] NetSolP predict.py not found: {PREDICT_PY}")
        return

    df = pd.read_csv(profiled_csv)
    print(f"  {len(df)} sequences → FASTA …")

    # Temp files alongside the profiled CSV
    fasta_path  = props_dir / f"{model_name}_seed{seed}_temp{temp}.fasta"
    netsolp_raw = props_dir / f"{model_name}_seed{seed}_temp{temp}_netsolp_raw.csv"

    t0 = time.time()
    csv_to_fasta(df, fasta_path)
    call_predict(fasta_path, netsolp_raw)
    merge_back(profiled_csv, netsolp_raw)

    # Clean up temp files
    fasta_path.unlink(missing_ok=True)
    netsolp_raw.unlink(missing_ok=True)

    print(f"  Done in {time.time() - t0:.0f}s  →  {profiled_csv.name}")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run NetSolP on one seed/temp profiled CSV per model.",
    )
    parser.add_argument("--model", default="all",
                        choices=ALL_MODELS + ["all"],
                        help="Which model(s) to process (default: all).")
    parser.add_argument("--seed",  type=int,   default=42,
                        help="Seed to target (default: 42).")
    parser.add_argument("--temp",  type=float, default=0.7,
                        help="Temperature to target (default: 0.7).")
    args = parser.parse_args()

    models = ALL_MODELS if args.model == "all" else [args.model]

    for model_name in models:
        run_model(model_name, args.seed, args.temp)

    print("\nDone.")


if __name__ == "__main__":
    main()
