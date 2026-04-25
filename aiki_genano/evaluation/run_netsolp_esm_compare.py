"""
Compare ESM12 vs ESM1b NetSolP predictions on the original ~640-seq profiled CSVs.

- Reads the 3 original profiled CSVs (SFT, DPO, GDPO — ~640 seqs each)
- Loads existing ESM12 raw results (already computed)
- Runs ESM1b on the same sequences (saves to *_esm1b_raw.csv)
- Prints a side-by-side comparison table

Usage (inside Docker):
    python -m src.binder_design.protgpt2_dpo.analysis.run_netsolp_esm_compare

    # Skip re-running ESM1b if already done:
    python -m src.binder_design.protgpt2_dpo.analysis.run_netsolp_esm_compare --no-rerun

    # Run only ESM1b for specific model:
    python -m src.binder_design.protgpt2_dpo.analysis.run_netsolp_esm_compare --model SFT
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import tempfile
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
CSV_DIR     = Path("/app/output/csv")
NETSOLP_DIR = Path("/opt/netsolp")
PREDICT_PY  = NETSOLP_DIR / "predict.py"
MODELS_DIR  = NETSOLP_DIR / "models"

# (label, profiled_csv, existing_esm12_raw)
TARGETS = [
    ("SFT",  "SFT_20k_profiled.csv",                   "SFT_20k_netsolp_raw.csv"),
    ("DPO",  "DPO_6k_profiled.csv",                    "DPO_6k_netsolp_raw.csv"),
    ("GDPO", "GDPO_dpo_final_gated_profiled.csv",       "GDPO_dpo_final_gated_netsolp_raw.csv"),
]

SEQ_COL = "generated_sequence"


# ── Helpers ────────────────────────────────────────────────────────────────

def write_fasta(df: pd.DataFrame, path: Path) -> None:
    with open(path, "w") as fh:
        for idx, row in df.iterrows():
            fh.write(f">seq_{idx}\n{row[SEQ_COL]}\n")


def run_netsolp(fasta: Path, out: Path, model_type: str) -> None:
    cmd = [
        sys.executable, str(PREDICT_PY),
        "--FASTA_PATH",      str(fasta),
        "--OUTPUT_PATH",     str(out),
        "--MODELS_PATH",     str(MODELS_DIR),
        "--MODEL_TYPE",      model_type,
        "--PREDICTION_TYPE", "SU",
    ]
    print(f"    [{model_type}] running predict.py …")
    subprocess.run(cmd, check=True, cwd=str(NETSOLP_DIR))


def summary_stats(df: pd.DataFrame, col: str) -> str:
    v = df[col]
    return f"{v.mean():.4f} ± {v.std():.4f}  [{v.min():.3f}, {v.max():.3f}]"


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="all",
                        choices=["SFT","DPO","GDPO","all"])
    parser.add_argument("--no-rerun", action="store_true",
                        help="Skip ESM1b inference if *_esm1b_raw.csv already exists.")
    args = parser.parse_args()

    targets = [t for t in TARGETS if args.model == "all" or t[0] == args.model]

    rows = []
    for label, profiled_fname, esm12_fname in targets:
        profiled_csv = CSV_DIR / profiled_fname
        esm12_raw    = CSV_DIR / esm12_fname
        esm1b_raw    = CSV_DIR / esm12_fname.replace("_raw.csv", "_esm1b_raw.csv")

        print(f"\n{'='*60}")
        print(f"  MODEL: {label}")
        print(f"{'='*60}")

        if not profiled_csv.exists():
            print(f"  [SKIP] profiled CSV not found: {profiled_csv}")
            continue

        df = pd.read_csv(profiled_csv)
        valid = df[df["is_valid_126"] == True] if "is_valid_126" in df.columns else df
        print(f"  {len(df)} total rows | {len(valid)} valid seqs")

        # ── ESM12 (already computed) ──────────────────────────────────────
        row: dict = {"model": label}
        if esm12_raw.exists():
            esm12_df = pd.read_csv(esm12_raw)
            row["esm12_sol_mean"]  = esm12_df["predicted_solubility"].mean()
            row["esm12_sol_std"]   = esm12_df["predicted_solubility"].std()
            row["esm12_sol_min"]   = esm12_df["predicted_solubility"].min()
            row["esm12_sol_max"]   = esm12_df["predicted_solubility"].max()
            row["esm12_usa_mean"]  = esm12_df["predicted_usability"].mean()
            row["esm12_usa_std"]   = esm12_df["predicted_usability"].std()
            print(f"  ESM12 sol  : {summary_stats(esm12_df, 'predicted_solubility')}")
            print(f"  ESM12 usa  : {summary_stats(esm12_df, 'predicted_usability')}")
        else:
            print(f"  [WARN] ESM12 raw not found: {esm12_raw}")

        # ── ESM1b (run or load) ───────────────────────────────────────────
        if esm1b_raw.exists() and args.no_rerun:
            print(f"  ESM1b raw found — loading (--no-rerun)")
        else:
            if not PREDICT_PY.exists():
                print(f"  [SKIP] NetSolP predict.py not found at {PREDICT_PY}")
                rows.append(row)
                continue

            with tempfile.NamedTemporaryFile(suffix=".fasta", delete=False) as tmp:
                fasta_path = Path(tmp.name)
            write_fasta(df, fasta_path)

            t0 = time.time()
            run_netsolp(fasta_path, esm1b_raw, model_type="ESM1b")
            fasta_path.unlink(missing_ok=True)
            print(f"  ESM1b done in {time.time()-t0:.0f}s  →  {esm1b_raw.name}")

        if esm1b_raw.exists():
            esm1b_df = pd.read_csv(esm1b_raw)
            row["esm1b_sol_mean"] = esm1b_df["predicted_solubility"].mean()
            row["esm1b_sol_std"]  = esm1b_df["predicted_solubility"].std()
            row["esm1b_sol_min"]  = esm1b_df["predicted_solubility"].min()
            row["esm1b_sol_max"]  = esm1b_df["predicted_solubility"].max()
            row["esm1b_usa_mean"] = esm1b_df["predicted_usability"].mean()
            row["esm1b_usa_std"]  = esm1b_df["predicted_usability"].std()
            print(f"  ESM1b sol  : {summary_stats(esm1b_df, 'predicted_solubility')}")
            print(f"  ESM1b usa  : {summary_stats(esm1b_df, 'predicted_usability')}")

        rows.append(row)

    # ── Print side-by-side comparison table ───────────────────────────────
    if not rows:
        print("\nNo data collected.")
        return

    print("\n" + "="*80)
    print("COMPARISON  —  ESM12 vs ESM1b  (mean ± std)")
    print("="*80)

    # Solubility
    print(f"\n{'Solubility':}")
    hdr = f"  {'Model':<8}  {'ESM12 mean±std':>22}  {'ESM1b mean±std':>22}  {'Delta (1b-12)':>14}"
    print(hdr)
    print("  " + "-" * (8 + 22 + 22 + 14 + 10))
    for r in rows:
        e12 = f"{r.get('esm12_sol_mean', float('nan')):.4f} ± {r.get('esm12_sol_std', float('nan')):.4f}"
        e1b = f"{r.get('esm1b_sol_mean', float('nan')):.4f} ± {r.get('esm1b_sol_std', float('nan')):.4f}"
        delta = r.get('esm1b_sol_mean', float('nan')) - r.get('esm12_sol_mean', float('nan'))
        print(f"  {r['model']:<8}  {e12:>22}  {e1b:>22}  {delta:>+14.4f}")

    # Usability
    print(f"\n{'Usability (E.coli expression)':}")
    print(hdr)
    print("  " + "-" * (8 + 22 + 22 + 14 + 10))
    for r in rows:
        e12 = f"{r.get('esm12_usa_mean', float('nan')):.4f} ± {r.get('esm12_usa_std', float('nan')):.4f}"
        e1b = f"{r.get('esm1b_usa_mean', float('nan')):.4f} ± {r.get('esm1b_usa_std', float('nan')):.4f}"
        delta = r.get('esm1b_usa_mean', float('nan')) - r.get('esm12_usa_mean', float('nan'))
        print(f"  {r['model']:<8}  {e12:>22}  {e1b:>22}  {delta:>+14.4f}")

    # Trend check
    print("\n" + "-"*80)
    print("Trend check (higher = better for both solubility and usability):")
    for metric, key in [("Solubility", "sol"), ("Usability", "usa")]:
        vals = [(r["model"], r.get(f"esm12_{key}_mean"), r.get(f"esm1b_{key}_mean")) for r in rows]
        esm12_order = sorted(vals, key=lambda x: x[1] or 0, reverse=True)
        esm1b_order = sorted(vals, key=lambda x: x[2] or 0, reverse=True)
        esm12_str = " > ".join(m for m, _, __ in esm12_order)
        esm1b_str = " > ".join(m for m, _, __ in esm1b_order)
        match = "✓ SAME" if esm12_str == esm1b_str else "✗ DIFFERENT"
        print(f"  {metric:<12}  ESM12: {esm12_str:<25}  ESM1b: {esm1b_str:<25}  {match}")

    # Save comparison
    out_csv = CSV_DIR / "netsolp_esm_comparison.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nComparison saved → {out_csv}")


if __name__ == "__main__":
    main()
