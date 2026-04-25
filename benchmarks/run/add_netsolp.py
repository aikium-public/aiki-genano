"""
Run NetSolP on an already-profiled CSV and merge netsolp_* columns back.

Used to avoid re-running the fast CPU/GPU steps (local profile, Sapiens, TEMPRO)
when we just need to add NetSolP predictions (which take ~60 min / 1000 seqs).

Guardrails:
- Pinned column names ('predicted_solubility', 'predicted_usability'). No fuzzy match.
- 1:1 sid alignment via int-extracted join; hard-assert all rows matched.
- Hard-abort if existing NetSolP columns are present with a different
  model_type (refuse to silently overwrite different-model-type values).

Usage:
    python benchmarks/run/add_netsolp.py \\
        --csv data/generated_2026_04_24/nanobert/nanobert_seed42_temp0.7_profiled.csv \\
        --model-type ESM1b
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
NETSOLP_DIR = Path("/opt/netsolp")


def fatal(msg: str):
    sys.stderr.write(f"\nFATAL: {msg}\n")
    sys.exit(2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--model-type", default="ESM1b", choices=["ESM1b", "ESM12", "Distilled"])
    args = p.parse_args()

    if not args.csv.exists():
        fatal(f"CSV not found: {args.csv}")
    df = pd.read_csv(args.csv)
    if "generated_sequence" not in df.columns:
        fatal("CSV missing 'generated_sequence' column")

    # If netsolp_model_type already present and non-null, confirm it matches --model-type
    existing_mt = df.get("netsolp_model_type")
    if existing_mt is not None and existing_mt.notna().any() and existing_mt.dropna().iloc[0] not in ("SKIPPED", args.model_type):
        fatal(f"CSV already has netsolp_model_type={existing_mt.dropna().iloc[0]}; refuse to overwrite "
              f"with {args.model_type} without explicit intent. Remove netsolp_* columns first.")

    # Write fasta, run NetSolP
    work_dir = args.csv.parent / "netsolp_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    fasta = work_dir / f"netsolp_input_{args.model_type}.fasta"
    out_csv = work_dir / f"netsolp_output_{args.model_type}.csv"
    if out_csv.exists():
        out_csv.unlink()
    with open(fasta, "w") as fh:
        for i in range(len(df)):
            fh.write(f">s{i}\n{df['generated_sequence'].iloc[i]}\n")

    cmd = [
        sys.executable, str(NETSOLP_DIR / "predict.py"),
        "--FASTA_PATH", str(fasta),
        "--OUTPUT_PATH", str(out_csv),
        "--MODELS_PATH", str(NETSOLP_DIR / "models"),
        "--MODEL_TYPE", args.model_type,
        "--PREDICTION_TYPE", "SU",
    ]
    print(f"[add_netsolp] {args.csv.name} ({len(df)} seqs) — {args.model_type}")
    print(f"[add_netsolp] cmd: {' '.join(cmd[-8:])}")
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=str(NETSOLP_DIR))
    print(f"[add_netsolp] NetSolP done in {time.time()-t0:.0f}s")

    res = pd.read_csv(out_csv)
    required = {"sid", "predicted_solubility", "predicted_usability"}
    missing = required - set(res.columns)
    if missing:
        fatal(f"NetSolP output missing expected columns: {missing}; got {list(res.columns)}")

    ord_series = res["sid"].str.extract(r"^s(\d+)$")[0]
    if ord_series.isna().any():
        fatal(f"NetSolP returned sid not matching 's\\d+': "
              f"{res['sid'][ord_series.isna()].head().tolist()}")
    res = res.assign(__order=ord_series.astype(int)).sort_values("__order").reset_index(drop=True)
    if len(res) != len(df):
        fatal(f"NetSolP returned {len(res)} rows for {len(df)} input")
    if not (res["__order"].values == np.arange(len(df))).all():
        fatal("NetSolP sid sequence has gaps after sort")

    df["netsolp_solubility"] = res["predicted_solubility"].values
    df["netsolp_usability"] = res["predicted_usability"].values
    df["netsolp_model_type"] = args.model_type

    df.to_csv(args.csv, index=False)
    sol_mean = df["netsolp_solubility"].mean()
    usa_mean = df["netsolp_usability"].mean()
    print(f"[add_netsolp] wrote {args.csv.name}: sol={sol_mean:.4f}  usa={usa_mean:.4f}")


if __name__ == "__main__":
    main()
