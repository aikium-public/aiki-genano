"""
Utilities for the Yotta Nb-v1 DPO developability dataset.

This repo's DPO loader (`src/binder_design/protgpt2_dpo/data_utils.py`) expects a directory with:
  - training.csv
  - testing.csv

and a row schema with columns:
  - peptide  : prompt input (target)
  - protein  : chosen output (preferred binder)
  - decoy    : rejected output (non-preferred binder)

The Yotta Nb-v1 CSV uses:
  - target
  - chosen_binder
  - rejected_binder

This script previews the dataset and (optionally) converts it into the expected train/test CSVs.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class YottaColumns:
    target: str = "target"
    chosen: str = "chosen_binder"
    rejected: str = "rejected_binder"


def preview(csv_path: str, n: int = 3) -> None:
    df = pd.read_csv(csv_path, nrows=n)
    print(f"csv: {csv_path}")
    print("columns:", list(df.columns))
    print()
    print(df.to_string(index=False))


def convert_to_repo_dpo_splits(
    csv_path: str,
    out_dir: str,
    test_frac: float = 0.01,
    seed: int = 0,
    limit_rows: int | None = None,
    cols: YottaColumns = YottaColumns(),
) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path, nrows=limit_rows)
    needed = [cols.target, cols.chosen, cols.rejected]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}. Found: {list(df.columns)}")

    out = pd.DataFrame(
        {
            # Match SFT direction: peptide/target -> binder/protein
            "peptide": df[cols.target],
            "protein": df[cols.chosen],
            "decoy": df[cols.rejected],
        }
    )

    # simple random split (fast + good enough for DPO debugging)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_test = max(1, int(len(out) * test_frac))
    test_df = out.iloc[:n_test].copy()
    train_df = out.iloc[n_test:].copy()

    train_path = os.path.join(out_dir, "training.csv")
    test_path = os.path.join(out_dir, "testing.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Wrote: {train_path} ({len(train_df)} rows)")
    print(f"Wrote: {test_path} ({len(test_df)} rows)")
    return train_path, test_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default="yotta10k_v1_dpo_developability.csv",
        help="Path to the raw yotta CSV (relative to this directory by default).",
    )
    ap.add_argument(
        "--out_dir",
        default="splits_repo_format",
        help="Output directory for training.csv/testing.csv (relative to this directory by default).",
    )
    ap.add_argument("--preview", action="store_true", help="Print columns and a few example rows.")
    ap.add_argument("--preview_n", type=int, default=3)
    ap.add_argument("--make_splits", action="store_true", help="Write training.csv/testing.csv in repo format.")
    ap.add_argument("--test_frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--limit_rows",
        type=int,
        default=None,
        help="If set, only load the first N rows from the raw CSV (useful for fast smoke tests).",
    )
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(here, args.csv)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(here, args.out_dir)

    if args.preview:
        preview(csv_path, n=args.preview_n)

    if args.make_splits:
        convert_to_repo_dpo_splits(
            csv_path,
            out_dir,
            test_frac=args.test_frac,
            seed=args.seed,
            limit_rows=args.limit_rows,
        )

    if not args.preview and not args.make_splits:
        # Default behavior: preview only (so "run dataset.py" is useful)
        preview(csv_path, n=args.preview_n)


if __name__ == "__main__":
    main()
