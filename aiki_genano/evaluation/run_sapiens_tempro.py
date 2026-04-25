"""
Run Sapiens humanness + TEMPRO Tm prediction on generated nanobody CSVs.

Usage (inside Docker):
    pip install sapiens
    python -m src.binder_design.protgpt2_dpo.analysis.run_sapiens_tempro
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CSV_DIR = Path("/app/output/csv")
MODELS = ["SFT_20k", "DPO_6k", "GDPO_dpo_final_gated"]


def compute_sapiens_humanness(sequences: list[str]) -> list[float]:
    """Compute Sapiens humanness score for each sequence.

    Score = mean predicted probability of the actual residue at each position,
    using the Sapiens VH language model. Higher = more human-like.
    """
    import sapiens

    scores = []
    for i, seq in enumerate(sequences):
        try:
            prob_df = sapiens.predict_scores(seq, "H")
            probs = [prob_df.iloc[j][seq[j]] for j in range(len(seq))]
            scores.append(float(np.mean(probs)))
        except Exception as e:
            print(f"    WARNING: seq {i} failed: {e}")
            scores.append(float("nan"))
        if (i + 1) % 100 == 0:
            print(f"    Sapiens: {i+1}/{len(sequences)}")
    return scores


def main():
    print("=" * 60)
    print("SAPIENS HUMANNESS SCORING")
    print("=" * 60)

    for model_name in MODELS:
        csv_path = CSV_DIR / f"{model_name}.csv"
        if not csv_path.exists():
            print(f"SKIP: {csv_path}")
            continue

        out_path = CSV_DIR / f"{model_name}_sapiens.csv"
        print(f"\n--- {model_name} ---")

        df = pd.read_csv(csv_path)
        seqs = df["generated_sequence"].tolist()
        print(f"  {len(seqs)} sequences")

        humanness = compute_sapiens_humanness(seqs)
        df["sapiens_humanness"] = humanness

        df.to_csv(out_path, index=False)
        valid_scores = [s for s in humanness if not np.isnan(s)]
        print(f"  Humanness: mean={np.mean(valid_scores):.4f}, "
              f"std={np.std(valid_scores):.4f}")
        print(f"  Saved: {out_path.name}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for model_name in MODELS:
        out_path = CSV_DIR / f"{model_name}_sapiens.csv"
        if out_path.exists():
            df = pd.read_csv(out_path)
            h = df["sapiens_humanness"]
            print(f"  {model_name:30s}  humanness={h.mean():.4f} (std={h.std():.4f})")


if __name__ == "__main__":
    main()
