"""
Run NetSolP on generated nanobody CSVs.

Converts CSVs → FASTA, runs NetSolP predict.py, merges solubility/usability
predictions back into the original CSVs.

Usage (inside Docker):
    pip install onnxruntime fair-esm
    python -m aiki_genano.evaluation.run_netsolp
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

CSV_DIR = Path("/app/output/csv")
NETSOLP_DIR = Path("/opt/netsolp")
PREDICT_PY = NETSOLP_DIR / "predict.py"
MODELS_DIR = NETSOLP_DIR / "models"

MODELS = ["SFT_20k", "DPO_6k", "GDPO_dpo_final_gated"]
MODEL_TYPE = "ESM12"  # fastest; use ESM1b for best accuracy


def csv_to_fasta(csv_path: Path, fasta_path: Path) -> int:
    df = pd.read_csv(csv_path)
    with open(fasta_path, "w") as f:
        for idx, row in df.iterrows():
            seq = row["generated_sequence"]
            f.write(f">seq_{idx}\n{seq}\n")
    return len(df)


def run_netsolp(fasta_path: Path, output_path: Path):
    cmd = [
        sys.executable, str(PREDICT_PY),
        "--FASTA_PATH", str(fasta_path),
        "--OUTPUT_PATH", str(output_path),
        "--MODELS_PATH", str(MODELS_DIR),
        "--MODEL_TYPE", MODEL_TYPE,
        "--PREDICTION_TYPE", "SU",
    ]
    print(f"  Running: {' '.join(cmd[-6:])}")
    subprocess.run(cmd, check=True, cwd=str(NETSOLP_DIR))


def merge_predictions(csv_path: Path, netsolp_csv: Path, out_path: Path):
    orig = pd.read_csv(csv_path)
    preds = pd.read_csv(netsolp_csv)

    sol_col = "predicted_solubility"
    usa_col = "predicted_usability"

    if sol_col in preds.columns:
        orig["netsolp_solubility"] = preds[sol_col].values
    if usa_col in preds.columns:
        orig["netsolp_usability"] = preds[usa_col].values

    orig.to_csv(out_path, index=False)
    return orig


def main():
    for model_name in MODELS:
        csv_path = CSV_DIR / f"{model_name}.csv"
        if not csv_path.exists():
            print(f"SKIP: {csv_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")

        fasta_path = CSV_DIR / f"{model_name}.fasta"
        netsolp_out = CSV_DIR / f"{model_name}_netsolp_raw.csv"
        merged_out = CSV_DIR / f"{model_name}_netsolp.csv"

        n = csv_to_fasta(csv_path, fasta_path)
        print(f"  Converted {n} sequences → {fasta_path.name}")

        run_netsolp(fasta_path, netsolp_out)

        df = merge_predictions(csv_path, netsolp_out, merged_out)

        sol = df["netsolp_solubility"]
        usa = df["netsolp_usability"]
        print(f"  Solubility: mean={sol.mean():.4f}, std={sol.std():.4f}")
        print(f"  Usability:  mean={usa.mean():.4f}, std={usa.std():.4f}")
        print(f"  Saved: {merged_out.name}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for model_name in MODELS:
        merged_out = CSV_DIR / f"{model_name}_netsolp.csv"
        if merged_out.exists():
            df = pd.read_csv(merged_out)
            print(f"  {model_name:30s}  sol={df['netsolp_solubility'].mean():.4f}  "
                  f"usa={df['netsolp_usability'].mean():.4f}")


if __name__ == "__main__":
    main()
