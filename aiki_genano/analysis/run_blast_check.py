"""
Run BLAST novelty check for all four models (SFT, DPO, GDPO, GDPO_SFT)
against the pre-built training sequence database.

# inside Docker:
#   apt-get update && apt-get install -y ncbi-blast+
#   pip install tqdm

Usage (inside Docker):
    python -m aiki_genano.analysis.run_blast_check
    python -m aiki_genano.analysis.run_blast_check --seed 42 --temp 0.7
    python -m aiki_genano.analysis.run_blast_check --models SFT DPO GDPO
"""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm


# ── Paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH     = _SCRIPT_DIR / "blast_db" / "train_nb126"
STAT_DIR    = _SCRIPT_DIR / "csv" / "statistical"

ALL_MODELS  = ["SFT", "DPO", "GDPO", "GDPO_SFT"]

NO_HIT = {
    "blast_pident":     0.0,
    "blast_alen":       0,
    "blast_mismatches": 0,
    "blast_gapopen":    0,
    "blast_evalue":     999.0,
    "blast_bitscore":   0.0,
    "blast_mutations":  0,
    "blast_is_novel":   True,
    "blast_is_exact":   False,
    "blast_subject_id": "",
}


def _verify_db() -> None:
    required = [".pin", ".phr", ".psq"]
    missing = [s for s in required if not DB_PATH.with_suffix(s).exists()]
    if missing:
        raise FileNotFoundError(
            f"BLAST database not found or incomplete at {DB_PATH}. Missing: {missing}\n"
            "Build it first:\n"
            "  python -m aiki_genano.analysis.blast_novelty "
            "--build-db --train-csv /app/data/training.csv"
        )
    print(f"BLAST DB: {DB_PATH}  ✓")


def _blast_batch(sequences: List[str], threads: int = 8) -> List[Dict]:
    """Run blastp for a batch of sequences in a single subprocess call."""
    if not sequences:
        return []

    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as tmp:
        for i, seq in enumerate(sequences):
            tmp.write(f">q{i}\n{seq}\n")
        tmp_path = tmp.name

    try:
        cmd = [
            "blastp",
            "-query", tmp_path,
            "-db", str(DB_PATH),
            "-evalue", "10",
            "-outfmt", "6 qseqid sseqid pident length mismatch gapopen "
                       "qstart qend sstart send evalue bitscore",
            "-max_target_seqs", "1",
            "-num_threads", str(threads),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            raise RuntimeError(
                f"blastp failed with code {result.returncode}\n"
                f"STDERR:\n{result.stderr}"
            )

        hits: Dict[str, Dict] = {}
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            f = line.split("\t")
            qid = f[0]

            if qid not in hits:
                pident   = float(f[2])
                alen     = int(float(f[3]))
                mismatch = int(float(f[4]))
                gapopen  = int(float(f[5]))
                qlen     = len(sequences[int(qid[1:])])
                mutations = mismatch + gapopen

                hits[qid] = {
                    "blast_pident":     pident,
                    "blast_alen":       alen,
                    "blast_mismatches": mismatch,
                    "blast_gapopen":    gapopen,
                    "blast_evalue":     float(f[10]),
                    "blast_bitscore":   float(f[11]),
                    "blast_mutations":  mutations,
                    "blast_is_novel":   pident < 95.0,
                    "blast_is_exact":   (pident == 100.0 and alen == qlen),
                    "blast_subject_id": f[1],
                }

        return [hits.get(f"q{i}", dict(NO_HIT)) for i in range(len(sequences))]
    finally:
        os.unlink(tmp_path)


def _chunk_list(items: List[str], chunk_size: int) -> List[List[str]]:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def run_blast_for_csv(
    csv_path: Path,
    seq_col: str = "generated_sequence",
    overwrite: bool = False,
    chunk_size: int = 250,
) -> Optional[pd.DataFrame]:
    """BLAST valid unique sequences from one profiled CSV. Returns annotated df."""
    out_path = csv_path.with_name(csv_path.stem + "_blast.csv")
    if out_path.exists() and not overwrite:
        print(f"  [skip] {csv_path.name} — blast file already exists ({out_path.name})")
        return pd.read_csv(out_path)

    if not csv_path.exists():
        print(f"  [skip] {csv_path.name} — file not found")
        return None

    df = pd.read_csv(csv_path)
    n_total = len(df)

    if "is_valid_126" in df.columns:
        df = df[df["is_valid_126"] == True].copy()
    n_valid = len(df)

    if seq_col not in df.columns:
        print(f"  [error] column '{seq_col}' not in {csv_path.name}")
        return None

    unique_seqs = df[seq_col].dropna().unique().tolist()
    n_unique = len(unique_seqs)

    print(f"  {csv_path.name}  →  total={n_total}  valid={n_valid}  unique={n_unique}")

    if n_unique == 0:
        print("  [skip] no unique sequences to BLAST")
        return df

    threads = min(os.cpu_count() or 4, 16)
    chunks = _chunk_list(unique_seqs, chunk_size)

    all_results: List[Dict] = []
    pbar = tqdm(
        chunks,
        desc=f"BLAST {csv_path.stem}",
        unit="chunk",
        leave=False,
    )
    for chunk in pbar:
        pbar.set_postfix({"seqs": len(chunk), "threads": threads})
        all_results.extend(_blast_batch(chunk, threads=threads))

    seq_to_blast = {seq: res for seq, res in zip(unique_seqs, all_results)}
    for col in NO_HIT:
        df[col] = df[seq_col].map(lambda s, c=col: seq_to_blast.get(s, NO_HIT)[c])

    df.to_csv(out_path, index=False)
    print(f"  Saved → {out_path.name}")
    return df


def _model_summary(df: pd.DataFrame, model: str) -> Dict:
    pi  = df["blast_pident"]
    mut = df["blast_mutations"]
    n   = len(df)
    return {
        "model":          model,
        "n_seqs":         n,
        "mean_pident":    round(pi.mean(), 2),
        "median_pident":  round(pi.median(), 2),
        "mean_mutations": round(mut.mean(), 2),
        "exact_match_%":  round(100 * df["blast_is_exact"].sum() / n, 2) if n else 0,
        "novel_<95%":     round(100 * (pi < 95).sum() / n, 2) if n else 0,
        "novel_<90%":     round(100 * (pi < 90).sum() / n, 2) if n else 0,
        "novel_<80%":     round(100 * (pi < 80).sum() / n, 2) if n else 0,
    }


def print_comparison_table(summaries: List[Dict]) -> None:
    cols = [
        "model", "n_seqs", "mean_pident", "median_pident",
        "mean_mutations", "exact_match_%", "novel_<95%", "novel_<90%", "novel_<80%"
    ]
    col_w = [14, 8, 13, 15, 16, 14, 12, 12, 12]
    header = "".join(f"{c:<{w}}" for c, w in zip(cols, col_w))
    sep = "-" * len(header)

    print(f"\n{'=' * len(header)}")
    print("BLAST Novelty Comparison — all models")
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)
    for s in summaries:
        row = "".join(f"{str(s.get(c, '')):<{w}}" for c, w in zip(cols, col_w))
        print(row)
    print(sep)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BLAST novelty check for generated nanobody sequences."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run BLAST even if _blast.csv already exists")
    parser.add_argument("--seq-col", type=str, default="generated_sequence")
    parser.add_argument("--chunk-size", type=int, default=250,
                        help="Number of unique sequences per BLAST call")
    args = parser.parse_args()

    _verify_db()

    summaries = []
    model_bar = tqdm(args.models, desc="Models", unit="model")
    for model in model_bar:
        model_bar.set_postfix({"seed": args.seed, "temp": args.temp})

        props_dir = STAT_DIR / model / "properties"
        csv_path  = props_dir / f"{model}_seed{args.seed}_temp{args.temp}_profiled.csv"

        print(f"\n{'=' * 60}")
        print(f"  MODEL: {model}  (seed={args.seed}, temp={args.temp})")
        print(f"{'=' * 60}")

        df = run_blast_for_csv(
            csv_path,
            seq_col=args.seq_col,
            overwrite=args.overwrite,
            chunk_size=args.chunk_size,
        )
        if df is not None and "blast_pident" in df.columns:
            summaries.append(_model_summary(df, model))

    if summaries:
        print_comparison_table(summaries)
        out_csv = STAT_DIR / f"blast_comparison_seed{args.seed}_temp{args.temp}.csv"
        pd.DataFrame(summaries).to_csv(out_csv, index=False)
        print(f"Comparison table saved → {out_csv}")


if __name__ == "__main__":
    main()