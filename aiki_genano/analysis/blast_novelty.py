"""
Measure novelty of generated nanobody sequences against the training set.

Workflow:
  1. Build a BLAST protein database from training CSV sequences
  2. Query each generated sequence against it
  3. Report best-hit % identity, alignment length, mutations, e-value

Usage (inside Docker):
    # Build DB once (already done):
    python -m aiki_genano.evaluation.blast_novelty \
        --build-db --train-csv /app/data/training.csv

    # Query a profiled CSV (valid + unique only, with novelty stats):
    python -m aiki_genano.evaluation.blast_novelty \
        --query-csv /app/output/statistical/SFT/properties/SFT_seed42_temp0.7_profiled.csv

    # Query all CSVs for a model:
    python -m aiki_genano.evaluation.blast_novelty --model SFT
"""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
DB_DIR = _SCRIPT_DIR / "blast_db"
DB_NAME = "train_nb126"
DB_PATH = DB_DIR / DB_NAME
STAT_DIR = _SCRIPT_DIR / "csv" / "statistical"


def build_db(train_csv: str, seq_column: str = "protein") -> None:
    """Create a BLAST protein database from training sequences."""
    DB_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(train_csv)
    if seq_column not in df.columns:
        raise ValueError(f"Column '{seq_column}' not in {train_csv}. Have: {list(df.columns)}")

    seqs = df[seq_column].dropna().unique()
    fasta_path = DB_DIR / "train_sequences.fasta"
    with open(fasta_path, "w") as f:
        for i, seq in enumerate(seqs):
            f.write(f">train_{i}\n{seq}\n")

    print(f"Wrote {len(seqs)} unique training sequences to {fasta_path}")

    cmd = [
        "makeblastdb",
        "-in", str(fasta_path),
        "-dbtype", "prot",
        "-out", str(DB_PATH),
        "-parse_seqids",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"makeblastdb failed:\n{result.stderr}")
    print(f"BLAST database created at {DB_PATH}")


def _blast_single(seq: str, db_path: str, evalue: float = 10.0) -> Dict[str, float]:
    """Run blastp for a single query sequence, return best hit stats."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as tmp:
        tmp.write(f">query\n{seq}\n")
        tmp_path = tmp.name

    try:
        cmd = [
            "blastp",
            "-query", tmp_path,
            "-db", db_path,
            "-evalue", str(evalue),
            "-outfmt", "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore",
            "-max_target_seqs", "1",
            "-num_threads", "1",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        lines = result.stdout.strip().split("\n")

        if not lines or not lines[0]:
            return {"blast_pident": 0.0, "blast_alen": 0.0, "blast_evalue": 999.0, "blast_bitscore": 0.0}

        fields = lines[0].split("\t")
        return {
            "blast_pident": float(fields[2]),
            "blast_alen": float(fields[3]),
            "blast_evalue": float(fields[10]),
            "blast_bitscore": float(fields[11]),
        }
    finally:
        os.unlink(tmp_path)


NO_HIT = {
    "blast_pident": 0.0, "blast_alen": 0, "blast_mismatches": 0,
    "blast_gapopen": 0, "blast_evalue": 999.0, "blast_bitscore": 0.0,
    "blast_mutations": 0, "blast_is_exact": False, "blast_subject_id": "",
}


def _blast_batch(sequences: List[str], db_path: str, evalue: float = 10.0,
                 threads: int = 4) -> List[Dict]:
    """Run blastp for a batch of sequences in a single call."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as tmp:
        for i, seq in enumerate(sequences):
            tmp.write(f">q{i}\n{seq}\n")
        tmp_path = tmp.name

    try:
        cmd = [
            "blastp",
            "-query", tmp_path,
            "-db", db_path,
            "-evalue", str(evalue),
            "-outfmt", "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore",
            "-max_target_seqs", "1",
            "-num_threads", str(threads),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        hits: Dict[str, Dict] = {}
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            fields = line.split("\t")
            qid = fields[0]
            if qid not in hits:
                pident = float(fields[2])
                alen = int(float(fields[3]))
                mismatch = int(float(fields[4]))
                gapopen = int(float(fields[5]))
                mutations = mismatch + gapopen
                hits[qid] = {
                    "blast_pident": pident,
                    "blast_alen": alen,
                    "blast_mismatches": mismatch,
                    "blast_gapopen": gapopen,
                    "blast_evalue": float(fields[10]),
                    "blast_bitscore": float(fields[11]),
                    "blast_mutations": mutations,
                    "blast_is_exact": (pident == 100.0 and alen >= len(sequences[int(qid[1:])])),
                    "blast_subject_id": fields[1],
                }

        results = []
        for i in range(len(sequences)):
            qid = f"q{i}"
            results.append(hits.get(qid, dict(NO_HIT)))
        return results
    finally:
        os.unlink(tmp_path)


def query_csv(
    csv_path: str,
    seq_column: str = "generated_sequence",
    output: Optional[str] = None,
) -> None:
    """BLAST valid unique sequences from a generation CSV against training DB."""
    db = str(DB_PATH)
    if not os.path.exists(db + ".pdb") and not os.path.exists(db + ".psq"):
        raise RuntimeError(
            f"BLAST DB not found at {db}. Run with --build-db first."
        )

    df = pd.read_csv(csv_path)
    n_total = len(df)

    if "is_valid_126" in df.columns:
        df = df[df["is_valid_126"] == True].copy()
    n_valid = len(df)

    unique_seqs = df[seq_column].dropna().unique().tolist()
    n_unique = len(unique_seqs)

    print(f"  Total: {n_total}  Valid: {n_valid}  Unique valid: {n_unique}")
    print(f"  BLASTing {n_unique} unique sequences against training DB …")

    blast_results = _blast_batch(unique_seqs, db, threads=min(os.cpu_count() or 4, 8))

    seq_to_blast = {}
    for seq, res in zip(unique_seqs, blast_results):
        seq_to_blast[seq] = res

    for col in NO_HIT.keys():
        df[col] = df[seq_column].map(lambda s, c=col: seq_to_blast.get(s, NO_HIT)[c])

    if output is None:
        p = Path(csv_path)
        output = str(p.with_name(p.stem + "_blast.csv"))

    df.to_csv(output, index=False)
    print(f"  Saved: {output}")
    _print_summary(df, n_total)


def _print_summary(df: pd.DataFrame, n_total_raw: int) -> None:
    n = len(df)
    pi = df["blast_pident"]
    mut = df["blast_mutations"]
    exact = df["blast_is_exact"].sum()

    print(f"\n  {'='*55}")
    print(f"  BLAST Novelty Summary  (n={n} valid unique-mapped seqs)")
    print(f"  {'='*55}")
    print(f"  Raw total:          {n_total_raw}")
    print(f"  Valid:               {n}")
    print(f"  Unique sequences:   {df['generated_sequence'].nunique()}")
    print()
    print(f"  --- Best-hit % Identity ---")
    print(f"  Mean:   {pi.mean():.1f}%")
    print(f"  Median: {pi.median():.1f}%")
    print(f"  Min:    {pi.min():.1f}%   Max: {pi.max():.1f}%")
    print()
    print(f"  --- Mutation Distance (mismatches + gaps from closest training seq) ---")
    print(f"  Mean:   {mut.mean():.1f} mutations")
    print(f"  Median: {mut.median():.0f} mutations")
    print(f"  Min:    {mut.min()}   Max: {mut.max()}")
    print()
    print(f"  --- Novelty Distribution ---")
    print(f"  Exact match (100% id, full length): {exact:>6} / {n}  ({100*exact/n:.1f}%)")
    for thresh in [99, 98, 95, 90, 80]:
        count = (pi < thresh).sum()
        print(f"  < {thresh}% identity:                   {count:>6} / {n}  ({100*count/n:.1f}%)")
    print()
    print(f"  --- Mutations Histogram ---")
    for lo, hi in [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 999)]:
        count = ((mut >= lo) & (mut <= hi)).sum()
        label = f"{lo}" if lo == hi else f"{lo}-{hi}" if hi < 999 else f"{lo}+"
        bar = "█" * min(50, int(50 * count / n)) if n else ""
        print(f"  {label:>5} mut: {count:>6} ({100*count/n:5.1f}%)  {bar}")


def run_model(model_name: str) -> None:
    """Run BLAST novelty on all profiled CSVs for a model."""
    props_dir = STAT_DIR / model_name / "properties"
    if not props_dir.is_dir():
        print(f"[SKIP] {props_dir} does not exist.")
        return

    csvs = sorted(
        p for p in props_dir.glob(f"{model_name}_seed*_profiled.csv")
        if "_all_" not in p.name and "_blast" not in p.name
    )
    if not csvs:
        print(f"No profiled CSVs found in {props_dir}")
        return

    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}  ({len(csvs)} CSVs)")
    print(f"{'='*60}")

    all_blast_dfs = []
    for i, csv_path in enumerate(csvs, 1):
        out_path = props_dir / (csv_path.stem + "_blast.csv")
        if out_path.exists():
            print(f"\n  [{i}/{len(csvs)}] {csv_path.name} — already done, loading.")
            all_blast_dfs.append(pd.read_csv(out_path))
            continue
        print(f"\n  [{i}/{len(csvs)}] {csv_path.name}")
        query_csv(str(csv_path), output=str(out_path))
        all_blast_dfs.append(pd.read_csv(out_path))

    if all_blast_dfs:
        merged = pd.concat(all_blast_dfs, ignore_index=True)
        print(f"\n{'='*60}")
        print(f"OVERALL {model_name} (all CSVs merged)")
        print(f"{'='*60}")
        _print_summary(merged, len(merged))

        merged_path = props_dir / f"{model_name}_all_blast.csv"
        merged.to_csv(merged_path, index=False)
        print(f"  Merged → {merged_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="BLAST novelty analysis for generated nanobodies.")
    parser.add_argument("--build-db", action="store_true", help="Build BLAST DB from training CSV.")
    parser.add_argument(
        "--train-csv", type=str,
        default="/app/data/training.csv",
        help="Training CSV (needs 'protein' column).",
    )
    parser.add_argument("--query-csv", type=str, default=None, help="Single CSV to query.")
    parser.add_argument("--model", type=str, default=None,
                        choices=["SFT", "DPO", "GDPO", "all"],
                        help="Process all profiled CSVs for a model.")
    parser.add_argument("--sequence-column", type=str, default="generated_sequence")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path.")
    args = parser.parse_args()

    if args.build_db:
        build_db(args.train_csv)

    if args.model:
        models = ["SFT", "DPO", "GDPO"] if args.model == "all" else [args.model]
        for m in models:
            run_model(m)
    elif args.query_csv:
        query_csv(args.query_csv, args.sequence_column, args.out)
    elif not args.build_db:
        raise SystemExit("Provide --build-db, --query-csv, or --model.")


if __name__ == "__main__":
    main()
