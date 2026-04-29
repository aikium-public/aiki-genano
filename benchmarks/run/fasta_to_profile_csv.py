"""
Convert competitor FASTA → profile-input CSV for run_properties.py.

Guardrails (silent-fallback audit):
- Header MUST be `<tool>|<uniprot>|<target_name>|<idx>`. 4 pipe-separated
  fields exactly. Anything else → sys.exit(2), no "best-effort" parsing.
- --tool value MUST match the tool field on every header. Mismatch → abort.
- UniProt ID MUST exist in targets CSV. Missing → abort.
- Both validity gates stored side-by-side: `is_valid_126` (strict NBv1
  126-AA + GGGGS + ≥2 Cys + canonical alphabet) and `is_valid_vhh_loose`
  (110-130 AA + ≥2 Cys + canonical alphabet, NO linker requirement).
  Neither column is ever silently substituted for the other downstream.
- PepMLM-like short-peptide input (median length < 100 AA) HARD-ABORTS
  unless --peptide-mode is explicitly passed. VHH-specific metrics
  (TEMPRO / Sapiens / NetSolP) will produce out-of-distribution values
  on short peptides; we refuse by default rather than silently profile.
- Non-canonical characters are COUNTED, not silently stripped.
- Sequence-level stats (length distribution, valid counts) are printed
  loud so the caller can sanity-check before profiling.

Usage:
    python benchmarks/run/fasta_to_profile_csv.py \\
        --tool nanobert \\
        --fasta data/generated_2026_04_24/nanobert/generated_T0.7_seed42.fasta \\
        --out data/generated_2026_04_24/nanobert/nanobert_seed42_temp0.7.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from aiki_genano.rewards.rewards import _is_valid_nbv1, STANDARD_AA_SET


def fatal(msg: str) -> None:
    sys.stderr.write(f"\nFATAL: {msg}\n")
    sys.exit(2)


def parse_fasta(path: Path) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    header, buf = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(buf)))
                header = line[1:]
                buf = []
            else:
                buf.append(line)
        if header is not None:
            records.append((header, "".join(buf)))
    return records


def strict_parse_header(header: str) -> Dict[str, object]:
    parts = header.split("|")
    if len(parts) != 4:
        fatal(f"expected 4-field header '<tool>|<uniprot>|<target>|<idx>', "
              f"got {len(parts)} fields: {header!r}")
    tool, uniprot, target_name, idx_str = parts
    try:
        idx = int(idx_str)
    except ValueError:
        fatal(f"idx field must be integer: {header!r}")
    if not tool or not uniprot:
        fatal(f"empty tool or uniprot in header: {header!r}")
    return {"tool": tool, "uniprot": uniprot, "target_name": target_name, "idx": idx}


def is_valid_vhh_loose(seq: str) -> bool:
    """Loose VHH gate: 110-130 AA, ≥2 Cys, canonical alphabet only.
    Explicitly does NOT require the NBv1 GGGGS linker (most competitor
    generators produce real-world VHH termini without the linker)."""
    if not (110 <= len(seq) <= 130):
        return False
    if set(seq) - STANDARD_AA_SET:
        return False
    if seq.count("C") < 2:
        return False
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tool", required=True, help="Tool name; must match header's tool field")
    p.add_argument("--fasta", required=True, type=Path)
    # Targets CSV expected at $AIKI_TARGETS_CSV (default: benchmarks/targets.fasta
    # converted to CSV form by the caller). The internal Aikium screening targets
    # CSV is not redistributed.
    p.add_argument("--targets", type=Path,
                   default=Path(os.environ.get("AIKI_TARGETS_CSV",
                                               REPO / "benchmarks/targets.csv")))
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--peptide-mode", action="store_true",
                   help="Required flag for sub-100-AA inputs (e.g. PepMLM peptides). "
                        "Adapter does not prevent downstream VHH-metric runs — "
                        "the profiling pipeline itself must respect this tag.")
    args = p.parse_args()

    if not args.fasta.exists():
        fatal(f"FASTA not found: {args.fasta}")
    if not args.targets.exists():
        fatal(f"targets CSV not found: {args.targets}")

    t = pd.read_csv(args.targets)
    required_cols = {"UniProt_ID", "epitope"}
    missing = required_cols - set(t.columns)
    if missing:
        fatal(f"targets CSV missing columns: {missing}; got {list(t.columns)}")
    uniprot_to_epi = dict(zip(t["UniProt_ID"], t["epitope"]))

    records = parse_fasta(args.fasta)
    if not records:
        fatal(f"no records in {args.fasta}")

    rows = []
    tool_values: set = set()
    for header, seq in records:
        meta = strict_parse_header(header)
        tool_values.add(meta["tool"])
        if meta["tool"] != args.tool:
            fatal(f"header tool {meta['tool']!r} != --tool {args.tool!r}: {header!r}")
        if meta["uniprot"] not in uniprot_to_epi:
            fatal(f"uniprot {meta['uniprot']} not in targets CSV {args.targets}")
        peptide = uniprot_to_epi[meta["uniprot"]]
        non_std = sum(1 for c in seq if c not in STANDARD_AA_SET)
        rows.append({
            "peptide": peptide,
            "generated_sequence": seq,
            "gen_length": len(seq),
            "is_valid_126": _is_valid_nbv1(seq),
            "is_valid_vhh_loose": is_valid_vhh_loose(seq),
            "non_standard_aa": non_std,
            "model": meta["tool"],
            "seed": args.seed,
            "temperature": args.temperature,
            "tool": meta["tool"],
            "uniprot": meta["uniprot"],
            "target_name": meta["target_name"],
            "idx": meta["idx"],
            "peptide_mode": bool(args.peptide_mode),
        })
    df = pd.DataFrame(rows)

    if len(tool_values) > 1:
        fatal(f"multiple tool values in one FASTA: {tool_values}; refuse to mix")

    med_len = float(df["gen_length"].median())
    if med_len < 100 and not args.peptide_mode:
        fatal(
            f"median sequence length = {med_len:.0f} AA. VHH-specific metrics "
            f"(TEMPRO/Sapiens/NetSolP/NBv1 CDR slicing) are not calibrated for "
            f"short peptides. Re-run with --peptide-mode to explicitly opt in. "
            f"The downstream profiler must then skip VHH-specific metrics."
        )

    # Loud sanity report
    print(f"[adapter] tool={args.tool}  records={len(df)}")
    print(f"[adapter]   length     min={int(df.gen_length.min())} "
          f"median={int(med_len)} max={int(df.gen_length.max())} "
          f"mean={df.gen_length.mean():.1f}")
    print(f"[adapter]   is_valid_126         : {int(df.is_valid_126.sum())}/{len(df)}")
    print(f"[adapter]   is_valid_vhh_loose   : {int(df.is_valid_vhh_loose.sum())}/{len(df)}")
    print(f"[adapter]   any non-canonical AA : {int((df.non_standard_aa>0).sum())}/{len(df)}")
    print(f"[adapter]   unique uniprots      : {df.uniprot.nunique()}")
    print(f"[adapter]   idx per uniprot      : "
          f"{df.groupby('uniprot').size().min()}-{df.groupby('uniprot').size().max()}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[adapter] wrote {args.out}")


if __name__ == "__main__":
    main()
