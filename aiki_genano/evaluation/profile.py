"""
Comprehensive nanobody sequence profiling.

Computes every developability metric we have — reusing reward functions from
rewards.py and BioPython for biophysical properties.

Metric groups:
  1. GDPO reward scores (exact values the optimizer saw)
  2. Liability motif breakdown (individual motif counts)
  3. Biophysical properties (GRAVY, pI, instability, aromaticity, MW, charge)
  4. CDR-level analysis (lengths, charges)
  5. Aggrescan aggregation profiling
  6. Secondary structure fractions

Usage (inside Docker):
    python -m aiki_genano.evaluation.profile \
        --csv /app/output/csv/SFT_20k.csv
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict

import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from aiki_genano.rewards.rewards import (
    # Constants
    NBV1_CDR1,
    NBV1_CDR2,
    NBV1_CDR3,
    NBV1_FR2,
    STANDARD_AA_SET,
    DEAMIDATION_MOTIFS,
    ISOMERIZATION_MOTIFS,
    FRAGMENTATION_MOTIFS,
    GLYCOSYLATION_PATTERN,
    INTEGRIN_MOTIFS,
    # Core scoring functions (used in GDPO rewards)
    scan_sequence_liabilities_core,
    find_hydrophobic_patches_core,
    nbv1_fr2_aggregation_score,
    nbv1_vhh_hallmark_score,
    calculate_expression_score_core,
    scaffold_integrity_score,
    _is_valid_nbv1,
    _clamp01,
)
from aiki_genano.rewards.nanobody_scaffold import (
    normalize_for_prediction,
)

# =========================================================================
# Aggrescan a3v scale (Conchillo-Solé et al., BMC Bioinformatics 2007)
# =========================================================================
_A3V: Dict[str, float] = {
    "A": 0.25, "C": -0.43, "D": -1.04, "E": -0.98, "F": 0.86,
    "G": 0.06, "H": -0.18, "I": 0.81, "K": -1.18, "L": 0.56,
    "M": 0.43, "N": -0.59, "P": -0.31, "Q": -0.26, "R": -0.72,
    "S": -0.28, "T": -0.04, "V": 0.62, "W": 0.14, "Y": 0.29,
}
_AGGRESCAN_WINDOW = 5
_AGGRESCAN_THRESHOLD = -0.02

_LIABILITY_K = 10.0  # same k used in GDPO liability reward


def _clean(seq: str) -> str:
    return "".join(c for c in seq.upper() if c in STANDARD_AA_SET)


def _region(seq: str, start: int, end: int) -> str:
    if start >= len(seq):
        return ""
    return seq[start : min(end, len(seq))]


def _simple_charge(seq: str) -> int:
    return seq.count("R") + seq.count("K") + seq.count("H") - seq.count("D") - seq.count("E")


def _aggrescan(seq: str) -> Dict[str, float]:
    n = len(seq)
    if n < _AGGRESCAN_WINDOW:
        return {"aggrescan_na4vss": float("nan"), "aggrescan_nhs": 0.0, "aggrescan_aat": 0.0}

    a3v_vals = [_A3V.get(aa, 0.0) for aa in seq]
    na4vss = sum(a3v_vals) / n

    half_w = _AGGRESCAN_WINDOW // 2
    smoothed = []
    for i in range(n):
        lo = max(0, i - half_w)
        hi = min(n, i + half_w + 1)
        smoothed.append(sum(a3v_vals[lo:hi]) / (hi - lo))

    nhs = 0
    aat = 0.0
    in_hs = False
    for val in smoothed:
        if val > _AGGRESCAN_THRESHOLD:
            aat += val - _AGGRESCAN_THRESHOLD
            if not in_hs:
                nhs += 1
                in_hs = True
        else:
            in_hs = False

    return {"aggrescan_na4vss": round(na4vss, 5), "aggrescan_nhs": float(nhs), "aggrescan_aat": round(aat, 4)}


def _count_motifs(seq: str) -> Dict[str, int]:
    """Count every individual liability motif."""
    out: Dict[str, int] = {}
    for motif in DEAMIDATION_MOTIFS:
        out[f"motif_{motif}"] = len(re.findall(motif, seq))
    for motif in ISOMERIZATION_MOTIFS:
        out[f"motif_{motif}"] = len(re.findall(motif, seq))
    for motif in FRAGMENTATION_MOTIFS:
        out[f"motif_{motif}"] = len(re.findall(motif, seq))
    out["motif_N_glyco"] = len(GLYCOSYLATION_PATTERN.findall(seq))
    for motif in INTEGRIN_MOTIFS:
        out[f"motif_{motif}"] = len(re.findall(motif, seq))

    cdr_ranges = [NBV1_CDR1, NBV1_CDR2, NBV1_CDR3]
    cdr_met = sum(
        1 for m in re.finditer("M", seq)
        if any(s <= m.start() < e for s, e in cdr_ranges)
    )
    out["motif_CDR_Met"] = cdr_met
    return out


def compute_sequence_profile(seq: str) -> Dict[str, float]:
    """Full profile: GDPO rewards + motifs + biophysical + CDR + aggrescan."""
    clean = _clean(seq)
    is_valid = _is_valid_nbv1(clean)

    # -- GDPO reward scores (exact same functions used during training) -------
    if is_valid:
        core = normalize_for_prediction(clean, strip_termini=True).core

        liab_result = scan_sequence_liabilities_core(core)
        liability_severity = float(liab_result["liability_severity"])

        patch_result = find_hydrophobic_patches_core(core)

        gdpo = {
            "reward_scaffold_integrity": scaffold_integrity_score(clean),
            "reward_liability": _clamp01(1.0 / (1.0 + liability_severity / _LIABILITY_K)),
            "reward_hydrophobic_patch": _clamp01(1.0 - float(patch_result["patch_score"])),
            "reward_fr2_aggregation": _clamp01(1.0 - nbv1_fr2_aggregation_score(core)),
            "reward_vhh_hallmark": _clamp01(nbv1_vhh_hallmark_score(core)),
            "reward_expression": _clamp01(float(calculate_expression_score_core(core)["expression_score"])),
        }

        detail = {
            "liability_severity": liability_severity,
            "n_hydrophobic_patches": float(patch_result["n_patches"]),
            "hydrophobic_patch_fraction": float(patch_result["patch_fraction"]),
            "fr2_hydrophobicity": nbv1_fr2_aggregation_score(core),
            "vhh_hallmark_score": nbv1_vhh_hallmark_score(core),
            "expression_score": float(calculate_expression_score_core(core)["expression_score"]),
        }
    else:
        gdpo = {
            "reward_scaffold_integrity": scaffold_integrity_score(clean),
            "reward_liability": 0.0,
            "reward_hydrophobic_patch": 0.0,
            "reward_fr2_aggregation": 0.0,
            "reward_vhh_hallmark": 0.0,
            "reward_expression": 0.0,
        }
        detail = {
            "liability_severity": float("nan"),
            "n_hydrophobic_patches": float("nan"),
            "hydrophobic_patch_fraction": float("nan"),
            "fr2_hydrophobicity": float("nan"),
            "vhh_hallmark_score": float("nan"),
            "expression_score": float("nan"),
        }

    # -- Individual motif counts ----------------------------------------------
    motifs = _count_motifs(clean)

    # -- BioPython biophysical properties -------------------------------------
    if len(clean) < 5:
        biophys = {
            "gravy": float("nan"), "isoelectric_point": float("nan"),
            "instability_index": float("nan"), "aromaticity": float("nan"),
            "molecular_weight": float("nan"), "charge_at_pH7": float("nan"),
            "charge_at_pH74": float("nan"),
            "helix_fraction": float("nan"), "turn_fraction": float("nan"),
            "sheet_fraction": float("nan"),
        }
    else:
        pa = ProteinAnalysis(clean)
        ss = pa.secondary_structure_fraction()
        biophys = {
            "gravy": pa.gravy(),
            "isoelectric_point": pa.isoelectric_point(),
            "instability_index": pa.instability_index(),
            "aromaticity": pa.aromaticity(),
            "molecular_weight": pa.molecular_weight(),
            "charge_at_pH7": pa.charge_at_pH(7.0),
            "charge_at_pH74": pa.charge_at_pH(7.4),
            "helix_fraction": ss[0],
            "turn_fraction": ss[1],
            "sheet_fraction": ss[2],
        }

    # -- CDR-level analysis ---------------------------------------------------
    cdr1 = _region(clean, NBV1_CDR1[0], NBV1_CDR1[1])
    cdr2 = _region(clean, NBV1_CDR2[0], NBV1_CDR2[1])
    cdr3 = _region(clean, NBV1_CDR3[0], NBV1_CDR3[1])
    fr2 = _region(clean, NBV1_FR2[0], NBV1_FR2[1])

    framework_chars = list(clean)
    for s, e in [NBV1_CDR1, NBV1_CDR2, NBV1_CDR3]:
        for i in range(s, min(e, len(clean))):
            framework_chars[i] = ""
    framework_seq = "".join(framework_chars)

    cdr = {
        "cdr1_length": float(len(cdr1)),
        "cdr2_length": float(len(cdr2)),
        "cdr3_length": float(len(cdr3)),
        "total_cdr_length": float(len(cdr1) + len(cdr2) + len(cdr3)),
        "cdr1_charge": float(_simple_charge(cdr1)),
        "cdr2_charge": float(_simple_charge(cdr2)),
        "cdr3_charge": float(_simple_charge(cdr3)),
        "cdr_total_charge": float(_simple_charge(cdr1 + cdr2 + cdr3)),
        "framework_charge": float(_simple_charge(framework_seq)),
        "net_charge_simple": float(_simple_charge(clean)),
        "fr2_length": float(len(fr2)),
    }

    # -- Aggrescan ------------------------------------------------------------
    agg = _aggrescan(clean)

    return {**gdpo, **detail, **motifs, **biophys, **cdr, **agg}


# =========================================================================
# CSV processing
# =========================================================================

def _profile_csv(input_csv: Path, output_csv: Path, sequence_column: str) -> None:
    df = pd.read_csv(input_csv)
    if sequence_column not in df.columns:
        raise ValueError(
            f"Column '{sequence_column}' not found in {input_csv}. "
            f"Available: {list(df.columns)}"
        )

    print(f"Profiling {len(df)} sequences from {input_csv} ...")
    prof_df = df[sequence_column].astype(str).apply(compute_sequence_profile).apply(pd.Series)
    out = pd.concat([df, prof_df], axis=1)
    out.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}  ({len(prof_df.columns)} profile columns added)")

    valid = out[out.get("is_valid_126", True) == True]
    n_valid = len(valid)
    if n_valid == 0:
        print("  No valid sequences — skipping summary.")
        return

    print(f"\n--- Quick summary (valid sequences, n={n_valid}) ---")
    reward_cols = [c for c in prof_df.columns if c.startswith("reward_")]
    if reward_cols:
        print("\n  GDPO Rewards (mean):")
        for col in reward_cols:
            print(f"    {col:35s} {valid[col].mean():.4f}")

    motif_cols = [c for c in prof_df.columns if c.startswith("motif_")]
    if motif_cols:
        print("\n  Motif counts (mean per sequence):")
        for col in motif_cols:
            m = valid[col].mean()
            if m > 0.001:
                print(f"    {col:35s} {m:.3f}")

    key_biophys = ["gravy", "isoelectric_point", "instability_index", "charge_at_pH74"]
    print("\n  Key biophysical (mean):")
    for col in key_biophys:
        if col in valid.columns:
            print(f"    {col:35s} {valid[col].mean():.4f}")

    agg_cols = [c for c in prof_df.columns if c.startswith("aggrescan_")]
    if agg_cols:
        print("\n  Aggrescan (mean):")
        for col in agg_cols:
            print(f"    {col:35s} {valid[col].mean():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile nanobody sequence properties.")
    parser.add_argument("--seq", type=str, default=None, help="Single amino acid sequence.")
    parser.add_argument("--csv", type=str, default=None, help="Input CSV with sequences.")
    parser.add_argument(
        "--sequence-column",
        type=str,
        default="generated_sequence",
        help="Sequence column name for --csv mode.",
    )
    parser.add_argument("--out", type=str, default=None, help="Output CSV path for --csv mode.")
    args = parser.parse_args()

    if args.seq:
        result = compute_sequence_profile(args.seq)
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.csv:
        input_csv = Path(args.csv)
        output_csv = Path(args.out) if args.out else input_csv.with_name(f"{input_csv.stem}_profiled.csv")
        _profile_csv(input_csv=input_csv, output_csv=output_csv, sequence_column=args.sequence_column)
        return

    raise SystemExit("Provide either --seq or --csv.")


if __name__ == "__main__":
    main()
