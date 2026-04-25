"""
NBv1 (Yotta Nb-v1, 126 AA) reward properties for GDPO/GRPO training.

Why this file exists
--------------------
The generic reward/property helpers in `fast_properties.py` include Kabat-based
position scaling like:

    scale = length / 120.0
    seq_pos = int(kabat_pos * scale) - 1

That scaling is *not* appropriate for Yotta backbones, where we have an exact,
IMGT-validated scaffold definition with fixed CDR boundaries and library-specific
terminal requirements.

This module provides NBv1-specific reward functions that:
- Use the Yotta single source of truth (`NBV1`) for regions and validation
- Avoid any length-based position scaling heuristics
- Compute properties on the normalized *core* sequence (strip `GGGGS` linker)
- Provide a scaffold-integrity reward grounded in library conventions

Rewards implemented here (final set):
  1) fr2_aggregation      (NBv1 exact FR2 segment hydrophobicity risk)
  2) hydrophobic_patch    (patch score on core sequence)
  3) liability            (liability severity on core sequence)
  4) expression           (expression proxy on core sequence)
  5) vhh_hallmark         (FR2 hallmark residues at fixed NBv1 indices)
  6) scaffold_integrity   (Yotta Nb-v1 backbone validity: length+suffix rules)

All reward functions return values in [0, 1] where higher is better.
"""
from __future__ import annotations
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple
from aiki_genano.rewards.nanobody_scaffold import (
    NB_V1_REFERENCE,
    NBV1,
    normalize_for_prediction,
    validate_nbv1_sequence,
)


# =============================================================================
# Core constants (copied here to avoid dependence on fast_properties.py)
# =============================================================================

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
STANDARD_AA_SET = set(STANDARD_AA)

# Normalized hydrophobicity for aggregation scoring (0..1, higher=more hydrophobic)
# This is a standard, fixed lookup table (not Yotta-specific).

HYDROPHOBICITY_NORM: Dict[str, float] = {
    "A": 0.62,
    "R": 0.00,
    "N": 0.11,
    "D": 0.11,
    "C": 0.68,
    "Q": 0.11,
    "E": 0.11,
    "G": 0.50,
    "H": 0.23,
    "I": 1.00,
    "L": 0.94,
    "K": 0.07,
    "M": 0.71,
    "F": 0.81,
    "P": 0.32,
    "S": 0.36,
    "T": 0.39,
    "W": 0.60,
    "Y": 0.26,
    "V": 0.97,
}

# # Kyte-Doolittle hydropathy scale (reference: Kyte & Doolittle, J Mol Biol 1982)
# # https://www.sciencedirect.com/science/article/pii/0022283682905150 ( Table 2)
# KD_HYDROPATHY: Dict[str, float] = {
#     "I": 4.5,
#     "V": 4.2,
#     "L": 3.8,
#     "F": 2.8,
#     "C": 2.5,
#     "M": 1.9,
#     "A": 1.8,
#     "G": -0.4,
#     "T": -0.7,
#     "S": -0.8,
#     "W": -0.9,
#     "Y": -1.3,
#     "P": -1.6,
#     "H": -3.2,
#     "E": -3.5,
#     "Q": -3.5,
#     "D": -3.5,
#     "N": -3.5,
#     "K": -3.9,
#     "R": -4.5,
# }

# def kd_minmax_normalize(kd_dict: Dict[str, float]) -> Dict[str, float]:
#     vmin = min(kd_dict.values())  # -4.5
#     vmax = max(kd_dict.values())  #  4.5
#     return {aa: (val - vmin) / (vmax - vmin) for aa, val in kd_dict.items()}

# # Normalized hydrophobicity for aggregation scoring (0..1, higher=more hydrophobic)
# HYDROPHOBICITY_NORM = kd_minmax_normalize(KD_HYDROPATHY)

# VHH FR2 hallmark positions and residues (Kabat numbering) from VHH literature.
# Format: position -> (good/VHH-like residues, bad/aggregation-prone residues)
VHH_FR2_POSITIONS: Dict[int, Tuple[set[str], set[str]]] = {
    37: ({"E", "Q", "G", "A", "S"}, {"F", "Y", "W", "L", "I", "V"}),
    44: ({"E", "Q", "R", "K", "G"}, {"W", "F", "Y", "L", "I", "V"}),
    45: ({"R", "K", "Q", "E", "L"}, {"W", "F", "Y", "I", "V", "M"}),
    47: ({"G", "S", "A", "L"}, {"W", "F", "Y", "I", "V"}),
}

# Liability motif definitions (simple, sequence-only)
DEAMIDATION_MOTIFS: Dict[str, int] = {"NG": 3, "NS": 3, "NT": 2, "NN": 2, "NH": 2, "NA": 1, "NQ": 1}
ISOMERIZATION_MOTIFS: Dict[str, int] = {"DG": 3, "DS": 2, "DT": 2, "DD": 1, "DH": 1}
FRAGMENTATION_MOTIFS: Dict[str, int] = {"DP": 3}
GLYCOSYLATION_PATTERN = re.compile(r"N[^P][ST]")
INTEGRIN_MOTIFS: Dict[str, int] = {"RGD": 2, "NGR": 1, "LDV": 1}

NBV1_LENGTH: int = NBV1.length  # 126
NBV1_CDR1: Tuple[int, int] = NBV1.cdr1
NBV1_CDR2: Tuple[int, int] = NBV1.cdr2
NBV1_CDR3: Tuple[int, int] = NBV1.cdr3

NBV1_FR2: Tuple[int, int] = (NBV1_CDR1[1], NBV1_CDR2[0])  # (36, 53)

NBV1_HALLMARK_KABAT_TO_INDEX: Dict[int, int] = {
    37: 37,
    44: 44,
    45: 45,
    47: 47,
}

# Scaffold sanity checks (fail fast if the backbone definition changes)
assert len(NB_V1_REFERENCE) == NBV1_LENGTH
assert NB_V1_REFERENCE.endswith("GGGGS")
assert NB_V1_REFERENCE[NBV1_FR2[0] : NBV1_FR2[0] + 6] == "LGWFRQ", (
    "Unexpected NBv1 FR2 anchor motif; hallmark index mapping likely invalid."
)

def clean_sequence(completion: str) -> str:
    """Extract a clean AA sequence from a model completion (keeps only standard AAs)."""
    if "<|im_end|>" in completion:
        completion = completion.split("<|im_end|>", 1)[0]
    completion = completion.replace("\n", "").strip()
    return "".join(c for c in completion.upper() if c in STANDARD_AA_SET)

def _nbv1_core(sequence_full: str) -> str:
    """
    Normalize to the core sequence for property prediction.
    For NBv1, this strips the trailing 'GGGGS' linker (126 -> 121).
    """
    return normalize_for_prediction(sequence_full, strip_termini=True).core

def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _is_valid_nbv1(seq_full: str) -> bool:
    """
    Validity gate for NBv1 property rewards (GDPO conditioned-reward pattern).

    All property rewards are gated behind this check so that invalid sequences
    receive 0.0 from EVERY reward channel, eliminating gradient leakage that
    drives reward hacking (EOS suppression / length gaming).

    Without this gate, 5 out of 6 rewards compute meaningful scores on
    biologically invalid sequences (wrong length, missing linker, etc.),
    giving the model a 0.90 reward-weight incentive to stay invalid.
    With the gate, the only path to non-zero reward is generating a
    structurally valid 126 AA NBv1 nanobody.

    See GDPO paper (arXiv 2601.05242) Section 3.2, Equation 8:
        r_k = r_k if r_l >= t, else 0
    Here r_l is structural validity and t = 1 (binary pass/fail).

    Checks (fast, no external dependencies):
      1. Exact length = 126 AA
      2. C-terminal GGGGS linker present
      3. Standard amino acid alphabet only
      4. At least 2 cysteines (disulfide bond capacity)
    """
    if len(seq_full) != NBV1_LENGTH:
        return False
    if not seq_full.endswith("GGGGS"):
        return False
    if set(seq_full) - STANDARD_AA_SET:
        return False
    if seq_full.count("C") < 2:
        return False
    return True


# =============================================================================
# Property implementations (NBv1-specific, implemented locally)
# =============================================================================

def find_hydrophobic_patches_core(sequence_core: str, *, min_patch_length: int = 5) -> Dict[str, object]:
    """
    Find consecutive hydrophobic stretches (sequence-only proxy for aggregation risk).

    Returns:
      - patch_score in [0,1], higher = worse
      - patch_fraction = fraction of residues contained in patches
    """
    seq = sequence_core.upper()
    hydrophobic = set("AILMFVWP") # check again the hydrophobic set here.
    patches = []
    current_start: Optional[int] = None
    current_len = 0

    for i, aa in enumerate(seq):
        if aa in hydrophobic:
            if current_start is None:
                current_start = i
            current_len += 1
        else:
            if current_len >= min_patch_length and current_start is not None:
                patches.append((current_start, i))
            current_start = None
            current_len = 0

    if current_len >= min_patch_length and current_start is not None:
        patches.append((current_start, len(seq)))

    total_patch_residues = sum(end - start for start, end in patches)
    patch_fraction = (total_patch_residues / len(seq)) if seq else 0.0
    patch_score = min(1.0, patch_fraction * 5.0)

    return {
        "n_patches": len(patches),
        "total_patch_residues": total_patch_residues,
        "patch_fraction": patch_fraction,
        "patch_score": patch_score,
    }


def scan_sequence_liabilities_core(sequence_core: str) -> Dict[str, object]:
    """
    Scan for common developability liabilities. Returns a severity summary.
    """
    seq = sequence_core.upper()
    hits_by_type: Dict[str, int] = {}
    severity = 0

    def add_hit(t: str, sev: int):
        nonlocal severity
        hits_by_type[t] = hits_by_type.get(t, 0) + 1
        severity += sev

    for motif, sev in DEAMIDATION_MOTIFS.items():
        for _ in re.finditer(motif, seq):
            add_hit("deamidation", sev)

    for motif, sev in ISOMERIZATION_MOTIFS.items():
        for _ in re.finditer(motif, seq):
            add_hit("isomerization", sev)

    for motif, sev in FRAGMENTATION_MOTIFS.items():
        for _ in re.finditer(motif, seq):
            add_hit("fragmentation", sev)

    for _ in GLYCOSYLATION_PATTERN.finditer(seq):
        add_hit("glycosylation", 3)

    # Oxidation-prone Met
    for _ in re.finditer("M", seq):
        add_hit("oxidation", 2)

    for motif, sev in INTEGRIN_MOTIFS.items():
        for _ in re.finditer(motif, seq):
            add_hit("integrin_binding", sev)

    # Charge clusters: 4+ charged residues in a 6-aa window
    charged = set("DEKRH")
    for i in range(max(0, len(seq) - 6)):
        window = seq[i : i + 6]
        if sum(1 for aa in window if aa in charged) >= 4:
            add_hit("charge_cluster", 2)

    return {
        "liability_severity": severity,
        "hits_by_type": hits_by_type,
    }


def calculate_expression_score_core(sequence_core: str) -> Dict[str, float]:
    """
    E. coli expression proxy based on amino acid composition (sequence-only heuristic).
    Returns `expression_score` in [0,1].
    """
    seq = sequence_core.upper()
    length = len(seq)
    if length == 0:
        return {"expression_score": 0.0}

    aa_counts = Counter(seq)
    rare_prone_fraction = sum(aa_counts.get(aa, 0) for aa in "RILGP") / length
    arg_fraction = aa_counts.get("R", 0) / length
    pro_fraction = aa_counts.get("P", 0) / length
    charged_fraction = sum(aa_counts.get(aa, 0) for aa in "DEKRH") / length

    score = 0.6

    if rare_prone_fraction > 0.25:
        score -= 0.15
    elif rare_prone_fraction < 0.15:
        score += 0.10

    if arg_fraction > 0.08:
        score -= 0.10
    elif arg_fraction < 0.03:
        score += 0.05

    if pro_fraction > 0.08:
        score -= 0.05

    if charged_fraction > 0.30:
        score -= 0.10
    elif charged_fraction < 0.10:
        score -= 0.05

    # Mild length preference (VHH typical range)
    if 110 <= length <= 130:
        score += 0.05
    elif length < 90 or length > 160:
        score -= 0.10

    return {"expression_score": _clamp01(score)}


def validate_sequence_basic(
    sequence_full: str,
    *,
    min_length: int = 126,
    max_length: int = 126,
    require_cysteines: bool = True,
) -> Dict[str, object]:
    """
    Basic nanobody sanity validation:
    - strict length
    - valid AA alphabet
    - no stop codon
    - >=2 cysteines (optional)
    - low-complexity check (protect against repetition collapse)
    """
    seq = sequence_full.upper()
    issues: List[str] = []

    if len(seq) < min_length:
        issues.append(f"too_short:{len(seq)}<{min_length}")
    if len(seq) > max_length:
        issues.append(f"too_long:{len(seq)}>{max_length}")

    invalid_chars = set(seq) - STANDARD_AA_SET
    if invalid_chars:
        issues.append(f"invalid_chars:{sorted(invalid_chars)}")

    if "*" in seq:
        issues.append("contains_stop_codon")

    cys_count = seq.count("C")
    if require_cysteines and cys_count < 2:
        issues.append(f"insufficient_cysteines:{cys_count}<2")

    # Low complexity: any AA occupies >80% of a 20-aa window
    if len(seq) >= 20:
        for i in range(len(seq) - 20):
            window = seq[i : i + 20]
            for aa in STANDARD_AA:
                if window.count(aa) / 20 > 0.8:
                    issues.append(f"low_complexity:{aa}@{i+1}")
                    break

    return {"is_valid": len(issues) == 0, "issues": issues, "length": len(seq), "cysteine_count": cys_count}


def nbv1_fr2_aggregation_score(sequence_core: str) -> Dict[str, float]:
    """
    NBv1 FR2 hydrophobicity risk score computed over the exact FR2 segment.

    Returns:
      dict with:
        - fr2_mean_hydrophobicity: mean HYDROPHOBICITY_NORM in FR2 (0..1, higher=worse)
        - aggregation_risk: alias of mean hydrophobicity (0..1, higher=worse)
    """
    if len(sequence_core) < NBV1_FR2[1]:
        # Not enough length to even define FR2; treat as high risk.
        return {"fr2_mean_hydrophobicity": 1.0, "aggregation_risk": 1.0}

    start, end = NBV1_FR2
    fr2 = sequence_core[start:end]
    if not fr2:
        return {"fr2_mean_hydrophobicity": 1.0, "aggregation_risk": 1.0}

    vals = [HYDROPHOBICITY_NORM.get(aa, 0.5) for aa in fr2]
    mean_h = sum(vals) / len(vals)
    mean_h = _clamp01(mean_h)
    return {"fr2_mean_hydrophobicity": mean_h, "aggregation_risk": mean_h}


def nbv1_vhh_hallmark_score(sequence_core: str) -> Dict[str, float]:
    """
    NBv1 VHH hallmark score at the FR2 tetrad (Kabat 37/44/45/47).

    Scoring per position:
      - good residue  -> 1.0
      - bad residue   -> 0.0
      - other residue -> 1 - HYDROPHOBICITY_NORM(res)  (penalize hydrophobics)

    Returns:
      dict with:
        - vhh_hallmark_score: average over 4 positions (0..1)
        - checked: list of per-position details
    """
    checked = []
    per_pos_scores: List[float] = []

    for kabat_pos, idx in NBV1_HALLMARK_KABAT_TO_INDEX.items():
        if idx < 0 or idx >= len(sequence_core):
            score = 0.0
            residue = None
        else:
            residue = sequence_core[idx]
            good_residues, bad_residues = VHH_FR2_POSITIONS[kabat_pos]
            if residue in good_residues:
                score = 1.0
            elif residue in bad_residues:
                score = 0.0
            else:
                score = _clamp01(1.0 - HYDROPHOBICITY_NORM.get(residue, 0.5))
                # change 0.5

        per_pos_scores.append(score)
        checked.append(
            {
                "kabat_position": kabat_pos,
                "sequence_index": idx,
                "residue": residue,
                "score": score,
            }
        )

    hallmark_score = sum(per_pos_scores) / len(per_pos_scores) if per_pos_scores else 0.0
    return {"vhh_hallmark_score": _clamp01(hallmark_score), "checked": checked}


def nbv1_scaffold_integrity(sequence_full: str) -> Dict[str, object]:
    """
    NBv1 scaffold integrity based on Yotta library conventions:
      - Must be a valid Yotta sequence
      - Must match the NBv1 backbone (126 AA and required suffix 'GGGGS')

    Returns dict with:
      - is_valid: bool
      - reward: float in {0.0, 1.0}
      - message: str
      - backbone_name: str | None
    """
    is_valid, message = validate_nbv1_sequence(sequence_full, raise_on_error=False)
    return {
        "is_valid": is_valid,
        "reward": 1.0 if is_valid else 0.0,
        "message": message,
        "backbone_name": NBV1.name if is_valid else None,
    }


# =============================================================================
# TRL reward function wrappers (List[str] -> List[float])
# =============================================================================

def fr2_aggregation_reward(completions: List[str], **kwargs) -> List[float]:
    """[0,1] where 1.0 = low FR2 hydrophobicity risk (NBv1 exact FR2).
    Gated: returns 0.0 for structurally invalid sequences."""
    out: List[float] = []
    for comp in completions:
        seq_full = clean_sequence(comp)
        # Validity gate: no reward for invalid sequences (prevents reward hacking)
        if not _is_valid_nbv1(seq_full):
            out.append(0.0)
            continue
        core = _nbv1_core(seq_full)
        risk = nbv1_fr2_aggregation_score(core)["aggregation_risk"]
        out.append(_clamp01(1.0 - risk))
    return out


def hydrophobic_patch_reward(completions: List[str], **kwargs) -> List[float]:
    """[0,1] where 1.0 = no hydrophobic patches (computed on NBv1 core).
    Gated: returns 0.0 for structurally invalid sequences."""
    out: List[float] = []
    for comp in completions:
        seq_full = clean_sequence(comp)
        # Validity gate: no reward for invalid sequences (prevents reward hacking)
        if not _is_valid_nbv1(seq_full):
            out.append(0.0)
            continue
        core = _nbv1_core(seq_full)
        patches = find_hydrophobic_patches_core(core)
        out.append(_clamp01(1.0 - float(patches.get("patch_score", 1.0))))
    return out


def liability_reward(completions: List[str], **kwargs) -> List[float]:
    """[0,1] where 1.0 = no liabilities (computed on NBv1 core).
    Gated: returns 0.0 for structurally invalid sequences."""
    out: List[float] = []
    for comp in completions:
        seq_full = clean_sequence(comp)
        # Validity gate: no reward for invalid sequences (prevents reward hacking)
        if not _is_valid_nbv1(seq_full):
            out.append(0.0)
            continue
        core = _nbv1_core(seq_full)
        liabilities = scan_sequence_liabilities_core(core)
        severity = float(liabilities.get("liability_severity", 0.0))
        # Diminishing penalty: severity 0 -> 1.0 ; severity 10 -> 0.5 ; severity 20 -> 0.33
        out.append(_clamp01(1.0 / (1.0 + severity / 10.0)))
    return out


def expression_reward(completions: List[str], **kwargs) -> List[float]:
    """[0,1] where 1.0 = high expression potential (computed on NBv1 core).
    Gated: returns 0.0 for structurally invalid sequences."""
    out: List[float] = []
    for comp in completions:
        seq_full = clean_sequence(comp)
        # Validity gate: no reward for invalid sequences (prevents reward hacking)
        if not _is_valid_nbv1(seq_full):
            out.append(0.0)
            continue
        core = _nbv1_core(seq_full)
        expr = calculate_expression_score_core(core)
        out.append(_clamp01(float(expr.get("expression_score", 0.0))))
    return out


# def nanobody_integrity_reward(completions: List[str], **kwargs) -> List[float]:
#     """
#     Nanobody structural integrity reward (NBv1).

#     This is the production-friendly replacement for a bare "length reward":
#     it encodes Yotta Nb-v1 scaffold validity + basic fold/sequence sanity.

#     Logic (high-level):
#       1) Must be valid Yotta Nb-v1 scaffold (length=126 and endswith GGGGS) -> else 0.0
#       2) Must pass strict basic validation (length=126, cysteines, no low-complexity) -> else 0.0
#       3) If both pass: return 0.5 + 0.5 * vhh_hallmark_score (graded quality signal)

#     Literature grounding (story):
#       - VHH domains have conserved scaffold constraints (Ig fold) and
#         Nb-v1 requires a specific terminal linker (Yotta library convention).
#       - VHH hallmark residues in FR2 are canonical features of camelid VHH.
#     """
#     out: List[float] = []
#     for comp in completions:
#         seq_full = clean_sequence(comp)

#         scaffold = nbv1_scaffold_integrity(seq_full)
#         if not scaffold["is_valid"]:
#             out.append(0.0)
#             continue

#         basic = validate_sequence_basic(seq_full, min_length=126, max_length=126, require_cysteines=True)
#         if not basic["is_valid"]:
#             out.append(0.0)
#             continue

#         core = _nbv1_core(seq_full)
#         hallmark = nbv1_vhh_hallmark_score(core)["vhh_hallmark_score"]
#         out.append(_clamp01(0.5 + 0.5 * float(hallmark)))

#     return out


def vhh_hallmark_reward(completions: List[str], **kwargs) -> List[float]:
    """[0,1] where 1.0 = VHH-like hallmarks at FR2 tetrad (NBv1 fixed indices).
    Gated: returns 0.0 for structurally invalid sequences."""
    out: List[float] = []
    for comp in completions:
        seq_full = clean_sequence(comp)
        # Validity gate: no reward for invalid sequences (prevents reward hacking)
        if not _is_valid_nbv1(seq_full):
            out.append(0.0)
            continue
        core = _nbv1_core(seq_full)
        out.append(_clamp01(float(nbv1_vhh_hallmark_score(core)["vhh_hallmark_score"])))
    return out


_HUBER_DELTA = 2.0     # quadratic-to-linear transition (AA deviation)
_HUBER_MAX_DEV = 10.0  # deviation where length score reaches 0
_HUBER_MAX_LOSS = _HUBER_DELTA * _HUBER_MAX_DEV - 0.5 * _HUBER_DELTA * _HUBER_DELTA


def _huber_length_score(n: int) -> float:
    """
    Huber-loss-based length proximity score [0, 1].

    Converts sequence length to a reward using the Huber loss function:
      |d| <= delta:  quadratic  (smooth, steepening gradient near target)
      |d| >  delta:  linear     (constant gradient, never vanishes)

    This is strictly superior to Gaussian for GDPO because:
      - Gaussian gradient vanishes exponentially far from target (d=8 -> ~0 signal)
      - Huber maintains constant -0.111/AA gradient everywhere beyond delta
      - When the policy destabilizes and generates 100-AA sequences, Huber still
        provides a strong push back toward 126; Gaussian gives nearly zero

    Scores:  d=0->1.000, d=1->0.972, d=2->0.889, d=3->0.778,
             d=5->0.556, d=8->0.222, d=10->0.000
    """
    d = abs(n - NBV1_LENGTH)
    if d <= _HUBER_DELTA:
        loss = 0.5 * d * d
    else:
        loss = _HUBER_DELTA * d - 0.5 * _HUBER_DELTA * _HUBER_DELTA
    return max(0.0, 1.0 - loss / _HUBER_MAX_LOSS)


def scaffold_integrity_reward(completions: List[str], **kwargs) -> List[float]:
    """
    Continuous scaffold integrity [0,1] with Huber-loss length proximity.

    Replaces the binary {0,1} version to prevent GDPO z-score collapse.
    Binary rewards create a variance cliff: once the model generates mostly-valid
    sequences, all get 1.0 -> zero variance -> z-score div-by-epsilon -> NaN crash.
    Additionally, the cliff between valid (all rewards computed) and invalid (all
    rewards = 0) causes sporadic large gradients that destabilize training.

    Components (weighted):
      40%  Length proximity  - Huber loss (delta=2, max_dev=10):
           Quadratic within +/-2 AA (smooth near target),
           linear beyond (constant gradient, never vanishes).
           126->1.00, 125/127->0.97, 124/128->0.89, 121/131->0.56, 116/136->0.00
      25%  C-terminal linker - 1.0 if ends with GGGGS, else 0.0
      15%  AA purity         - fraction of standard amino acids (continuous)
      20%  Cysteine adequacy - soft sigmoid cys/(cys+1.5), unsaturated at 2-4 Cys
           Guarantees non-zero variance among valid nanobodies
           (2 Cys -> 0.57, 3 Cys -> 0.67, 4 Cys -> 0.73)

    Example scores:
      Valid 126-AA, 2 Cys:  0.40*1.00 + 0.25 + 0.15 + 0.20*0.57 = 0.914
      Valid 126-AA, 3 Cys:  0.40*1.00 + 0.25 + 0.15 + 0.20*0.67 = 0.933
      Near-miss 125-AA:     0.40*0.97 + 0.00 + 0.15 + 0.20*0.57 = 0.652
      Far-miss 100-AA:      0.40*0.00 + 0.00 + 0.15 + 0.20*0.40 = 0.230
    """
    out: List[float] = []
    for comp in completions:
        seq_full = clean_sequence(comp)
        n = len(seq_full)

        if n == 0:
            out.append(0.0)
            continue

        # (1) Length proximity via Huber loss
        len_score = _huber_length_score(n)

        # (2) C-terminal linker (GGGGS) - binary structural check
        linker = 1.0 if seq_full.endswith("GGGGS") else 0.0

        # (3) AA alphabet purity - continuous fraction of valid amino acids
        aa_purity = sum(1 for aa in seq_full if aa in STANDARD_AA_SET) / n

        # (4) Cysteine adequacy - soft sigmoid, deliberately NOT saturated at 2 Cys
        #     so that valid nanobodies with 2, 3, 4 Cys get different scores,
        #     guaranteeing non-zero reward variance within every batch.
        cys = seq_full.count("C")
        cys_score = cys / (cys + 1.5)
         # then add cys score ( only to 2)
         
        reward = (0.40 * len_score
                  + 0.25 * linker
                  + 0.15 * aa_purity
                  + 0.20 * cys_score)

        out.append(_clamp01(reward))
    return out

