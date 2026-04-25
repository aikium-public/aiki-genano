from __future__ import annotations
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple
from aiki_genano.rewards.nanobody_scaffold import (
    NB_V1_REFERENCE,
    NBV1,
    normalize_for_prediction,
)

# =============================================================================
# Core constants (copied here to avoid dependence on fast_properties.py)
# =============================================================================

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
STANDARD_AA_SET = set(STANDARD_AA)

# Kyte-Doolittle hydropathy scale (reference: Kyte & Doolittle, J Mol Biol 1982)
# https://www.sciencedirect.com/science/article/pii/0022283682905150 ( Table 2)
KD_HYDROPATHY: Dict[str, float] = {
    "I": 4.5,
    "V": 4.2,
    "L": 3.8,
    "F": 2.8,
    "C": 2.5,
    "M": 1.9,
    "A": 1.8,
    "G": -0.4,
    "T": -0.7,
    "S": -0.8,
    "W": -0.9,
    "Y": -1.3,
    "P": -1.6,
    "H": -3.2,
    "E": -3.5,
    "Q": -3.5,
    "D": -3.5,
    "N": -3.5,
    "K": -3.9,
    "R": -4.5,
}

def kd_minmax_normalize(kd_dict: Dict[str, float]) -> Dict[str, float]:
    vmin = min(kd_dict.values())  # -4.5
    vmax = max(kd_dict.values())  #  4.5
    return {aa: (val - vmin) / (vmax - vmin) for aa, val in kd_dict.items()}

# Normalized hydrophobicity for aggregation scoring (0..1, higher=more hydrophobic)
HYDROPHOBICITY_NORM = kd_minmax_normalize(KD_HYDROPATHY)

# VHH FR2 hallmark positions and residues (Kabat numbering) from VHH literature.
# Format: position -> (good/VHH-like residues, bad/aggregation-prone residues)
VHH_FR2_POSITIONS: Dict[int, Tuple[set[str], set[str]]] = {
    37: ({"E", "Q", "G", "A", "S"}, {"F", "Y", "W", "L", "I", "V"}),
    44: ({"E", "Q", "R", "K", "G"}, {"W", "F", "Y", "L", "I", "V"}),
    45: ({"R", "K", "Q", "E", "L"}, {"W", "F", "Y", "I", "V", "M"}),
    47: ({"G", "S", "A", "L"}, {"W", "F", "Y", "I", "V"}),
}
# check the following:
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

# --------------------------------------------------------------------------------
# Hydrophobic patches detection
# --------------------------------------------------------------------------------
HYDROPHOBIC_PATCH_AA = set("ILVFM") # based on KD values (hydrophobic)
# https://link.springer.com/article/10.1186/1471-2105-8-65 (hotspots detection)
HYDROPHOBIC_MIN_PATCH_LEN = 5
def find_hydrophobic_patches_core(
    sequence_core: str,
    *,
    min_patch_length: int = HYDROPHOBIC_MIN_PATCH_LEN,
) -> Dict[str, object]:
    """
    Consecutive-hydrophobic-stretch detector (sequence-only aggregation proxy).

    Scans for runs of >= `min_patch_length` residues from HYDROPHOBIC_PATCH_AA.
    Returns patch_score = patch_fraction (fraction of core residues in patches),
    used directly as the [0,1] aggregation risk score.
    No arbitrary scaling — GDPO z-scoring normalises within each group.
    """
    seq = sequence_core.upper()
    n = len(seq)
    if n == 0:
        return {"n_patches": 0, "total_patch_residues": 0,
                "patch_fraction": 0.0, "patch_score": 0.0}

    patches: List[Tuple[int, int]] = []
    current_start: Optional[int] = None
    current_len = 0

    for i, aa in enumerate(seq):
        if aa in HYDROPHOBIC_PATCH_AA:
            if current_start is None:
                current_start = i
            current_len += 1
        else:
            if current_len >= min_patch_length and current_start is not None:
                patches.append((current_start, i))
            current_start = None
            current_len = 0

    if current_len >= min_patch_length and current_start is not None:
        patches.append((current_start, n))

    total_patch_residues = sum(end - start for start, end in patches)
    patch_fraction = total_patch_residues / n

    return {
        "n_patches": len(patches),
        "total_patch_residues": total_patch_residues,
        "patch_fraction": patch_fraction,
        "patch_score": patch_fraction,
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

    # Oxidation-prone Met (CDR-only): surface-exposed CDR Met are relevant;
    # framework Met are often buried and should not be penalized equally.
    cdr_ranges = [NBV1_CDR1, NBV1_CDR2, NBV1_CDR3]
    for m in re.finditer("M", seq):
        pos = m.start()
        if any(start <= pos < end for start, end in cdr_ranges):
            add_hit("oxidation", 2)

    # Integrin-like motifs (RGD/NGR/LDV) are retained as an off-target screen,
    # but excluded from chemical-liability severity to avoid conflation.
    integrin_hits = 0
    for motif in INTEGRIN_MOTIFS:
        integrin_hits += len(list(re.finditer(motif, seq)))
    if integrin_hits:
        hits_by_type["integrin_binding_screen"] = integrin_hits

    # Charge clusters: detect 4+ charged residues in a 6-aa window, then merge
    # overlapping windows into distinct clusters to avoid overcounting.
    charged = set("DEKR")
    cluster_flags = [False] * len(seq)
    for i in range(max(0, len(seq) - 5)):
        window = seq[i : i + 6]
        if sum(1 for aa in window if aa in charged) >= 4:
            for j in range(i, min(i + 6, len(seq))):
                cluster_flags[j] = True

    in_cluster = False
    n_clusters = 0
    for flag in cluster_flags:
        if flag and not in_cluster:
            n_clusters += 1
            in_cluster = True
        elif not flag:
            in_cluster = False
    for _ in range(n_clusters):
        add_hit("charge_cluster", 2)

    return {
        "liability_severity": severity,
        "hits_by_type": hits_by_type,
    }

# Inspired from the following repository:
# https://github.com/brunoV/bio-tools-solubility-wilkinson?tab=readme-ov-file
_WH_L1  =  15.43    
_WH_L2  = -29.56    
_WH_CVP =   1.71    
_WH_A   =   0.4934  
_WH_B   =   0.276   
_WH_C   =  -0.0392  
def calculate_expression_score_core(sequence_core: str) -> Dict[str, float]:
    """
    E. coli soluble expression probability (Wilkinson-Harrison model)
    """
    seq = sequence_core.upper()
    n = len(seq)
    if n == 0:
        return {"expression_score": 0.0}

    aa = Counter(seq)

    turn_fraction = (aa.get("N", 0) + aa.get("G", 0)
                     + aa.get("P", 0) + aa.get("S", 0)) / n
    charge_avg = abs((aa.get("R", 0) + aa.get("K", 0)
                      - aa.get("D", 0) - aa.get("E", 0)) / n - 0.03)

    cv = _WH_L1 * turn_fraction + _WH_L2 * charge_avg
    cv_norm = abs(cv - _WH_CVP)
    probability = _WH_A + _WH_B * cv_norm + _WH_C * cv_norm ** 2

    return {"expression_score": _clamp01(probability)}


def nbv1_fr2_aggregation_score(sequence_core: str) -> float:
    """
    NBv1 FR2 hydrophobicity risk: mean KD-normalized hydropathy over FR2.

    FR2 (positions 36-53) is solvent-exposed in VHH domains (no VL partner).
    Higher mean hydrophobicity → higher aggregation risk.
    Returns a single float in [0,1], higher = worse.
    """
    if len(sequence_core) < NBV1_FR2[1]:
        return 1.0

    fr2 = sequence_core[NBV1_FR2[0]:NBV1_FR2[1]]
    if not fr2:
        return 1.0

    vals = [HYDROPHOBICITY_NORM.get(aa, 0.0) for aa in fr2]
    return _clamp01(sum(vals) / len(vals))

def nbv1_vhh_hallmark_score(sequence_core: str) -> float:
    """
    VHH hallmark score at FR2 tetrad (Kabat 37/44/45/47).

    Categorical scoring with KD-based fallback for unclassified residues:
      - VHH-canonical residue  → 1.0
      - VH-interface residue   → 0.0
      - Other                  → 1 - KD_norm (hydrophilicity at solvent-exposed position)

    Returns mean score over 4 positions [0,1].
    """
    scores: List[float] = []

    for kabat_pos, idx in NBV1_HALLMARK_KABAT_TO_INDEX.items():
        if idx < 0 or idx >= len(sequence_core):
            scores.append(0.0)
            continue
        residue = sequence_core[idx]
        good, bad = VHH_FR2_POSITIONS[kabat_pos]
        if residue in good:
            scores.append(1.0)
        elif residue in bad:
            scores.append(0.0)
        else:
            scores.append(_clamp01(1.0 - HYDROPHOBICITY_NORM.get(residue, 0.0)))

    return _clamp01(sum(scores) / len(scores)) if scores else 0.0



_HUBER_DELTA = 2.0
_HUBER_MAX_DEV = 10.0
_HUBER_MAX_LOSS = _HUBER_DELTA * _HUBER_MAX_DEV - 0.5 * _HUBER_DELTA ** 2


def _huber_length_score(n: int) -> float:
    """Huber-loss length proximity [0,1]. d=0→1.0, d=2→0.89, d=5→0.56, d=10→0.0"""
    d = abs(n - NBV1_LENGTH)
    if d <= _HUBER_DELTA:
        loss = 0.5 * d * d
    else:
        loss = _HUBER_DELTA * d - 0.5 * _HUBER_DELTA ** 2
    return max(0.0, 1.0 - loss / _HUBER_MAX_LOSS)


def scaffold_integrity_score(sequence_full: str) -> float:
    """
    Continuous scaffold integrity [0,1] for GDPO gradient signal.

    50%  Length proximity (Huber loss)
    30%  C-terminal GGGGS linker (Yotta NBv1 construct requirement)
    20%  Cysteine == 2 (canonical Cys23-Cys104 disulfide, Muyldermans 2013)
    """
    seq = sequence_full.upper()
    n = len(seq)
    if n == 0:
        return 0.0
    len_score = _huber_length_score(n)
    linker = 1.0 if seq.endswith("GGGGS") else 0.0
    cys_score = 1.0 if seq.count("C") == 2 else 0.0
    return _clamp01(0.50 * len_score + 0.30 * linker + 0.20 * cys_score)


# =============================================================================
# TRL reward function wrappers (List[str] -> List[float])
# These are passed directly to the GDPO trainer as reward_funcs.
# =============================================================================

def fr2_aggregation_reward(completions: List[str], **kwargs) -> List[float]:
    """[0,1] where 1.0 = low FR2 hydrophobicity risk. Gated."""
    out: List[float] = []
    for comp in completions:
        seq_full = clean_sequence(comp)
        if not _is_valid_nbv1(seq_full):
            out.append(0.0)
            continue
        core = _nbv1_core(seq_full)
        out.append(_clamp01(1.0 - nbv1_fr2_aggregation_score(core)))
    return out


def hydrophobic_patch_reward(completions: List[str], **kwargs) -> List[float]:
    """[0,1] where 1.0 = no hydrophobic patches. Gated."""
    out: List[float] = []
    for comp in completions:
        seq_full = clean_sequence(comp)
        if not _is_valid_nbv1(seq_full):
            out.append(0.0)
            continue
        core = _nbv1_core(seq_full)
        score = find_hydrophobic_patches_core(core)["patch_score"]
        out.append(_clamp01(1.0 - float(score)))
    return out


_LIABILITY_K = 10.0
def liability_reward(completions: List[str], **kwargs) -> List[float]:
    """[0,1] where 1.0 = no liabilities. Gated.
    Mapping: 1/(1 + severity/k), k=10 (NBv1 reference severity ≈ 13)."""
    out: List[float] = []
    for comp in completions:
        seq_full = clean_sequence(comp)
        if not _is_valid_nbv1(seq_full):
            out.append(0.0)
            continue
        core = _nbv1_core(seq_full)
        severity = float(scan_sequence_liabilities_core(core)["liability_severity"])
        out.append(_clamp01(1.0 / (1.0 + severity / _LIABILITY_K)))
    return out


def expression_reward(completions: List[str], **kwargs) -> List[float]:
    """[0,1] where 1.0 = high expression potential (Wilkinson-Harrison). Gated."""
    out: List[float] = []
    for comp in completions:
        seq_full = clean_sequence(comp)
        if not _is_valid_nbv1(seq_full):
            out.append(0.0)
            continue
        core = _nbv1_core(seq_full)
        out.append(_clamp01(float(calculate_expression_score_core(core)["expression_score"])))
    return out


def vhh_hallmark_reward(completions: List[str], **kwargs) -> List[float]:
    """[0,1] where 1.0 = VHH-like hallmarks at FR2 tetrad. Gated."""
    out: List[float] = []
    for comp in completions:
        seq_full = clean_sequence(comp)
        if not _is_valid_nbv1(seq_full):
            out.append(0.0)
            continue
        core = _nbv1_core(seq_full)
        out.append(_clamp01(nbv1_vhh_hallmark_score(core)))
    return out
# add conventions 

def scaffold_integrity_reward(completions: List[str], **kwargs) -> List[float]:
    """[0,1] continuous scaffold integrity (Huber length + linker + cysteine).
    NOT gated — runs on all sequences to provide gradient for invalid ones."""
    out: List[float] = []
    for comp in completions:
        seq_full = clean_sequence(comp)
        out.append(scaffold_integrity_score(seq_full))
    return out
