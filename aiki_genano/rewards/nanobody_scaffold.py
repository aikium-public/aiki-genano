"""
Nb-v1 nanobody scaffold: constants, dataclass, and sequence normalization.

The 126-residue VHH backbone used throughout this work follows the synthetic
nanobody library described by Contreras et al. (2023): a 121-AA core VHH
(FR1-CDR1-FR2-CDR2-FR3-CDR3-FR4) plus a five-residue C-terminal GGGGS linker
carried over from the display construct. CDR boundaries are IMGT-style
(0-indexed, end-exclusive).

This module is the single source of truth for Nb-v1 scaffold geometry. Reward
functions and evaluation scripts import from here rather than hard-coding
positions.

Public API:
    NB_V1_REFERENCE     - canonical 126-AA reference sequence.
    NBV1                - NanobodyScaffold instance (length, CDRs, linker).
    NanobodyScaffold    - frozen dataclass describing a scaffold.
    NormalizedSequence  - result of normalize_for_prediction.
    normalize_for_prediction(seq)   - strip the GGGGS linker before property
                                      predictors that expect the core VHH.
    get_core_sequence(seq)          - convenience: return just the stripped core.
    validate_nbv1_sequence(seq)     - check length + linker; (ok, message).

References:
    Contreras, J. et al. "A synthetic nanobody library for ..." (2023).
    Muyldermans, S. "Nanobodies: natural single-domain antibodies."
        Annu Rev Biochem 82:775-797 (2013).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class NanobodyScaffold:
    """Static geometry of a nanobody scaffold (length, CDR ranges, linker)."""

    name: str
    length: int
    cdr1: Tuple[int, int]
    cdr2: Tuple[int, int]
    cdr3: Tuple[int, int]
    required_suffix: Optional[str] = None
    required_prefix: Optional[str] = None


# Nb-v1: 126 AA = 121-AA core VHH + 5-AA GGGGS C-terminal linker.
# CDR boundaries follow IMGT numbering validated on the reference backbone.
NBV1 = NanobodyScaffold(
    name="nbv1",
    length=126,
    cdr1=(25, 36),
    cdr2=(53, 61),
    cdr3=(99, 111),
    required_suffix="GGGGS",
)

NB_V1_REFERENCE = (
    "QVQLVESGGGSVQAGGSLRLSCTASGGSEYSYSTFSLGWFRQAPGQEREAVAAIASMGGLTYYADSVKG"
    "RFTISRDNAKNTVTLQMNNLKPVDTAIYYCAAVRGYFMRLPSWGQGTQVTVSGGGGS"
)
assert len(NB_V1_REFERENCE) == NBV1.length
assert NBV1.required_suffix is not None and NB_V1_REFERENCE.endswith(NBV1.required_suffix)


@dataclass(frozen=True)
class NormalizedSequence:
    """Output of normalize_for_prediction: core sequence + stripping metadata."""

    original: str
    core: str
    stripped_suffix: str
    original_length: int
    core_length: int


def normalize_for_prediction(
    sequence: str, strip_termini: bool = True
) -> NormalizedSequence:
    """Strip the Nb-v1 GGGGS linker so property predictors see the core VHH.

    Predictors such as TEMPRO, NetSolP, and Sapiens are trained on VHH domains
    without display-construct linkers; feeding them the raw 126-AA sequence
    biases downstream scores. Call this once at the entry point and pass
    ``.core`` to every predictor.
    """
    seq = sequence.upper().strip()
    suffix = NBV1.required_suffix
    if strip_termini and suffix and seq.endswith(suffix):
        core = seq[: -len(suffix)]
        return NormalizedSequence(seq, core, suffix, len(seq), len(core))
    return NormalizedSequence(seq, seq, "", len(seq), len(seq))


def get_core_sequence(sequence: str) -> str:
    """Return just the core VHH (no C-terminal linker) for property prediction."""
    return normalize_for_prediction(sequence).core


def validate_nbv1_sequence(
    sequence: str, raise_on_error: bool = True
) -> Tuple[bool, str]:
    """Length + required-suffix sanity check for the Nb-v1 backbone.

    Returns ``(is_valid, message)``. When ``raise_on_error=True`` (default),
    a failing check raises ``ValueError`` instead of returning False.
    """
    if len(sequence) != NBV1.length:
        msg = f"Length mismatch: expected {NBV1.length}, got {len(sequence)}"
        if raise_on_error:
            raise ValueError(msg)
        return False, msg
    suffix = NBV1.required_suffix
    if suffix and not sequence.endswith(suffix):
        msg = (
            f"Nb-v1 ({NBV1.length} AA) must end with '{suffix}'; "
            f"got ...{sequence[-10:]}"
        )
        if raise_on_error:
            raise ValueError(msg)
        return False, msg
    return True, "Valid Nb-v1 sequence"
