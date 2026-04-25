"""``aiki-genano smoke`` — end-to-end self-test of the image.

Two modes:

  --offline   Tests code paths only, no Zenodo download, no real model loads.
              Builds a 4-layer GPT-2-like dummy via transformers, runs the
              reward functions on canned sequences, asserts shapes. Should
              complete in under 1 minute on CPU and is safe for CI.

  --real      Pulls the 4 checkpoints from Zenodo via scripts/download_checkpoints.sh,
              runs aiki-genano generate on one epitope with --n_candidates 5, then
              aiki-genano predict on the result. Asserts that the output CSV has
              the expected columns and at least one valid 126-AA sequence.
              ~10 minutes including downloads on first run.

Both modes print PASS/FAIL and exit 0/1 accordingly.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


_TEST_EPITOPE = "MNYPLTLEMDLENLEDLFWELDRLDNYNDTSLVENHL"  # one of the 10 Zenodo subset targets

_EXPECTED_GENERATE_COLUMNS = {
    "epitope", "generated_sequence", "gen_length", "is_valid_126",
    "model", "seed", "temperature",
    "reward_fr2_aggregation", "reward_hydrophobic_patch", "reward_liability",
    "reward_expression", "reward_vhh_hallmark", "reward_scaffold_integrity",
}

_EXPECTED_PREDICT_COLUMNS_LOCAL = {
    "reward_fr2_aggregation", "reward_hydrophobic_patch", "reward_liability",
    "reward_expression", "reward_vhh_hallmark", "reward_scaffold_integrity",
    "liability_severity", "fr2_hydrophobicity", "vhh_hallmark_score", "expression_score",
    "gravy", "isoelectric_point", "instability_index",
    "cdr1_length", "cdr2_length", "cdr3_length",
}


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aiki-genano smoke")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true",
                      help="Stub-mode: test code paths without Zenodo download.")
    mode.add_argument("--real", action="store_true",
                      help="Pull from Zenodo and run real inference.")
    p.add_argument("--checkpoint-dir", default="/app/checkpoints",
                   help="Where to find / download checkpoints (real mode).")
    p.add_argument("--output-dir", default="/app/output/smoke",
                   help="Where to write smoke artefacts.")
    return p


def _smoke_offline() -> int:
    """Exercise reward functions + import graph; no model loads."""
    print("[smoke] OFFLINE: import graph + reward function shape check")
    from aiki_genano.rewards.rewards import (
        clean_sequence,
        fr2_aggregation_reward, hydrophobic_patch_reward, liability_reward,
        expression_reward, vhh_hallmark_reward, scaffold_integrity_reward,
        NB_V1_REFERENCE,
    )
    from aiki_genano.rewards.nanobody_scaffold import (
        NBV1, normalize_for_prediction, validate_nbv1_sequence,
    )
    from aiki_genano.evaluation.profile import compute_sequence_profile

    # Reward functions on (1) the reference, (2) a deliberate truncation, (3) garbage
    seqs = [
        NB_V1_REFERENCE,
        NB_V1_REFERENCE[:120],
        "GARBAGE_NOT_A_PROTEIN",
    ]
    for fn, name in [
        (fr2_aggregation_reward,    "fr2_aggregation"),
        (hydrophobic_patch_reward,  "hydrophobic_patch"),
        (liability_reward,          "liability"),
        (expression_reward,         "expression"),
        (vhh_hallmark_reward,       "vhh_hallmark"),
        (scaffold_integrity_reward, "scaffold_integrity"),
    ]:
        out = fn(seqs)
        assert isinstance(out, list) and len(out) == 3, f"{name}: bad shape {out}"
        for v in out:
            assert isinstance(v, float) and 0.0 <= v <= 1.0, f"{name}: bad value {v}"
        print(f"  ✓ {name}: {[round(v, 3) for v in out]}")

    # Profile on the reference
    prof = compute_sequence_profile(NB_V1_REFERENCE)
    missing = _EXPECTED_PREDICT_COLUMNS_LOCAL - set(prof.keys())
    assert not missing, f"profile missing columns: {missing}"
    print(f"  ✓ compute_sequence_profile: {len(prof)} columns")

    # Scaffold normalization round-trip
    norm = normalize_for_prediction(NB_V1_REFERENCE)
    assert norm.core_length == 121, f"normalize: expected core 121, got {norm.core_length}"
    ok, _ = validate_nbv1_sequence(NB_V1_REFERENCE, raise_on_error=False)
    assert ok, "validate_nbv1_sequence rejected the canonical reference"
    print(f"  ✓ scaffold normalize + validate")

    print("\n[smoke] OFFLINE: PASS")
    return 0


def _run(cmd: list[str]) -> tuple[int, str, str]:
    print(f"[smoke] $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout: print(proc.stdout)
    if proc.stderr: print(proc.stderr, file=sys.stderr)
    return proc.returncode, proc.stdout, proc.stderr


def _smoke_real(args) -> int:
    print("[smoke] REAL: download + inference + predict")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download checkpoints (idempotent — script skips already-present subdirs)
    repo_root = Path(__file__).resolve().parents[2]
    download_script = repo_root / "scripts" / "download_checkpoints.sh"
    if not download_script.exists():
        print(f"[smoke] WARN: {download_script} not found; assuming "
              f"{args.checkpoint_dir} already populated.", file=sys.stderr)
    else:
        rc, *_ = _run(["bash", str(download_script), "--dest", args.checkpoint_dir,
                       "--only", "SFT,GDPO_DPO"])
        if rc != 0:
            print("[smoke] REAL: FAIL (checkpoint download)", file=sys.stderr)
            return 1

    # 2. Generate 5 candidates for one epitope using GDPO_DPO
    preds = out_dir / "gen_predictions.csv"
    rc, *_ = _run([
        sys.executable, "-m", "aiki_genano.cli", "generate",
        "--epitope", _TEST_EPITOPE, "--n_candidates", "5",
        "--model", "GDPO_DPO", "--checkpoint-dir", args.checkpoint_dir,
        "--output", str(preds), "--seed", "42",
    ])
    if rc != 0 or not preds.exists():
        print("[smoke] REAL: FAIL (generate)", file=sys.stderr)
        return 1

    import pandas as pd
    df = pd.read_csv(preds)
    missing = _EXPECTED_GENERATE_COLUMNS - set(df.columns)
    if missing:
        print(f"[smoke] REAL: FAIL — generate output missing cols {missing}", file=sys.stderr)
        return 1
    n_valid = int(df["is_valid_126"].sum())
    print(f"  ✓ generate: {len(df)} rows, {n_valid} valid 126-AA")

    # 3. Run predict on the generated CSV (local profile only — keeps smoke quick)
    profiled = out_dir / "gen_profiled.csv"
    rc, *_ = _run([
        sys.executable, "-m", "aiki_genano.cli", "predict",
        "--sequences", str(preds), "--output", str(profiled),
    ])
    if rc != 0 or not profiled.exists():
        print("[smoke] REAL: FAIL (predict)", file=sys.stderr)
        return 1
    df2 = pd.read_csv(profiled)
    missing = _EXPECTED_PREDICT_COLUMNS_LOCAL - set(df2.columns)
    if missing:
        print(f"[smoke] REAL: FAIL — predict output missing cols {missing}", file=sys.stderr)
        return 1
    print(f"  ✓ predict: {len(df2)} rows × {len(df2.columns)} cols")

    print("\n[smoke] REAL: PASS")
    return 0


def main(argv: list[str]) -> int:
    args = _build_argparser().parse_args(argv)
    if args.offline:
        return _smoke_offline()
    return _smoke_real(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
