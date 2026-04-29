"""``aiki-genano generate`` — sample VHH candidates for an epitope.

Loads the SFT base + chosen LoRA adapter from ``--checkpoint-dir``, decodes
``--n_candidates`` ChatML completions for the supplied epitope, scores each
with the six GDPO reward functions, and writes a CSV.

The reward column dictionary matches the paper exactly so the output CSV
schema is consistent with the per-sequence property tables published in the
Zenodo deposit (``full_property_tables/{MODEL}_all_properties.csv``).

Heavy lifting (model load + tokenisation + sampling) is delegated to
``aiki_genano.training.inference`` so this entrypoint stays a thin CLI shell.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

# Locations of the four checkpoint subdirectories inside --checkpoint-dir.
# Trained checkpoints are NDA-gated (request via partnerships@aikium.com);
# place them under these subpaths after receipt.
_DEFAULT_CHECKPOINT_SUBDIRS = {
    "SFT": "SFT/NanoBody-design-sft-response-only-100k-len126-r64",
    "DPO": "DPO/checkpoint-6000",
    "GDPO_DPO": "GDPO_DPO/checkpoint-2000",
    "GDPO_SFT": "GDPO_SFT/checkpoint-2000",
}

_REWARD_COLUMNS = [
    "reward_fr2_aggregation",
    "reward_hydrophobic_patch",
    "reward_liability",
    "reward_expression",
    "reward_vhh_hallmark",
    "reward_scaffold_integrity",
]


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aiki-genano generate",
        description=(
            "Sample N candidate 126-AA VHH nanobody sequences for a target epitope, "
            "scored with the six GDPO reward functions. Writes a CSV whose schema "
            "matches the per-sequence property tables in the Zenodo deposit "
            "(full_property_tables/)."
        ),
    )
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument(
        "--epitope",
        type=str,
        help="Target epitope peptide sequence (single string, single AA alphabet).",
    )
    inp.add_argument(
        "--epitope-file",
        type=str,
        help="Path to a CSV with an 'epitope' column; one row per target.",
    )
    p.add_argument("--n_candidates", type=int, default=50,
                   help="Candidates per epitope (default 50, max 200).")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Sampling temperature (default 0.7, paper primary).")
    p.add_argument("--top_p", type=float, default=0.92,
                   help="Nucleus-sampling top-p (default 0.92, matches paper Stage-3 configs).")
    p.add_argument("--top_k", type=int, default=0,
                   help="Top-k filter; 0 disables (default).")
    p.add_argument("--max_new_tokens", type=int, default=256,
                   help="Decoder budget per candidate (default 256, fits 126-AA VHH + a few extra tokens).")
    p.add_argument("--model", choices=sorted(_DEFAULT_CHECKPOINT_SUBDIRS.keys()),
                   default="GDPO_DPO",
                   help="Which trained checkpoint to sample from (default GDPO_DPO — paper headline).")
    p.add_argument("--checkpoint-dir", type=str,
                   default=os.environ.get("AIKI_GENANO_CHECKPOINTS", "/app/checkpoints"),
                   help="Root dir holding SFT/, DPO/, GDPO_DPO/, GDPO_SFT/ subdirs from Zenodo. "
                        "Default: /app/checkpoints (Docker mount) or $AIKI_GENANO_CHECKPOINTS.")
    p.add_argument("--checkpoint-path", type=str, default=None,
                   help="Override --model/--checkpoint-dir lookup with an explicit checkpoint directory.")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="Compute device (default auto: use CUDA if available else CPU).")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Decoder batch size (default 8; tune to GPU memory).")
    p.add_argument("--seed", type=int, default=42,
                   help="Sampling seed (default 42, paper primary).")
    p.add_argument("--output", type=str,
                   default=os.environ.get("AIKI_GENANO_OUTPUT", "/app/output/predictions.csv"),
                   help="Output CSV path. Default: /app/output/predictions.csv.")
    return p


def _resolve_device(arg: str) -> str:
    if arg == "cuda":
        return "cuda"
    if arg == "cpu":
        return "cpu"
    try:
        import torch  # imported lazily so --help works without torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_checkpoint(args) -> Path:
    if args.checkpoint_path:
        path = Path(args.checkpoint_path)
    else:
        path = Path(args.checkpoint_dir) / _DEFAULT_CHECKPOINT_SUBDIRS[args.model]
    if not path.is_dir():
        raise SystemExit(
            f"Checkpoint directory not found: {path}\n"
            f"Run scripts/download_checkpoints.sh first, or pass --checkpoint-path."
        )
    return path


def _load_epitopes(args) -> List[str]:
    if args.epitope:
        return [args.epitope.strip().upper()]
    import pandas as pd
    df = pd.read_csv(args.epitope_file)
    col = "epitope" if "epitope" in df.columns else ("peptide" if "peptide" in df.columns else None)
    if col is None:
        raise SystemExit(
            f"--epitope-file {args.epitope_file} must contain an 'epitope' or 'peptide' column."
        )
    return [str(s).strip().upper() for s in df[col].tolist() if str(s).strip()]


def _generate_for_epitope(
    model, tokenizer, epitope: str, args, device: str
) -> List[str]:
    """Sample n_candidates completions for one epitope, returning raw decoder strings."""
    from aiki_genano.training.inference import generate_batch  # heavy import, lazy

    sampling_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
    }
    if args.top_k > 0:
        sampling_config["top_k"] = int(args.top_k)

    out: List[str] = []
    remaining = args.n_candidates
    while remaining > 0:
        bs = min(args.batch_size, remaining)
        batch_peptides = [epitope] * bs
        completions = generate_batch(
            model, tokenizer, batch_peptides, sampling_config, device=device
        )
        out.extend(completions)
        remaining -= bs
    return out


def _score(completions: List[str]) -> List[dict]:
    """Run the six paper reward functions on a list of decoder completions."""
    from aiki_genano.rewards.rewards import (
        clean_sequence,
        fr2_aggregation_reward,
        hydrophobic_patch_reward,
        liability_reward,
        expression_reward,
        vhh_hallmark_reward,
        scaffold_integrity_reward,
        NBV1_LENGTH,
    )

    fr2 = fr2_aggregation_reward(completions)
    hyd = hydrophobic_patch_reward(completions)
    lia = liability_reward(completions)
    exp = expression_reward(completions)
    vhh = vhh_hallmark_reward(completions)
    sca = scaffold_integrity_reward(completions)

    rows = []
    for raw, f, h, l, e, v, s in zip(completions, fr2, hyd, lia, exp, vhh, sca):
        seq = clean_sequence(raw)
        rows.append({
            "generated_sequence": seq,
            "gen_length": len(seq),
            "is_valid_126": len(seq) == NBV1_LENGTH and seq.endswith("GGGGS")
                             and seq.count("C") >= 2,
            "reward_fr2_aggregation": float(f),
            "reward_hydrophobic_patch": float(h),
            "reward_liability": float(l),
            "reward_expression": float(e),
            "reward_vhh_hallmark": float(v),
            "reward_scaffold_integrity": float(s),
        })
    return rows


def main(argv: list[str]) -> int:
    args = _build_argparser().parse_args(argv)

    if args.n_candidates < 1 or args.n_candidates > 200:
        raise SystemExit(f"--n_candidates must be in [1, 200], got {args.n_candidates}")
    if not (0.0 < args.temperature <= 5.0):
        raise SystemExit(f"--temperature must be in (0, 5], got {args.temperature}")
    if not (0.0 < args.top_p <= 1.0):
        raise SystemExit(f"--top_p must be in (0, 1], got {args.top_p}")

    device = _resolve_device(args.device)
    checkpoint = _resolve_checkpoint(args)
    epitopes = _load_epitopes(args)

    print(f"[generate] device={device} model={args.model} checkpoint={checkpoint}")
    print(f"[generate] epitopes={len(epitopes)} n_candidates={args.n_candidates} "
          f"T={args.temperature} top_p={args.top_p} seed={args.seed}")

    # Lazy heavy imports — keep --help fast and avoid torch import on argparse error.
    import torch
    import pandas as pd
    from aiki_genano.training.inference import load_model_from_checkpoint

    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model, tokenizer = load_model_from_checkpoint(str(checkpoint), device=device)

    all_rows: List[dict] = []
    for epitope in epitopes:
        completions = _generate_for_epitope(model, tokenizer, epitope, args, device)
        rows = _score(completions)
        for r in rows:
            r["epitope"] = epitope
            r["model"] = args.model
            r["seed"] = args.seed
            r["temperature"] = float(args.temperature)
        all_rows.extend(rows)

    column_order = [
        "epitope",
        "generated_sequence", "gen_length", "is_valid_126",
        "model", "seed", "temperature",
    ] + _REWARD_COLUMNS
    df = pd.DataFrame(all_rows)[column_order]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    n_valid = int(df["is_valid_126"].sum())
    print(
        f"[generate] wrote {len(df):,} rows ({n_valid:,} valid 126-AA) → {out}"
    )

    if n_valid == 0:
        print("[generate] WARNING: zero valid 126-AA sequences. Check checkpoint, "
              "tokenizer chat template, and sampling parameters.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
