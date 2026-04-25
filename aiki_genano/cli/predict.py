"""``aiki-genano predict`` — score sequences with the developability profile.

Default mode runs the local profile (six GDPO rewards + motif counts +
biophysical descriptors + CDR/FR breakdown + Aggrescan) on every input
sequence. Adding ``--with-properties`` also runs the heavy external
predictors (TEMPRO Tm, NetSolP solubility, Sapiens humanness) — these
require GPU and the TEMPRO Keras model file mounted at
``$TEMPRO_MODEL_PATH``.

Inputs (auto-detected by ``--sequences`` extension):
    .csv     reads the ``generated_sequence`` column (falls back to ``binder``,
             then ``protein``, then the first non-metadata column).
    .fasta   one record per sequence, header used as ``id``.

Output: a CSV with the input rows plus one column per profile/property field.
The column names match the Zenodo ``generated_sequences/{MODEL}/properties/``
schema, so a downstream join just works.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple


_LIKELY_SEQUENCE_COLUMNS = ["generated_sequence", "binder", "protein", "sequence", "seq"]


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aiki-genano predict",
        description=(
            "Compute developability profile (rewards + motifs + biophysical + CDR + Aggrescan) "
            "for a list of nanobody sequences. Add --with-properties for TEMPRO Tm, "
            "NetSolP solubility, and Sapiens humanness."
        ),
    )
    p.add_argument("--sequences", required=True,
                   help="Input CSV (reads first matching sequence column) or FASTA.")
    p.add_argument("--sequence-column", default=None,
                   help="Override CSV column name (default: auto-detect from "
                        f"{_LIKELY_SEQUENCE_COLUMNS}).")
    p.add_argument("--with-properties", action="store_true",
                   help="Also run TEMPRO Tm, NetSolP solubility, Sapiens humanness "
                        "(GPU + TEMPRO Keras file required).")
    p.add_argument("--tempro-model",
                   default=os.environ.get("TEMPRO_MODEL_PATH", ""),
                   help="Path to TEMPRO Keras model "
                        "(or set $TEMPRO_MODEL_PATH). Only required with --with-properties.")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="Compute device for property predictors (default auto).")
    p.add_argument("--batch_size", type=int, default=25,
                   help="ESM-2 embedding batch size (default 25).")
    p.add_argument("--output",
                   default=os.environ.get("AIKI_GENANO_OUTPUT", "/app/output/predictions_profiled.csv"),
                   help="Output CSV path (default /app/output/predictions_profiled.csv).")
    return p


def _read_inputs(path: str, override_col: str | None) -> Tuple["pd.DataFrame", str]:
    """Return (rows, sequence_column_name). Rows include all original input columns."""
    import pandas as pd

    p = Path(path)
    if not p.exists():
        raise SystemExit(f"--sequences path not found: {p}")

    if p.suffix.lower() in (".fasta", ".fa", ".faa"):
        ids, seqs = [], []
        cur_id, cur_seq = None, []
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    ids.append(cur_id); seqs.append("".join(cur_seq))
                cur_id = line[1:].split()[0] if line[1:] else f"seq_{len(ids)}"
                cur_seq = []
            else:
                cur_seq.append(line.upper())
        if cur_id is not None:
            ids.append(cur_id); seqs.append("".join(cur_seq))
        return pd.DataFrame({"id": ids, "generated_sequence": seqs}), "generated_sequence"

    df = pd.read_csv(p)
    if override_col:
        if override_col not in df.columns:
            raise SystemExit(f"--sequence-column '{override_col}' not in {p}; "
                             f"have: {list(df.columns)}")
        col = override_col
    else:
        col = next((c for c in _LIKELY_SEQUENCE_COLUMNS if c in df.columns), None)
        if col is None:
            raise SystemExit(
                f"could not auto-detect a sequence column in {p}. "
                f"Tried {_LIKELY_SEQUENCE_COLUMNS}; have {list(df.columns)}. "
                f"Pass --sequence-column."
            )
    return df, col


def _local_profile(seqs: List[str]) -> "pd.DataFrame":
    import pandas as pd
    from aiki_genano.evaluation.profile import compute_sequence_profile

    rows = [compute_sequence_profile(s) for s in seqs]
    return pd.DataFrame(rows)


def _external_properties(seqs: List[str], args, device: str) -> "pd.DataFrame":
    """Run TEMPRO + NetSolP + Sapiens. Each is best-effort; failing predictors
    surface a clear NaN column with a warning to stderr (no silent fallback)."""
    import numpy as np
    import pandas as pd

    out = {"tempro_tm": [float("nan")] * len(seqs),
           "sapiens_humanness": [float("nan")] * len(seqs),
           "netsolp_solubility": [float("nan")] * len(seqs),
           "netsolp_usability": [float("nan")] * len(seqs)}

    # TEMPRO ----------------------------------------------------------------
    if not args.tempro_model:
        print("[predict] WARNING: TEMPRO Keras model path not provided; "
              "tempro_tm column will be NaN. Pass --tempro-model or set $TEMPRO_MODEL_PATH.",
              file=sys.stderr)
    elif not Path(args.tempro_model).exists():
        print(f"[predict] WARNING: TEMPRO model not found at {args.tempro_model}; "
              "tempro_tm column will be NaN.", file=sys.stderr)
    else:
        try:
            from aiki_genano.evaluation import run_tempro
            run_tempro.TEMPRO_MODEL = Path(args.tempro_model)
            run_tempro.BATCH_SIZE = args.batch_size
            run_tempro.DEVICE = device
            embeddings = run_tempro.generate_esm_embeddings(seqs)
            out["tempro_tm"] = list(map(float, run_tempro.predict_tm(embeddings)))
        except Exception as exc:
            print(f"[predict] WARNING: TEMPRO failed ({type(exc).__name__}: {exc}); "
                  "tempro_tm column will be NaN.", file=sys.stderr)

    # Sapiens humanness -----------------------------------------------------
    try:
        from aiki_genano.evaluation.run_sapiens_tempro import compute_sapiens_humanness
        out["sapiens_humanness"] = list(map(float, compute_sapiens_humanness(seqs)))
    except Exception as exc:
        print(f"[predict] WARNING: Sapiens humanness failed ({type(exc).__name__}: {exc}); "
              "sapiens_humanness column will be NaN.", file=sys.stderr)

    # NetSolP --------------------------------------------------------------
    try:
        import tempfile
        from aiki_genano.evaluation import run_netsolp
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            fasta = tmp / "in.fasta"
            netsolp_csv = tmp / "out.csv"
            with open(fasta, "w") as fh:
                for i, s in enumerate(seqs):
                    fh.write(f">seq_{i}\n{s}\n")
            run_netsolp.run_netsolp(fasta, netsolp_csv)
            ns = pd.read_csv(netsolp_csv)
            out["netsolp_solubility"] = ns["solubility"].astype(float).tolist() \
                if "solubility" in ns.columns else out["netsolp_solubility"]
            out["netsolp_usability"] = ns["usability"].astype(float).tolist() \
                if "usability" in ns.columns else out["netsolp_usability"]
    except Exception as exc:
        print(f"[predict] WARNING: NetSolP failed ({type(exc).__name__}: {exc}); "
              "netsolp_* columns will be NaN.", file=sys.stderr)

    return pd.DataFrame(out)


def _resolve_device(arg: str) -> str:
    if arg in ("cuda", "cpu"):
        return arg
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def main(argv: list[str]) -> int:
    args = _build_argparser().parse_args(argv)
    device = _resolve_device(args.device)

    df, col = _read_inputs(args.sequences, args.sequence_column)
    seqs = [str(s).strip().upper() for s in df[col].tolist()]
    print(f"[predict] {len(seqs):,} sequences from {args.sequences} "
          f"(column: '{col}')")

    profile = _local_profile(seqs)
    out = df.reset_index(drop=True).join(profile.reset_index(drop=True))

    if args.with_properties:
        print(f"[predict] running external predictors on device={device}…")
        ext = _external_properties(seqs, args, device)
        out = out.join(ext.reset_index(drop=True))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[predict] wrote {len(out):,} rows × {len(out.columns)} cols → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
