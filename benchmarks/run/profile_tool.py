"""
End-to-end profile driver for one competitor tool's CSV.

Inputs:  `<tool>_seed42_temp0.7.csv` from fasta_to_profile_csv.py
Outputs: `<tool>_seed42_temp0.7_profiled.csv` with the same column schema as
         the cached Aiki-GeNano profiled CSVs, so downstream aggregation
         and comparison notebooks work unchanged.

Predictor order (fast → slow):
  1. Local profile (motifs + biophysical + NBv1 CDR slicing) — always run
  2. Sapiens humanness — VHH-only; skipped on peptide_mode
  3. TEMPRO Tm — VHH-only; skipped on peptide_mode
  4. NetSolP (ESM1b, 5-fold ensemble) — always run; trained on general proteins

Guardrails (silent-fallback audit):
- Skipped predictor columns are filled with NaN, NOT 0, NOT the cached mean.
- NetSolP column lookup uses EXACT names — no substring match.
- NetSolP rows aligned 1:1 by sid via int-extracted join; hard-assert all-matched.
- Any predictor failure on a row writes NaN to that row's column for THAT
  predictor only (other predictors continue). We never crash the whole run
  because one input is malformed — but we DO print the sequence idx loudly.
- peptide_mode tag on input CSV is honoured: VHH-only predictors skipped.

Run from repo root inside the `tempro` conda env:
    PYTHONUNBUFFERED=1 python benchmarks/run/profile_tool.py \\
        --in data/generated_2026_04_24/nanobert/nanobert_seed42_temp0.7.csv \\
        --out data/generated_2026_04_24/nanobert/nanobert_seed42_temp0.7_profiled.csv
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from aiki_genano.evaluation.profile import compute_sequence_profile

NETSOLP_DIR = Path("/opt/netsolp")
TEMPRO_MODEL = Path("/models/tempro/ESM_650M.keras")
DEVICE = "cuda"

NETSOLP_MODEL_TYPE = "ESM1b"


def fatal(msg: str):
    sys.stderr.write(f"\nFATAL: {msg}\n")
    sys.exit(2)


def step_local_profile(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== STEP 1: local profile (motifs + biophysical + CDR) ===")
    t0 = time.time()
    prof = df["generated_sequence"].astype(str).apply(compute_sequence_profile).apply(pd.Series)
    print(f"  done in {time.time()-t0:.1f}s; {len(prof.columns)} columns")
    return prof


def step_sapiens(df: pd.DataFrame, mask: pd.Series) -> pd.Series:
    print(f"\n=== STEP 2: Sapiens humanness (n={int(mask.sum())} eligible) ===")
    import sapiens
    t0 = time.time()
    scores = np.full(len(df), np.nan, dtype=float)
    idxs = np.where(mask.values)[0]
    for k, i in enumerate(idxs):
        seq = str(df["generated_sequence"].iloc[i])
        try:
            prob_df = sapiens.predict_scores(seq, "H")
            probs = [prob_df.iloc[j][seq[j]] for j in range(len(seq))]
            scores[i] = float(np.mean(probs))
        except Exception as e:
            print(f"    WARN: sapiens failed on idx {i} ({e.__class__.__name__}: {e}); leaving NaN")
        if (k + 1) % 200 == 0:
            print(f"    {k+1}/{len(idxs)}")
    print(f"  done in {time.time()-t0:.1f}s; non-NaN {np.isfinite(scores).sum()}/{len(df)}")
    return pd.Series(scores, name="sapiens_humanness", index=df.index)


def _build_esm_embeds(seqs, batch_size=8):
    import torch, esm
    print("  loading ESM-2 650M …")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    bc = alphabet.get_batch_converter()
    model.eval().to(DEVICE)
    embeds = []
    data = [(f"s{i}", s) for i, s in enumerate(seqs)]
    for start in range(0, len(data), batch_size):
        chunk = data[start:start+batch_size]
        _, _, toks = bc(chunk)
        toks = toks.to(DEVICE)
        with torch.no_grad():
            r = model(toks, repr_layers=[33], return_contacts=False)
        rep = r["representations"][33]
        lens = (toks != alphabet.padding_idx).sum(1)
        for i, L in enumerate(lens):
            emb = rep[i, 1:L-1].mean(0).cpu()
            embeds.append(emb)
        if (start // batch_size) % 25 == 0 or start + batch_size >= len(data):
            print(f"    embed {min(start+batch_size,len(data))}/{len(data)}")
    del model
    torch.cuda.empty_cache()
    return torch.stack(embeds).numpy()


def _load_tempro_keras():
    import h5py, json as _j
    import tensorflow as tf
    with h5py.File(str(TEMPRO_MODEL), "r") as f:
        config = _j.loads(f.attrs["model_config"])
    keras_model = tf.keras.models.model_from_json(_j.dumps(config))
    keras_model.load_weights(str(TEMPRO_MODEL))
    return keras_model


def step_tempro(df: pd.DataFrame, mask: pd.Series) -> pd.Series:
    print(f"\n=== STEP 3: TEMPRO Tm (n={int(mask.sum())} eligible) ===")
    t0 = time.time()
    idxs = np.where(mask.values)[0]
    scores = np.full(len(df), np.nan, dtype=float)
    if len(idxs) == 0:
        print("  no eligible rows; skipping")
        return pd.Series(scores, name="tempro_tm", index=df.index)
    seqs = [str(df["generated_sequence"].iloc[i]) for i in idxs]
    emb = _build_esm_embeds(seqs, batch_size=8)
    keras_model = _load_tempro_keras()
    pred = keras_model.predict(emb, batch_size=32, verbose=0).flatten()
    scores[idxs] = pred
    print(f"  done in {time.time()-t0:.1f}s")
    return pd.Series(scores, name="tempro_tm", index=df.index)


def step_netsolp(df: pd.DataFrame, work_dir: Path, model_type: str = NETSOLP_MODEL_TYPE) -> pd.DataFrame:
    print(f"\n=== STEP 4: NetSolP ({model_type}) on all n={len(df)} sequences ===")
    t0 = time.time()
    work_dir.mkdir(parents=True, exist_ok=True)
    fasta = work_dir / f"netsolp_input_{model_type}.fasta"
    with open(fasta, "w") as fh:
        for i in range(len(df)):
            fh.write(f">s{i}\n{df['generated_sequence'].iloc[i]}\n")
    out_csv = work_dir / f"netsolp_output_{model_type}.csv"
    if out_csv.exists():
        out_csv.unlink()
    cmd = [
        sys.executable, str(NETSOLP_DIR / "predict.py"),
        "--FASTA_PATH", str(fasta),
        "--OUTPUT_PATH", str(out_csv),
        "--MODELS_PATH", str(NETSOLP_DIR / "models"),
        "--MODEL_TYPE", model_type,
        "--PREDICTION_TYPE", "SU",
    ]
    print(f"  cmd: {' '.join(cmd[-8:])}")
    subprocess.run(cmd, check=True, cwd=str(NETSOLP_DIR))
    res = pd.read_csv(out_csv)
    required = {"sid", "predicted_solubility", "predicted_usability"}
    missing = required - set(res.columns)
    if missing:
        fatal(f"NetSolP output missing expected columns: {missing}; got {list(res.columns)}")
    # Strict row alignment
    ord_series = res["sid"].str.extract(r"^s(\d+)$")[0]
    if ord_series.isna().any():
        fatal(f"NetSolP returned sid not matching 's\\d+': "
              f"{res['sid'][ord_series.isna()].head().tolist()}")
    res = res.assign(__order=ord_series.astype(int)).sort_values("__order").reset_index(drop=True)
    if len(res) != len(df):
        fatal(f"NetSolP returned {len(res)} rows for {len(df)} input")
    if not (res["__order"].values == np.arange(len(df))).all():
        fatal("NetSolP sid sequence has gaps after sort")
    print(f"  done in {time.time()-t0:.1f}s")
    return res[["predicted_solubility", "predicted_usability"]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--skip-netsolp", action="store_true",
                   help="Skip NetSolP (saves ~60 min on 1000 seqs). netsolp_* columns will be NaN.")
    args = p.parse_args()

    if not args.inp.exists():
        fatal(f"input CSV not found: {args.inp}")
    df = pd.read_csv(args.inp).reset_index(drop=True)
    if "generated_sequence" not in df.columns:
        fatal(f"input CSV missing 'generated_sequence' column; got {list(df.columns)}")
    peptide_mode = bool(df["peptide_mode"].iloc[0]) if "peptide_mode" in df.columns else False
    print(f"[profile_tool] input={args.inp.name}  rows={len(df)}  peptide_mode={peptide_mode}")

    # Step 1: local profile (always)
    prof = step_local_profile(df)

    # Eligibility masks
    # VHH-specific predictors (TEMPRO, Sapiens) run on rows that look like a VHH.
    # Strict NBv1 rejects all competitors — we run on loose gate.
    if "is_valid_vhh_loose" not in df.columns:
        fatal("input CSV missing is_valid_vhh_loose column — rebuild with latest adapter")
    vhh_mask = df["is_valid_vhh_loose"].astype(bool) & ~pd.Series([peptide_mode]*len(df))

    # Step 2: Sapiens (VHH-only)
    if peptide_mode:
        print("\n=== STEP 2: Sapiens humanness ===\n  SKIPPED: peptide_mode=True (Sapiens trained on VH chains)")
        sap = pd.Series(np.full(len(df), np.nan), name="sapiens_humanness", index=df.index)
    else:
        sap = step_sapiens(df, vhh_mask)

    # Step 3: TEMPRO (VHH-only)
    if peptide_mode:
        print("\n=== STEP 3: TEMPRO ===\n  SKIPPED: peptide_mode=True (TEMPRO trained on VHHs)")
        tm = pd.Series(np.full(len(df), np.nan), name="tempro_tm", index=df.index)
    else:
        tm = step_tempro(df, vhh_mask)

    # Step 4: NetSolP (always, unless --skip-netsolp)
    if args.skip_netsolp:
        print("\n=== STEP 4: NetSolP ===\n  SKIPPED: --skip-netsolp")
        nsp = pd.DataFrame({
            "predicted_solubility": np.full(len(df), np.nan),
            "predicted_usability": np.full(len(df), np.nan),
        }, index=df.index)
    else:
        work_dir = args.out.parent / "netsolp_work"
        nsp = step_netsolp(df, work_dir=work_dir)
        nsp = nsp.set_index(df.index)

    # Assemble output
    out = pd.concat([df, prof], axis=1)
    out["sapiens_humanness"] = sap.values
    out["tempro_tm"] = tm.values
    out["netsolp_solubility"] = nsp["predicted_solubility"].values
    out["netsolp_usability"] = nsp["predicted_usability"].values
    out["netsolp_model_type"] = NETSOLP_MODEL_TYPE if not args.skip_netsolp else "SKIPPED"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\n[profile_tool] wrote {args.out} ({len(out)} rows, {len(out.columns)} cols)")

    # Loud summary per-gate
    def mean_str(s):
        v = pd.Series(s).dropna()
        return f"n={len(v):4d} mean={v.mean():8.4f}" if len(v) else f"n=   0  ALL NaN"

    print("\n=== PROFILE SUMMARY ===")
    for gate_name, gate_mask in (("strict is_valid_126",    out["is_valid_126"].astype(bool)),
                                 ("loose is_valid_vhh",     out["is_valid_vhh_loose"].astype(bool)),
                                 ("all rows",               pd.Series([True]*len(out)))):
        sub = out[gate_mask]
        print(f"\n  [{gate_name}]  n={len(sub)}")
        for col in ("tempro_tm", "instability_index", "sapiens_humanness",
                    "netsolp_solubility", "gravy"):
            if col in sub.columns:
                print(f"    {col:22s} {mean_str(sub[col])}")


if __name__ == "__main__":
    main()
