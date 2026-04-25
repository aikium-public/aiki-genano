"""
Predictor self-consistency check (PROFILING_HANDOFF.md §7).

Runs TEMPRO, NetSolP (ESM1b), Sapiens, motif+biophysical on a 100-row sample
of cached SFT seed42 sequences and compares to the cached columns.

Run from repo root inside the `tempro` conda env:
    python benchmarks/run/selfcheck.py
"""
from __future__ import annotations

import json
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

SAMPLE_CSV = REPO / "benchmarks/run/selfcheck_sft_sample100.csv"
OUT_DIR = REPO / "benchmarks/run/selfcheck_out"
OUT_DIR.mkdir(exist_ok=True)

TEMPRO_MODEL = Path("/models/tempro/ESM_650M.keras")
NETSOLP_DIR = Path("/opt/netsolp")

DEVICE = "cuda"


def step_local_profile(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== STEP 1: motifs + biophysical (local) ===")
    t0 = time.time()
    prof = df["generated_sequence"].astype(str).apply(compute_sequence_profile).apply(pd.Series)
    print(f"  done in {time.time()-t0:.1f}s; {len(prof.columns)} columns")
    return prof


def step_sapiens(df: pd.DataFrame) -> pd.Series:
    print("\n=== STEP 2: Sapiens humanness ===")
    import sapiens
    t0 = time.time()
    out = []
    for i, seq in enumerate(df["generated_sequence"].astype(str)):
        prob_df = sapiens.predict_scores(seq, "H")
        probs = [prob_df.iloc[j][seq[j]] for j in range(len(seq))]
        out.append(float(np.mean(probs)))
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(df)}")
    print(f"  done in {time.time()-t0:.1f}s")
    return pd.Series(out, name="sapiens_humanness")


def step_tempro(df: pd.DataFrame) -> pd.Series:
    print("\n=== STEP 3: TEMPRO Tm ===")
    import torch, esm, h5py, json as _j
    import tensorflow as tf
    t0 = time.time()
    # ESM-2 650M embeddings (mean-pooled, layer 33)
    print("  loading ESM-2 650M …")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    bc = alphabet.get_batch_converter()
    model.eval().to(DEVICE)

    seqs = df["generated_sequence"].astype(str).tolist()
    data = [(f"s{i}", s) for i, s in enumerate(seqs)]
    embeds = []
    BS = 8
    for start in range(0, len(data), BS):
        chunk = data[start:start+BS]
        _, _, toks = bc(chunk)
        toks = toks.to(DEVICE)
        with torch.no_grad():
            r = model(toks, repr_layers=[33], return_contacts=False)
        rep = r["representations"][33]
        lens = (toks != alphabet.padding_idx).sum(1)
        for i, L in enumerate(lens):
            emb = rep[i, 1:L-1].mean(0).cpu()
            embeds.append(emb)
        print(f"    embed {min(start+BS,len(data))}/{len(data)}")
    emb_np = torch.stack(embeds).numpy()
    del model
    torch.cuda.empty_cache()

    # Load TEMPRO Keras (HDF5) by reconstructing from config + weights
    print(f"  loading TEMPRO {TEMPRO_MODEL.name}")
    with h5py.File(str(TEMPRO_MODEL), "r") as f:
        config = _j.loads(f.attrs["model_config"])
    keras_model = tf.keras.models.model_from_json(_j.dumps(config))
    keras_model.load_weights(str(TEMPRO_MODEL))
    pred = keras_model.predict(emb_np, batch_size=32, verbose=0).flatten()
    print(f"  done in {time.time()-t0:.1f}s")
    return pd.Series(pred, name="tempro_tm")


def step_netsolp(df: pd.DataFrame, model_type: str = "ESM1b") -> pd.DataFrame:
    print(f"\n=== STEP 4: NetSolP ({model_type}) ===")
    t0 = time.time()
    fasta = OUT_DIR / "selfcheck.fasta"
    with open(fasta, "w") as fh:
        for i, s in enumerate(df["generated_sequence"].astype(str)):
            fh.write(f">s{i}\n{s}\n")
    out_csv = OUT_DIR / f"selfcheck_netsolp_{model_type}.csv"
    cmd = [
        sys.executable, str(NETSOLP_DIR / "predict.py"),
        "--FASTA_PATH", str(fasta),
        "--OUTPUT_PATH", str(out_csv),
        "--MODELS_PATH", str(NETSOLP_DIR / "models"),
        "--MODEL_TYPE", model_type,
        "--PREDICTION_TYPE", "SU",
    ]
    print("  cmd:", " ".join(cmd[-8:]))
    subprocess.run(cmd, check=True, cwd=str(NETSOLP_DIR))
    res = pd.read_csv(out_csv)
    print(f"  done in {time.time()-t0:.1f}s; cols={list(res.columns)}")
    return res


def diff_summary(name: str, fresh: pd.Series, cached: pd.Series, tol: float):
    fresh = pd.Series(fresh).reset_index(drop=True)
    cached = pd.Series(cached).reset_index(drop=True)
    d = fresh - cached
    abs_d = d.abs()
    n_within = (abs_d <= tol).sum()
    print(f"  {name:25s} mean(fresh)={fresh.mean():8.4f}  mean(cached)={cached.mean():8.4f}  "
          f"max|Δ|={abs_d.max():7.4f}  mean|Δ|={abs_d.mean():7.4f}  within±{tol}: {n_within}/{len(fresh)}")
    return {"name": name, "n": len(fresh), "max_abs": float(abs_d.max()),
            "mean_abs": float(abs_d.mean()), "within_tol": int(n_within),
            "fresh_mean": float(fresh.mean()), "cached_mean": float(cached.mean()), "tol": tol}


def main():
    df = pd.read_csv(SAMPLE_CSV).reset_index(drop=True)
    print(f"Loaded {len(df)} sample sequences from {SAMPLE_CSV.name}")
    print(f"Cached netsolp_model_type: {df.netsolp_model_type.unique().tolist() if 'netsolp_model_type' in df.columns else 'N/A'}")

    results = {}

    # Step 1: motifs + biophysical
    prof = step_local_profile(df)
    print("\n--- diffs (local profile vs cached) ---")
    diffs = []
    motif_cols = [c for c in df.columns if c.startswith("motif_")]
    for c in motif_cols:
        if c in prof.columns:
            diffs.append(diff_summary(c, prof[c], df[c], tol=0.0))
    for c in ("instability_index", "gravy"):
        if c in prof.columns and c in df.columns:
            diffs.append(diff_summary(c, prof[c], df[c], tol=0.01))
    results["local_profile"] = diffs

    # Step 2: Sapiens
    sap = step_sapiens(df)
    print("\n--- Sapiens diff ---")
    results["sapiens"] = [diff_summary("sapiens_humanness", sap, df["sapiens_humanness"], tol=0.01)]

    # Step 3: TEMPRO
    tm = step_tempro(df)
    print("\n--- TEMPRO diff ---")
    results["tempro"] = [diff_summary("tempro_tm", tm, df["tempro_tm"], tol=1.0)]

    # Step 4: NetSolP ESM1b
    nsp = step_netsolp(df, model_type="ESM1b")
    print("\n--- NetSolP diff ---")
    # NetSolP outputs columns: sid + various predicted_*; we need solubility & usability
    # Find the right column names; column may be 'predicted_solubility' or similar
    sol_col = next((c for c in nsp.columns if "solub" in c.lower()), None)
    usa_col = next((c for c in nsp.columns if "usab" in c.lower()), None)
    if sol_col is None:
        print(f"  WARNING: no solubility column in NetSolP output; cols={list(nsp.columns)}")
    # NetSolP may reorder rows (sid order) — align by sid
    nsp = nsp.sort_values(by=nsp.columns[0]).reset_index(drop=True)
    # our sids were s0..s99 — they may not sort lex; coerce
    nsp["_idx"] = nsp[nsp.columns[0]].str.extract(r"s(\d+)").astype(int)
    nsp = nsp.sort_values("_idx").reset_index(drop=True)
    diffs = []
    if sol_col:
        diffs.append(diff_summary("netsolp_solubility", nsp[sol_col], df["netsolp_solubility"], tol=0.02))
    if usa_col and "netsolp_usability" in df.columns:
        diffs.append(diff_summary("netsolp_usability", nsp[usa_col], df["netsolp_usability"], tol=0.02))
    results["netsolp"] = diffs

    # Save full report
    out_json = OUT_DIR / "selfcheck_report.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_json}")

    # Verdict
    print("\n=== VERDICT ===")
    for grp, ds in results.items():
        any_fail = any(d["within_tol"] < d["n"] for d in ds)
        if grp == "local_profile":
            # be strict on motifs (tol=0); biophysical (tol=0.01) — fail if any out-of-tol
            print(f"  {grp:15s} {'FAIL' if any_fail else 'OK'}")
        else:
            print(f"  {grp:15s} {'FAIL' if any_fail else 'OK'}")


if __name__ == "__main__":
    main()
