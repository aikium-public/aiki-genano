"""
Compute all developability properties for generated nanobody sequences.
# pip install biopython sapiens fair-esm onnxruntime tensorflow h5py

Pipeline per model:
  1. Local profiling  — GDPO rewards, motifs, biophysical, CDR, Aggrescan
  2. NetSolP          — predicted solubility & usability
  3. Sapiens          — humanness score
  4. TEMPRO           — predicted melting temperature (Tm)

Outputs (inside statistical/{model}/):
  • {model}_seed{s}_temp{t}_profiled.csv  — per-run with all local properties
  • {model}_all_profiled.csv              — merged + all properties
  • {model}_summary.csv                   — per-run aggregated means

Usage (inside Docker):
    python -m src.binder_design.protgpt2_dpo.analysis.run_properties --model SFT
    python -m src.binder_design.protgpt2_dpo.analysis.run_properties --model DPO
    python -m src.binder_design.protgpt2_dpo.analysis.run_properties --model GDPO
    python -m src.binder_design.protgpt2_dpo.analysis.run_properties --model all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from aiki_genano.evaluation.profile import compute_sequence_profile

# =========================================================================
# Configuration
# =========================================================================

STAT_DIR = Path(
    "/app/output/statistical"
)
ALL_MODELS = ["SFT", "DPO", "GDPO", "GDPO_SFT"]
SEQ_COL = "generated_sequence"

NETSOLP_DIR = Path("/opt/netsolp")
NETSOLP_PREDICT = NETSOLP_DIR / "predict.py"
NETSOLP_MODELS_DIR = NETSOLP_DIR / "models"
NETSOLP_MODEL_TYPE = "ESM12"

TEMPRO_MODEL_PATH = Path(
    "/models/tempro/ESM_650M.keras"
)
ESM_BATCH_SIZE = 128


# =========================================================================
# 1. Local profiling (fast, CPU-only)
# =========================================================================

def profile_csv(csv_path: Path, props_dir: Path) -> Path:
    """Run compute_sequence_profile on valid sequences only; save into props_dir."""
    out_path = props_dir / (csv_path.stem + "_profiled.csv")

    if out_path.exists():
        existing = pd.read_csv(out_path, nrows=1)
        if "reward_liability" in existing.columns:
            print(f"  [skip] already profiled: {out_path.name}")
            return out_path

    df = pd.read_csv(csv_path)
    if SEQ_COL not in df.columns:
        raise ValueError(f"Column '{SEQ_COL}' not found in {csv_path}")

    n_total = len(df)
    df = df[df["is_valid_126"] == True].reset_index(drop=True)
    n_valid = len(df)
    print(f"  Profiling {n_valid}/{n_total} valid sequences from {csv_path.name} …")

    t0 = time.time()
    prof = (
        df[SEQ_COL]
        .astype(str)
        .apply(compute_sequence_profile)
        .apply(pd.Series)
    )
    result = pd.concat([df, prof], axis=1)
    result.to_csv(out_path, index=False)
    print(f"    → properties/{out_path.name}  ({time.time() - t0:.1f}s)")
    return out_path


# =========================================================================
# 2. Aggregate per-run profiled CSVs into one merged file
# =========================================================================

def aggregate_model(model_name: str, props_dir: Path) -> Path:
    """Merge per-run profiled CSVs from props_dir, save summary."""
    profiled_csvs = sorted(props_dir.glob("*_profiled.csv"))
    profiled_csvs = [p for p in profiled_csvs if "_all_" not in p.name]

    if not profiled_csvs:
        raise FileNotFoundError(
            f"No profiled CSVs in {props_dir}. Run profiling first."
        )

    dfs = [pd.read_csv(p) for p in profiled_csvs]
    merged = pd.concat(dfs, ignore_index=True)

    merged_path = props_dir / f"{model_name}_all_profiled.csv"
    merged.to_csv(merged_path, index=False)
    print(f"  Merged → properties/{merged_path.name} ({len(merged)} rows, valid-only)")

    run_summaries = []
    for (seed, temp), grp in merged.groupby(["seed", "temperature"]):
        row: Dict[str, object] = {
            "model": model_name,
            "seed": seed,
            "temperature": temp,
            "n_valid": len(grp),
            "unique_seqs": int(grp[SEQ_COL].nunique()),
        }
        for col in grp.select_dtypes(include="number").columns:
            if col in ("seed", "temperature"):
                continue
            row[f"{col}_mean"] = grp[col].mean()
        run_summaries.append(row)

    summary_df = pd.DataFrame(run_summaries)
    summary_path = props_dir / f"{model_name}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Summary → properties/{summary_path.name} ({len(summary_df)} runs)")

    return merged_path


# =========================================================================
# 3. NetSolP — solubility & usability (direct call, proper batching)
# =========================================================================

NETSOLP_TOKS_PER_BATCH = 131072  # ~1024 seqs of length 126 per batch


def run_netsolp(csv_path: Path) -> None:
    if not NETSOLP_PREDICT.exists():
        print("  [netsolp] not found — skipping.")
        return

    df = pd.read_csv(csv_path)
    if "netsolp_solubility" in df.columns:
        print("  [netsolp] already computed — skipping.")
        return

    try:
        import onnxruntime
        import pickle as pkl
    except ImportError:
        print("  [netsolp] onnxruntime not installed — skipping.")
        return

    sys.path.insert(0, str(NETSOLP_DIR))
    from data import FastaBatchedDataset, BatchConverter

    print(f"  [netsolp] Scoring {len(df)} valid sequences …")
    t0 = time.time()

    seqs = df[SEQ_COL].astype(str).tolist()
    sids = [f"seq_{i}" for i in range(len(seqs))]
    test_df = pd.DataFrame({"sid": sids, "fasta": [s[:1022] for s in seqs]})

    alphabet_path = str(NETSOLP_MODELS_DIR / "ESM12_alphabet.pkl")
    with open(alphabet_path, "rb") as f:
        alphabet = pkl.load(f)

    embed_dataset = FastaBatchedDataset(test_df)
    embed_batches = embed_dataset.get_batch_indices(
        NETSOLP_TOKS_PER_BATCH, extra_toks_per_seq=1,
    )
    embed_dataloader = torch.utils.data.DataLoader(
        embed_dataset, collate_fn=BatchConverter(alphabet),
        batch_sampler=embed_batches,
    )

    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    onnxruntime.set_default_logger_severity(3)  # suppress shape warnings
    opts = onnxruntime.SessionOptions()
    opts.intra_op_num_threads = min(os.cpu_count() or 4, 8)
    opts.inter_op_num_threads = min(os.cpu_count() or 4, 8)

    for pred_type, col_name in [("Solubility", "netsolp_solubility"),
                                 ("Usability", "netsolp_usability")]:
        print(f"    {pred_type} …")
        preds_per_split = []
        for split_i in range(5):
            model_path = str(
                NETSOLP_MODELS_DIR / f"{pred_type}_{NETSOLP_MODEL_TYPE}_{split_i}_quantized.onnx"
            )
            session = onnxruntime.InferenceSession(model_path, sess_options=opts)
            input_names = session.get_inputs()

            embed_dict = {}
            for toks, lengths, np_mask, labels in embed_dataloader:
                ort_inputs = {
                    input_names[0].name: toks.numpy(),
                    input_names[1].name: lengths.numpy(),
                    input_names[2].name: np_mask.numpy(),
                }
                out = session.run(None, ort_inputs)[0]
                if out.ndim == 0:
                    embed_dict[labels[0]] = float(out)
                else:
                    for j, label in enumerate(labels):
                        embed_dict[label] = float(out[j]) if j < len(out) else float(out[-1])

            pred_series = test_df["sid"].map(embed_dict)
            preds_i = _sigmoid(pred_series.to_numpy().astype(float))
            preds_per_split.append(preds_i)

        avg_pred = sum(preds_per_split) / 5
        df[col_name] = avg_pred

    df.to_csv(csv_path, index=False)
    sol = df["netsolp_solubility"]
    usa = df["netsolp_usability"]
    print(f"  [netsolp] sol={sol.mean():.4f} usa={usa.mean():.4f}  ({time.time() - t0:.0f}s)")


# =========================================================================
# 4. Sapiens — humanness
# =========================================================================

def run_sapiens(merged_csv: Path) -> None:
    try:
        import sapiens
    except ImportError:
        print("  [sapiens] not installed (pip install sapiens) — skipping.")
        return

    df = pd.read_csv(merged_csv)
    if "sapiens_humanness" in df.columns:
        print("  [sapiens] already computed — skipping.")
        return

    seqs = df[SEQ_COL].astype(str).tolist()
    print(f"  [sapiens] Scoring {len(seqs)} valid sequences …")
    t0 = time.time()
    scores: List[float] = []
    for i, seq in enumerate(seqs):
        try:
            prob_df = sapiens.predict_scores(seq, "H")
            probs = [prob_df.iloc[j][seq[j]] for j in range(len(seq))]
            scores.append(float(np.mean(probs)))
        except Exception:
            scores.append(float("nan"))
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"    {i + 1}/{len(seqs)}  ({elapsed:.0f}s)")

    df["sapiens_humanness"] = scores
    df.to_csv(merged_csv, index=False)
    valid = [s for s in scores if not np.isnan(s)]
    print(f"  [sapiens] mean={np.mean(valid):.4f}  ({time.time() - t0:.0f}s)")


# =========================================================================
# 5. TEMPRO — melting temperature (ESM-2 embeddings + Keras model)
# =========================================================================

def run_tempro(merged_csv: Path) -> None:
    if not TEMPRO_MODEL_PATH.exists():
        print("  [tempro] model not found — skipping.")
        return
    try:
        import esm
    except ImportError:
        print("  [tempro] fair-esm not installed — skipping.")
        return
    try:
        import tensorflow as tf
        import h5py
    except ImportError:
        print("  [tempro] tensorflow/h5py not installed — skipping.")
        return

    df = pd.read_csv(merged_csv)
    if "tempro_tm" in df.columns:
        print("  [tempro] already computed — skipping.")
        return

    seqs = df[SEQ_COL].astype(str).tolist()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    print(f"  [tempro] ESM-2 650M embeddings for {len(seqs)} valid seqs …")
    t0 = time.time()
    model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model_esm.eval().to(device)

    data = [(f"seq_{i}", seq) for i, seq in enumerate(seqs)]
    all_embeddings = []

    for start in range(0, len(data), ESM_BATCH_SIZE):
        chunk = data[start : start + ESM_BATCH_SIZE]
        _, _, batch_tokens = batch_converter(chunk)
        batch_tokens = batch_tokens.to(device)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        with torch.no_grad():
            results = model_esm(
                batch_tokens, repr_layers=[33], return_contacts=False,
            )
        reps = results["representations"][33]
        for i, tlen in enumerate(batch_lens):
            emb = reps[i, 1 : tlen - 1].mean(0)
            all_embeddings.append(emb.cpu())
        done = min(start + ESM_BATCH_SIZE, len(data))
        if done % 500 < ESM_BATCH_SIZE or done == len(data):
            print(f"    ESM embeddings: {done}/{len(data)}")

    del model_esm
    torch.cuda.empty_cache()
    embeddings = torch.stack(all_embeddings).numpy()

    print("  [tempro] Loading TEMPRO Keras model …")
    with h5py.File(str(TEMPRO_MODEL_PATH), "r") as f:
        config = json.loads(f.attrs["model_config"])
        layers_cfg = config["config"]["layers"]
        keras_model = tf.keras.Sequential()
        dense_layers = []
        for lc in layers_cfg:
            cls = lc["class_name"]
            cfg = lc["config"]
            if cls == "InputLayer":
                keras_model.add(
                    tf.keras.layers.InputLayer(
                        shape=tuple(cfg["batch_input_shape"][1:])
                    )
                )
            elif cls == "Dense":
                layer = tf.keras.layers.Dense(
                    cfg["units"], activation=cfg["activation"], name=cfg["name"],
                )
                keras_model.add(layer)
                dense_layers.append((cfg["name"], layer))
            elif cls == "Dropout":
                keras_model.add(
                    tf.keras.layers.Dropout(cfg["rate"], name=cfg["name"])
                )
        keras_model.build(input_shape=(None, 1280))
        mw = f["model_weights"]
        for name, layer in dense_layers:
            kernel = np.array(mw[name][name]["kernel:0"])
            bias = np.array(mw[name][name]["bias:0"])
            layer.set_weights([kernel, bias])

    tm_preds = keras_model.predict(embeddings, verbose=0).flatten()
    df["tempro_tm"] = tm_preds
    df.to_csv(merged_csv, index=False)
    print(f"  [tempro] Tm mean={tm_preds.mean():.2f}°C  ({time.time() - t0:.0f}s)")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute all properties for generated nanobody CSVs.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["SFT", "DPO", "GDPO", "GDPO_SFT", "all"],
        help="Which model to process (default: all).",
    )
    args = parser.parse_args()

    models_to_run = ALL_MODELS if args.model == "all" else [args.model]

    for model_name in models_to_run:
        model_dir = STAT_DIR / model_name
        if not model_dir.is_dir():
            print(f"\n[SKIP] {model_dir} does not exist yet.")
            continue

        print(f"\n{'=' * 60}")
        print(f"MODEL: {model_name}")
        print(f"{'=' * 60}")

        props_dir = model_dir / "properties"
        props_dir.mkdir(exist_ok=True)

        # Step 1: Local profiling (per-run CSVs → properties/)
        run_csvs = sorted(
            p for p in model_dir.glob(f"{model_name}_seed*_temp*.csv")
            if "_profiled" not in p.name
        )
        if not run_csvs:
            print(f"  No generation CSVs found in {model_dir}")
            continue

        print(f"  Found {len(run_csvs)} run CSVs → properties/")
        profiled_csvs = []
        for csv_path in run_csvs:
            profiled = profile_csv(csv_path, props_dir)
            profiled_csvs.append(profiled)

        # Step 2: External predictors (per-run, one CSV at a time)
        for i, profiled_csv in enumerate(profiled_csvs, 1):
            print(f"\n  [{i}/{len(profiled_csvs)}] External predictors: {profiled_csv.name}")
            run_tempro(profiled_csv)
            run_sapiens(profiled_csv)
            # run_netsolp(profiled_csv)
            

        # Step 3: Aggregate into merged CSV
        aggregate_model(model_name, props_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
