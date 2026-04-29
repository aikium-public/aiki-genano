"""
Run TEMPRO melting temperature prediction on generated nanobody CSVs.

Pipeline: sequences → ESM-2 650M embeddings → TEMPRO Keras model → Tm prediction

Usage (inside Docker):
    pip install tensorflow fair-esm
    python -m aiki_genano.evaluation.run_tempro
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

CSV_DIR = Path("/app/output/csv")
TEMPRO_MODEL = Path("/models/tempro/ESM_650M.keras")
MODELS = ["SFT_20k", "DPO_6k", "GDPO_dpo_final_gated"]
BATCH_SIZE = 25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_esm_embeddings(sequences: list[str]) -> np.ndarray:
    """Generate ESM-2 650M mean-pooled embeddings for a list of sequences."""
    import esm

    print("  Loading ESM-2 650M...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    model = model.to(DEVICE)

    data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
    all_embeddings = []

    for start in range(0, len(data), BATCH_SIZE):
        chunk = data[start : start + BATCH_SIZE]
        batch_labels, batch_strs, batch_tokens = batch_converter(chunk)
        batch_tokens = batch_tokens.to(DEVICE)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]

        for i, tokens_len in enumerate(batch_lens):
            emb = token_representations[i, 1 : tokens_len - 1].mean(0)
            all_embeddings.append(emb.cpu())

        if (start + BATCH_SIZE) % 100 == 0 or start + BATCH_SIZE >= len(data):
            print(f"    ESM embeddings: {min(start + BATCH_SIZE, len(data))}/{len(data)}")

    del model
    torch.cuda.empty_cache()

    return torch.stack(all_embeddings).numpy()


def predict_tm(embeddings: np.ndarray) -> np.ndarray:
    """Predict melting temperature using TEMPRO Keras model.

    The .keras file is HDF5 (Keras 2 format). We reconstruct the
    architecture from the stored config and load weights from h5py
    since Keras 3 can't read old HDF5 weight files directly.
    """
    import h5py
    import json
    import tensorflow as tf

    print(f"  Loading TEMPRO model: {TEMPRO_MODEL.name}")

    with h5py.File(str(TEMPRO_MODEL), "r") as f:
        config = json.loads(f.attrs["model_config"])
        layers_cfg = config["config"]["layers"]

        model = tf.keras.Sequential()
        dense_layers = []
        for lc in layers_cfg:
            cls = lc["class_name"]
            cfg = lc["config"]
            if cls == "InputLayer":
                model.add(tf.keras.layers.InputLayer(shape=tuple(cfg["batch_input_shape"][1:])))
            elif cls == "Dense":
                layer = tf.keras.layers.Dense(
                    cfg["units"], activation=cfg["activation"], name=cfg["name"])
                model.add(layer)
                dense_layers.append((cfg["name"], layer))
            elif cls == "Dropout":
                model.add(tf.keras.layers.Dropout(cfg["rate"], name=cfg["name"]))

        # Build the model so weights are allocated
        model.build(input_shape=(None, 1280))

        # Load weights from HDF5
        mw = f["model_weights"]
        for name, layer in dense_layers:
            kernel = np.array(mw[name][name]["kernel:0"])
            bias = np.array(mw[name][name]["bias:0"])
            layer.set_weights([kernel, bias])

    preds = model.predict(embeddings, verbose=0)
    return preds.flatten()


def main():
    print("=" * 60)
    print("TEMPRO MELTING TEMPERATURE PREDICTION")
    print("=" * 60)

    for model_name in MODELS:
        csv_path = CSV_DIR / f"{model_name}.csv"
        if not csv_path.exists():
            print(f"SKIP: {csv_path}")
            continue

        out_path = CSV_DIR / f"{model_name}_tempro.csv"
        print(f"\n--- {model_name} ---")

        df = pd.read_csv(csv_path)
        seqs = df["generated_sequence"].tolist()
        print(f"  {len(seqs)} sequences")

        embeddings = generate_esm_embeddings(seqs)
        print(f"  Embeddings shape: {embeddings.shape}")

        tm_preds = predict_tm(embeddings)
        df["tempro_tm"] = tm_preds

        df.to_csv(out_path, index=False)
        print(f"  Tm: mean={tm_preds.mean():.2f}°C, std={tm_preds.std():.2f}°C, "
              f"min={tm_preds.min():.1f}, max={tm_preds.max():.1f}")
        print(f"  Saved: {out_path.name}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for model_name in MODELS:
        out_path = CSV_DIR / f"{model_name}_tempro.csv"
        if out_path.exists():
            df = pd.read_csv(out_path)
            tm = df["tempro_tm"]
            print(f"  {model_name:30s}  Tm={tm.mean():.2f}°C (std={tm.std():.2f})")


if __name__ == "__main__":
    main()
