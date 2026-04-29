"""
Generate nanobody sequences with controlled randomness for statistical rigor.

Runs a full grid of (seed × temperature) for each model, saving one CSV per
combination plus a merged file.  Downstream analysis can report mean ± std
across seeds/temperatures and compute p-values.

Folder layout
─────────────
OUTPUT_DIR/
    {model}/
        {model}_seed{s}_temp{t}.csv   ← one per (seed, temp) combination
        {model}_all.csv               ← merged across all runs
    summary.csv                       ← one-row-per-model overview

Usage (inside Docker):
    python -m aiki_genano.evaluation.inference
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import List, Optional

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from aiki_genano.training.data_utils import add_newline_every_60_characters

# =========================================================================
# CONFIGURATION
# =========================================================================

SFT_PATH = (
    "/app/output/results/output_trl/"
    "NanoBody-design_response_only_A100/100k_len126_sft/"
    "NanoBody-design-sft-response-only-100k-len126-r64"
)

DPO_BASE = (
    "/app/output/results/output_trl/"
    "NanoBody-design_response_only_A100/"
    "yotta10k_v1_dpo_developability/dpo"
)

GDPO_PATH = "/app/output/results/dpo_final_gated/final_checkpoint"

# SFT→GDPO ablation checkpoint (copy from remote machine into results/sft_final_gated/)
GDPO_SFT_PATH = "/app/output/results/sft_final_gated/final_checkpoint"

MODELS = {
    "SFT": {
        "path": SFT_PATH,
        "base_model": None,
        "adapter_chain": None,
    },
    "DPO": {
        "path": f"{DPO_BASE}/checkpoint-6000",
        "base_model": SFT_PATH,
        "adapter_chain": None,
    },
    "GDPO": {
        "path": GDPO_PATH,
        "base_model": SFT_PATH,
        "adapter_chain": [f"{DPO_BASE}/checkpoint-6000"],
    },
    # Ablation: GDPO trained directly from SFT (no DPO intermediate)
    "GDPO_SFT": {
        "path": GDPO_SFT_PATH,
        "base_model": SFT_PATH,
        "adapter_chain": None,  # SFT is already a merged model, no chain needed
    },
}

SAMPLING_BASE = {
    "do_sample": True,
    "top_p": 0.9,
    "max_new_tokens": 256,
}

SEEDS = [42, 123, 456]
TEMPERATURES = [0.7, 0.9, 1.2]

TEST_CSV = "/app/data/testing.csv"
OUTPUT_DIR = "/app/output/statistical"

N_SAMPLES = None  # use all epitopes in the test set
N_GENS_PER_EPITOPE = 100  # 50 for fast iteration; set to 100 for publication-grade runs
BATCH_SIZE = 64
DEVICE = "cuda"

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# =========================================================================
# Model loading
# =========================================================================

def load_model(
    checkpoint_path: str,
    base_model: Optional[str] = None,
    adapter_chain: Optional[List[str]] = None,
    device: str = "cuda",
):
    """Load a model checkpoint (merged, single LoRA, or chained adapters)."""
    print(f"\nLoading model from: {checkpoint_path}")

    is_merged = os.path.exists(os.path.join(checkpoint_path, "model.safetensors"))
    is_lora = os.path.exists(os.path.join(checkpoint_path, "adapter_model.safetensors"))

    if is_merged:
        print("  Type: Merged model")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path, torch_dtype=torch.float16, device_map=device,
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    elif is_lora:
        resolved_base = base_model or "nferruz/ProtGPT2"
        adapter_cfg_path = os.path.join(checkpoint_path, "adapter_config.json")
        if os.path.exists(adapter_cfg_path):
            try:
                with open(adapter_cfg_path, "r") as f:
                    cfg = json.load(f)
                if not base_model:
                    resolved_base = cfg.get("base_model_name_or_path", resolved_base)
            except Exception:
                pass

        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        print(f"  Base: {resolved_base}")
        model = AutoModelForCausalLM.from_pretrained(
            resolved_base, torch_dtype=torch.float16, device_map=device,
        )
        model.resize_token_embeddings(len(tokenizer))

        if adapter_chain:
            for i, adapter in enumerate(adapter_chain):
                print(f"  Merging intermediate adapter [{i}]: {adapter}")
                model = PeftModel.from_pretrained(model, adapter)
                model = model.merge_and_unload()

        print(f"  Applying final adapter: {checkpoint_path}")
        model = PeftModel.from_pretrained(model, checkpoint_path)
    else:
        raise ValueError(f"No model files found in {checkpoint_path}")

    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


# =========================================================================
# Generation
# =========================================================================

def _build_prompt(tokenizer, peptide_seq: str) -> str:
    formatted = add_newline_every_60_characters(peptide_seq)
    messages = [{"role": "user", "content": formatted}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def generate_batch(model, tokenizer, peptides, temperature: float, device="cuda"):
    prompts = [_build_prompt(tokenizer, p) for p in peptides]
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=512,
    ).to(device)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    sampling_params = {**SAMPLING_BASE, "temperature": temperature}

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=im_end_id,
            **sampling_params,
        )

    generated = []
    for i, out in enumerate(outputs):
        input_len = inputs["input_ids"][i].shape[0]
        decoded = tokenizer.decode(out[input_len:], skip_special_tokens=True)
        generated.append(decoded.replace("\n", "").strip())
    return generated


# =========================================================================
# Validity checks
# =========================================================================

def basic_checks(seq: str) -> dict:
    n = len(seq)
    return {
        "gen_length": n,
        "is_valid_126": n == 126 and seq.endswith("GGGGS") and seq.count("C") == 2,
        "non_standard_aa": len(set(seq) - STANDARD_AA),
    }


# =========================================================================
# Main
# =========================================================================

def _generate_one_run(model, tokenizer, peptides, seed, temperature, model_name):
    """Generate sequences for a single (seed, temperature) combination."""
    set_seed(seed)
    n_total = len(peptides)

    generations = []
    for i in tqdm(
        range(0, n_total, BATCH_SIZE),
        desc=f"  seed={seed} T={temperature}",
    ):
        batch = peptides[i : i + BATCH_SIZE]
        gens = generate_batch(model, tokenizer, batch, temperature, DEVICE)
        generations.extend(gens)

    records = []
    for pep, gen in zip(peptides, generations):
        rec = {"peptide": pep, "generated_sequence": gen}
        rec.update(basic_checks(gen))
        records.append(rec)

    df = pd.DataFrame(records)
    df["model"] = model_name
    df["seed"] = seed
    df["temperature"] = temperature
    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Generate nanobody sequences.")
    parser.add_argument(
        "--model", nargs="+", default=None,
        choices=list(MODELS.keys()),
        help="Run only these model(s). Default: all models.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Seeds to use. Default: all seeds in SEEDS list.",
    )
    parser.add_argument(
        "--temps", nargs="+", type=float, default=None,
        help="Temperatures to use. Default: all temps in TEMPERATURES list.",
    )
    return parser.parse_args()


def run():
    args = parse_args()

    seeds = args.seeds if args.seeds else SEEDS
    temps = args.temps if args.temps else TEMPERATURES
    model_filter = args.model  # None = run all

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    test_df = pd.read_csv(TEST_CSV)
    if "peptide" not in test_df.columns and "epitope" in test_df.columns:
        test_df = test_df.rename(columns={"epitope": "peptide"})
    if "peptide" not in test_df.columns:
        raise ValueError("Test CSV needs a 'peptide' column")

    test_df = test_df.drop_duplicates(subset="peptide").reset_index(drop=True)

    if N_SAMPLES and N_SAMPLES < len(test_df):
        test_df = test_df.sample(n=N_SAMPLES, random_state=42).reset_index(drop=True)

    if N_GENS_PER_EPITOPE > 1:
        test_df = test_df.loc[
            test_df.index.repeat(N_GENS_PER_EPITOPE)
        ].reset_index(drop=True)

    peptides = test_df["peptide"].tolist()
    n_total = len(peptides)
    n_epitopes = test_df["peptide"].nunique()
    n_runs = len(seeds) * len(temps)

    print(
        f"Test set: {n_total} requests "
        f"({n_epitopes} unique peptides x {N_GENS_PER_EPITOPE} gens)"
    )
    print(f"Seeds:        {seeds}")
    print(f"Temperatures: {temps}")
    print(f"Runs/model:   {n_runs} ({len(seeds)} seeds x {len(temps)} temps)")
    print(f"Seqs/model:   {n_total * n_runs}")

    active_models = {
        k: v for k, v in MODELS.items()
        if model_filter is None or k in model_filter
    }
    print(f"Models:       {list(active_models.keys())}")

    all_summaries = []

    for model_name, model_cfg in active_models.items():
        print("\n" + "=" * 70)
        print(f"MODEL: {model_name}")
        print("=" * 70)

        ckpt = model_cfg["path"]
        if not os.path.exists(ckpt):
            print(f"  SKIP -- checkpoint not found: {ckpt}")
            continue

        model, tokenizer = load_model(
            ckpt, model_cfg.get("base_model"), model_cfg.get("adapter_chain"), DEVICE,
        )

        model_dir = os.path.join(OUTPUT_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)

        run_dfs = []
        t0 = time.time()

        for seed in seeds:
            for temp in temps:
                fname = f"{model_name}_seed{seed}_temp{temp}.csv"
                fpath = os.path.join(model_dir, fname)
                if os.path.exists(fpath):
                    print(f"    seed={seed} T={temp}: SKIP (already exists)")
                    run_dfs.append(pd.read_csv(fpath))
                    continue

                df = _generate_one_run(
                    model, tokenizer, peptides, seed, temp, model_name,
                )
                run_dfs.append(df)
                df.to_csv(fpath, index=False)

                n_valid = int(df["is_valid_126"].sum())
                print(
                    f"    seed={seed} T={temp}: "
                    f"valid={n_valid}/{n_total} ({100 * n_valid / n_total:.1f}%)"
                )

        merged = pd.concat(run_dfs, ignore_index=True)
        merged.to_csv(os.path.join(model_dir, f"{model_name}_all.csv"), index=False)

        elapsed = time.time() - t0
        n_total_all = len(merged)
        n_valid_all = int(merged["is_valid_126"].sum())
        n_unique_all = int(merged["generated_sequence"].nunique())

        per_run_valid = [
            int(d["is_valid_126"].sum()) / n_total * 100 for d in run_dfs
        ]
        summary = {
            "Model": model_name,
            "N_seeds": len(seeds),
            "N_temps": len(temps),
            "N_runs": n_runs,
            "N_per_run": n_total,
            "N_total": n_total_all,
            "Valid_total": n_valid_all,
            "Valid_pct_mean": f"{sum(per_run_valid) / len(per_run_valid):.1f}",
            "Valid_pct_std": f"{pd.Series(per_run_valid).std():.1f}",
            "Unique": n_unique_all,
            "Time_s": f"{elapsed:.0f}",
        }
        all_summaries.append(summary)

        print(
            f"  -> Combined: {n_valid_all}/{n_total_all} valid, "
            f"{n_unique_all} unique [{elapsed:.0f}s]"
        )

        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    summary_df = pd.DataFrame(all_summaries)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)
    print(f"\nAll results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
