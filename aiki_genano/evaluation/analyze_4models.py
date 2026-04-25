"""
4-Model Comparison: Generate sequences and evaluate properties.

Models:
  1. SFT base
  2. DPO base (SFT + DPO adapter)
  3. SFT → GDPO (gated)
  4. DPO → GDPO (gated, 6 rewards)

For each model, generates sequences for N test peptides, then evaluates
all 6 NBv1 reward functions on the generated sequences.
"""

import json
import os
import sys
import csv
import torch
import numpy as np
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from aiki_genano.training.data_utils import add_newline_every_60_characters
from aiki_genano.rewards.nbv1_properties import (
    clean_sequence,
    _is_valid_nbv1,
    fr2_aggregation_reward,
    hydrophobic_patch_reward,
    liability_reward,
    expression_reward,
    vhh_hallmark_reward,
    scaffold_integrity_reward,
)

REWARD_NAMES = [
    "fr2_aggregation",
    "hydrophobic_patch",
    "liability",
    "expression",
    "vhh_hallmark",
    "scaffold_integrity",
]
REWARD_FUNCS = [
    fr2_aggregation_reward,
    hydrophobic_patch_reward,
    liability_reward,
    expression_reward,
    vhh_hallmark_reward,
    scaffold_integrity_reward,
]
REWARD_WEIGHTS = [0.15, 0.20, 0.25, 0.15, 0.15, 0.10]

MODELS = {
    "SFT_base": {
        "path": "/app/checkpoints/SFT",
        "type": "merged",
    },
    "DPO_base": {
        "path": "/app/checkpoints/DPO/checkpoint-6000",
        "type": "adapter",
        "base_model": "/app/checkpoints/SFT",
    },
    "SFT_GDPO": {
        "path": "/app/output/results/final_sft_gdpo_gated/checkpoint-4000",
        "type": "adapter",
        "base_model": "/app/checkpoints/SFT",
    },
    "DPO_GDPO": {
        "path": "/app/output/results/final_dpo_gdpo_gated_6rewrads/checkpoint-1500",
        "type": "adapter",
        "base_model": "/app/checkpoints/SFT",
    },
}

TEST_CSV = "/app/data/testing.csv"
NUM_PEPTIDES = 20
NUM_GENERATIONS_PER_PEPTIDE = 5
BATCH_SIZE = 8
TEMPERATURE = 0.9
TOP_P = 0.90
MAX_NEW_TOKENS = 60
OUTPUT_DIR = "/app/output/results/model_comparison_4way"


def load_model(model_cfg, device="cuda"):
    path = model_cfg["path"]
    print(f"\n  Loading from: {path}")

    if model_cfg["type"] == "merged":
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16, device_map=device
        )
        tokenizer = AutoTokenizer.from_pretrained(path)
    else:
        base_path = model_cfg["base_model"]
        tokenizer = AutoTokenizer.from_pretrained(path)
        print(f"  Base model: {base_path}")
        print(f"  Tokenizer vocab: {len(tokenizer)}")

        base_model = AutoModelForCausalLM.from_pretrained(
            base_path, torch_dtype=torch.float16, device_map=device
        )
        base_model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base_model, path)

    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def format_prompt(peptide):
    formatted = add_newline_every_60_characters(peptide)
    return f"<|im_start|>user\n{formatted}<|im_end|>\n<|im_start|>assistant\n"


def generate_sequences(model, tokenizer, peptides, device="cuda"):
    all_seqs = []
    for i in range(0, len(peptides), BATCH_SIZE):
        batch = peptides[i : i + BATCH_SIZE]
        prompts = [format_prompt(p) for p in batch]
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )

        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            gen_tokens = output[input_len:]
            decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            all_seqs.append(decoded.replace("\n", "").strip())

    return all_seqs


def evaluate_sequences(raw_sequences):
    cleaned = [clean_sequence(s) for s in raw_sequences]
    n = len(cleaned)

    validity = [_is_valid_nbv1(s) for s in cleaned]
    lengths = [len(s) for s in cleaned]

    reward_scores = {}
    for name, func in zip(REWARD_NAMES, REWARD_FUNCS):
        scores = func(raw_sequences)
        reward_scores[name] = scores

    composite = []
    for i in range(n):
        total = sum(
            REWARD_WEIGHTS[j] * reward_scores[REWARD_NAMES[j]][i]
            for j in range(len(REWARD_NAMES))
        )
        composite.append(total)

    return {
        "cleaned": cleaned,
        "lengths": lengths,
        "validity": validity,
        "reward_scores": reward_scores,
        "composite": composite,
    }


def print_summary(model_name, results):
    n = len(results["lengths"])
    valid_count = sum(results["validity"])
    valid_pct = valid_count / n * 100

    print(f"\n{'='*70}")
    print(f"  {model_name}  ({n} sequences)")
    print(f"{'='*70}")
    print(f"  Validity:  {valid_count}/{n} ({valid_pct:.1f}%)")
    print(f"  AA length: mean={np.mean(results['lengths']):.1f}, "
          f"std={np.std(results['lengths']):.1f}, "
          f"min={min(results['lengths'])}, max={max(results['lengths'])}")
    print(f"  Exactly 126 AA: {sum(1 for l in results['lengths'] if l == 126)}/{n} "
          f"({sum(1 for l in results['lengths'] if l == 126)/n*100:.1f}%)")

    print(f"\n  {'Reward':<25} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*57}")
    for name in REWARD_NAMES:
        scores = results["reward_scores"][name]
        print(f"  {name:<25} {np.mean(scores):8.4f} {np.std(scores):8.4f} "
              f"{min(scores):8.4f} {max(scores):8.4f}")

    print(f"  {'-'*57}")
    print(f"  {'COMPOSITE (weighted)':<25} {np.mean(results['composite']):8.4f} "
          f"{np.std(results['composite']):8.4f} "
          f"{min(results['composite']):8.4f} {max(results['composite']):8.4f}")

    # Valid-only stats
    if valid_count > 0:
        valid_composites = [results["composite"][i] for i in range(n) if results["validity"][i]]
        print(f"\n  Valid-only composite:   mean={np.mean(valid_composites):.4f}, "
              f"std={np.std(valid_composites):.4f}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load test peptides
    print(f"Loading test peptides from: {TEST_CSV}")
    peptides = []
    with open(TEST_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            peptides.append(row["peptide"])
    unique_peptides = list(dict.fromkeys(peptides))[:NUM_PEPTIDES]
    print(f"Using {len(unique_peptides)} unique peptides, {NUM_GENERATIONS_PER_PEPTIDE} gens each")

    expanded_peptides = unique_peptides * NUM_GENERATIONS_PER_PEPTIDE
    print(f"Total sequences to generate per model: {len(expanded_peptides)}")

    all_results = {}

    for model_name, model_cfg in MODELS.items():
        print(f"\n{'#'*70}")
        print(f"# MODEL: {model_name}")
        print(f"{'#'*70}")

        try:
            model, tokenizer = load_model(model_cfg)

            print(f"  Generating {len(expanded_peptides)} sequences...")
            raw_seqs = generate_sequences(model, tokenizer, expanded_peptides)

            print(f"  Evaluating properties...")
            results = evaluate_sequences(raw_seqs)
            results["raw_sequences"] = raw_seqs
            results["peptides"] = expanded_peptides

            all_results[model_name] = results
            print_summary(model_name, results)

            # Save per-model CSV
            csv_path = os.path.join(OUTPUT_DIR, f"{model_name}_generations.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                header = ["peptide", "raw_sequence", "cleaned_sequence", "aa_length",
                          "is_valid"] + REWARD_NAMES + ["composite"]
                writer.writerow(header)
                for i in range(len(raw_seqs)):
                    row = [
                        expanded_peptides[i],
                        raw_seqs[i],
                        results["cleaned"][i],
                        results["lengths"][i],
                        results["validity"][i],
                    ]
                    for name in REWARD_NAMES:
                        row.append(f"{results['reward_scores'][name][i]:.4f}")
                    row.append(f"{results['composite'][i]:.4f}")
                    writer.writerow(row)
            print(f"  Saved to: {csv_path}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if "model" in dir():
                del model
            torch.cuda.empty_cache()

    # Final comparison table
    print(f"\n\n{'#'*70}")
    print(f"# FINAL COMPARISON")
    print(f"{'#'*70}")
    
    header = f"{'Metric':<28}"
    for name in all_results:
        header += f" {name:>12}"
    print(header)
    print("-" * (28 + 13 * len(all_results)))

    # Validity
    row = f"{'Valid %':<28}"
    for name in all_results:
        r = all_results[name]
        pct = sum(r["validity"]) / len(r["validity"]) * 100
        row += f" {pct:>11.1f}%"
    print(row)

    # Exactly 126 AA
    row = f"{'Exactly 126 AA %':<28}"
    for name in all_results:
        r = all_results[name]
        pct = sum(1 for l in r["lengths"] if l == 126) / len(r["lengths"]) * 100
        row += f" {pct:>11.1f}%"
    print(row)

    # Mean AA length
    row = f"{'Mean AA length':<28}"
    for name in all_results:
        row += f" {np.mean(all_results[name]['lengths']):>12.1f}"
    print(row)

    # Each reward
    for rname in REWARD_NAMES:
        row = f"{rname:<28}"
        for name in all_results:
            row += f" {np.mean(all_results[name]['reward_scores'][rname]):>12.4f}"
        print(row)

    # Composite
    row = f"{'COMPOSITE':<28}"
    for name in all_results:
        row += f" {np.mean(all_results[name]['composite']):>12.4f}"
    print(row)

    # Valid-only composite
    row = f"{'COMPOSITE (valid only)':<28}"
    for name in all_results:
        r = all_results[name]
        valid_c = [r["composite"][i] for i in range(len(r["composite"])) if r["validity"][i]]
        if valid_c:
            row += f" {np.mean(valid_c):>12.4f}"
        else:
            row += f" {'N/A':>12}"
    print(row)

    print(f"\nResults saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
