"""
Simple inference script for GDPO sampling analysis.

Generates N sequences per target peptide with nucleus (top-p) sampling.
Reports AA lengths, BPE token counts, and basic quality metrics.

Usage:
    python -m aiki_genano.evaluation.inference_gdpo \
        --checkpoint /path/to/model \
        --test_csv data/nanobody/nanobody126/testing.csv \
        --num_targets 5 \
        --num_generations 16 \
        --temperature 1.0 \
        --top_p 0.95
"""

import argparse
import json
import math
import os
from typing import Dict, List

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from aiki_genano.training.data_utils import add_newline_every_60_characters
from aiki_genano.rewards.nanobody_scaffold import NBV1, NB_V1_REFERENCE


STANDARD_AA_SET = set("ACDEFGHIKLMNPQRSTVWY")


# ──────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load a merged model or LoRA adapter checkpoint."""
    print(f"\n  Loading model from: {checkpoint_path}")

    is_merged = os.path.exists(os.path.join(checkpoint_path, "model.safetensors"))
    is_lora = os.path.exists(os.path.join(checkpoint_path, "adapter_model.safetensors"))

    if is_merged:
        print("    Type: Merged model")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path, torch_dtype=torch.float16, device_map=device,
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    elif is_lora:
        print("    Type: LoRA adapter")
        base_model_name = "nferruz/ProtGPT2"
        adapter_cfg_path = os.path.join(checkpoint_path, "adapter_config.json")
        if os.path.exists(adapter_cfg_path):
            try:
                with open(adapter_cfg_path, "r") as f:
                    base_model_name = json.load(f).get(
                        "base_model_name_or_path", base_model_name
                    )
            except Exception:
                pass
        print(f"    Base model: {base_model_name}")

        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.float16, device_map=device,
        )
        base_model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
    else:
        raise ValueError(f"No model files found in {checkpoint_path}")

    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"    Vocab size: {len(tokenizer)}, pad_token={tokenizer.pad_token}")
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────────────

def format_prompt(peptide: str) -> str:
    formatted = add_newline_every_60_characters(peptide)
    return f"<|im_start|>user\n{formatted}<|im_end|>\n<|im_start|>assistant\n"


def clean_sequence(text: str) -> str:
    """Extract clean AA sequence from model output."""
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>", 1)[0]
    text = text.replace("\n", "").strip()
    return "".join(c for c in text.upper() if c in STANDARD_AA_SET)


def generate_batch(
    model, tokenizer, peptides: List[str],
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_p: float = 0.95,
    device: str = "cuda",
) -> List[Dict]:
    """Generate completions and measure lengths."""
    prompts = [format_prompt(p) for p in peptides]
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=512,
    ).to(device)

    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_id,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

    results = []
    for i in range(len(peptides)):
        input_len = inputs["input_ids"][i].shape[0]
        completion_tokens = outputs[i][input_len:]
        non_pad = completion_tokens[completion_tokens != tokenizer.pad_token_id]
        token_count = len(non_pad)
        terminated = eos_id in completion_tokens.tolist()
        raw_completion = tokenizer.decode(non_pad, skip_special_tokens=False)
        clean_aa = clean_sequence(raw_completion)

        results.append({
            "peptide": peptides[i],
            "clean_aa": clean_aa,
            "aa_length": len(clean_aa),
            "token_count": token_count,
            "terminated": terminated,
            "is_126aa": len(clean_aa) == 126,
            "ends_GGGGS": clean_aa.endswith("GGGGS"),
        })

    return results


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Simple GDPO inference analysis")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_csv", type=str,
                        default="data/nanobody/nanobody126/testing.csv")
    parser.add_argument("--num_targets", type=int, default=5,
                        help="Number of target peptides to sample")
    parser.add_argument("--num_generations", type=int, default=16,
                        help="Generations per peptide (GDPO group size)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=200,
                        help="Set high to observe natural stopping")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="results/inference_analysis")
    parser.add_argument("--show_tokens", action="store_true",
                        help="Show BPE tokenization breakdown for generated sequences")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    import random
    random.seed(args.seed)

    # Load test peptides and sample targets
    test_df = pd.read_csv(args.test_csv)
    if "peptide" not in test_df.columns and "epitope" in test_df.columns:
        test_df = test_df.rename(columns={"epitope": "peptide"})

    all_peptides = test_df["peptide"].unique().tolist()
    targets = random.sample(all_peptides, min(args.num_targets, len(all_peptides)))

    print(f"\n  Targets: {len(targets)} peptides x {args.num_generations} generations = "
          f"{len(targets) * args.num_generations} total")
    print(f"  Temperature: {args.temperature}, top_p: {args.top_p}")

    # Generate: repeat each target num_generations times
    peptides_all = []
    for t in targets:
        peptides_all.extend([t] * args.num_generations)

    model, tokenizer = load_model(args.checkpoint, args.device)

    all_results = []
    for i in tqdm(range(0, len(peptides_all), args.batch_size), desc="  Generating"):
        batch = peptides_all[i : i + args.batch_size]
        all_results.extend(generate_batch(
            model, tokenizer, batch,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device,
        ))

    df = pd.DataFrame(all_results)

    # ── Per-target group summary (mimics GDPO rollout groups) ────────
    print(f"\n{'='*70}")
    print(f"  RESULTS: temp={args.temperature}, top_p={args.top_p}")
    print(f"  Checkpoint: {os.path.basename(args.checkpoint)}")
    print(f"{'='*70}")

    for target in targets:
        group = df[df["peptide"] == target]
        print(f"\n  Target: {target[:40]}...")
        print(f"    AA lengths:  {group['aa_length'].tolist()}")
        print(f"    Mean AA:     {group['aa_length'].mean():.1f} +/- {group['aa_length'].std():.1f}")
        print(f"    Token count: mean={group['token_count'].mean():.1f}, "
              f"min={group['token_count'].min()}, max={group['token_count'].max()}")
        print(f"    126 AA:      {group['is_126aa'].sum()}/{len(group)}")
        print(f"    Terminated:  {group['terminated'].sum()}/{len(group)}")
        print(f"    Ends GGGGS:  {group['ends_GGGGS'].sum()}/{len(group)}")

    # ── Aggregate summary ────────────────────────────────────────────
    print(f"\n  {'─'*50}")
    print(f"  AGGREGATE ({len(df)} sequences)")
    print(f"  {'─'*50}")
    print(f"  AA length:    mean={df['aa_length'].mean():.1f}, "
          f"std={df['aa_length'].std():.1f}, "
          f"median={df['aa_length'].median():.0f}")
    print(f"  Token count:  mean={df['token_count'].mean():.1f}, "
          f"std={df['token_count'].std():.1f}, "
          f"p99={df['token_count'].quantile(0.99):.0f}")
    print(f"  AA/token:     {(df['aa_length'] / df['token_count'].clip(1)).mean():.2f}")
    print(f"  Frac 126 AA:  {df['is_126aa'].mean():.3f}")
    print(f"  Frac terminated: {df['terminated'].mean():.3f}")
    print(f"  Frac ends GGGGS: {df['ends_GGGGS'].mean():.3f}")

    # Recommended max_completion_length
    terminated = df[df["terminated"]]
    if len(terminated) > 0:
        p99 = terminated["token_count"].quantile(0.99)
        rec = int(math.ceil(p99 * 1.15))
        print(f"\n  >>> Recommended max_completion_length: {rec}")
        print(f"      (p99 terminated = {p99:.0f} tokens + 15% headroom)")
    else:
        print(f"\n  >>> WARNING: No sequences terminated naturally!")

    print(f"{'='*70}\n")

    # ── Tokenization analysis (first seq per target) ────────────────
    if args.show_tokens:
        # Nb-v1 backbone: 126 AA = 121 AA core + 5 AA GGGGS suffix
        suffix_str = NBV1.required_suffix or ""  # "GGGGS"
        suffix_len = len(suffix_str)                   # 5

        print(f"\n{'='*70}")
        print(f"  TOKENIZATION ANALYSIS")
        print(f"  Nb-v1 backbone: {NBV1.length} AA, "
              f"suffix='{suffix_str}' ({suffix_len} AA)")
        print(f"{'='*70}")

        # First: tokenize the REFERENCE sequence for baseline
        ref = NB_V1_REFERENCE  # 126 AA canonical
        ref_formatted = add_newline_every_60_characters(ref)
        ref_tokens = tokenizer.tokenize(ref_formatted)
        ref_core = ref[:-suffix_len]  # 121 AA without GGGGS
        ref_core_formatted = add_newline_every_60_characters(ref_core)
        ref_core_tokens = tokenizer.tokenize(ref_core_formatted)
        ref_suffix_tokens = tokenizer.tokenize(suffix_str)

        print(f"\n  REFERENCE ({len(ref)} AA): {ref[:25]}...{ref[-10:]}")
        print(f"    Full sequence:       {len(ref_tokens)} tokens")
        print(f"    Core (no GGGGS):     {len(ref_core_tokens)} tokens")
        print(f"    Suffix '{suffix_str}':        {len(ref_suffix_tokens)} tokens")
        print(f"    Core + Suffix:       {len(ref_core_tokens) + len(ref_suffix_tokens)} tokens")
        diff = (len(ref_core_tokens) + len(ref_suffix_tokens)) - len(ref_tokens)
        print(f"    Boundary effect:     {diff:+d} tokens "
              f"({'NONE' if diff == 0 else 'BPE merges differ at join'})")

        print(f"\n    Full token map:")
        _print_token_map(tokenizer, ref_formatted, indent=6)
        print(f"\n    Core-only token map:")
        _print_token_map(tokenizer, ref_core_formatted, indent=6)
        print(f"\n    Suffix-only token map:")
        _print_token_map(tokenizer, suffix_str, indent=6)

        # Now: same analysis on generated sequences
        for target in targets:
            group = df[df["peptide"] == target]
            seq = group.iloc[0]["clean_aa"]
            if len(seq) == 0:
                continue

            print(f"\n  {'─'*60}")
            print(f"  GENERATED ({len(seq)} AA): {seq[:25]}...{seq[-10:]}")

            formatted_seq = add_newline_every_60_characters(seq)
            full_tokens = tokenizer.tokenize(formatted_seq)

            # Split using Nb-v1 suffix
            has_suffix = seq.endswith(suffix_str) if suffix_str else False
            if has_suffix and len(seq) > suffix_len:
                core = seq[:-suffix_len]
                core_formatted = add_newline_every_60_characters(core)
                core_tokens = tokenizer.tokenize(core_formatted)
                suf_tokens = tokenizer.tokenize(suffix_str)

                print(f"    Full sequence:       {len(full_tokens)} tokens")
                print(f"    Core (no {suffix_str}):     {len(core_tokens)} tokens")
                print(f"    Suffix '{suffix_str}':        {len(suf_tokens)} tokens")
                print(f"    Core + Suffix:       {len(core_tokens) + len(suf_tokens)} tokens")
                diff = (len(core_tokens) + len(suf_tokens)) - len(full_tokens)
                print(f"    Boundary effect:     {diff:+d} tokens "
                      f"({'NONE' if diff == 0 else 'BPE merges differ at join'})")
            else:
                print(f"    Full sequence:       {len(full_tokens)} tokens")
                print(f"    (no '{suffix_str}' suffix detected)")

            print(f"\n    Full token map:")
            _print_token_map(tokenizer, formatted_seq, indent=6)

        print(f"\n{'='*70}\n")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"inference_t{args.temperature}_p{args.top_p}.csv")
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    del model
    torch.cuda.empty_cache()


def _print_token_map(tokenizer, text: str, indent: int = 4):
    """Print each BPE token and the text it decodes to."""
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    pad = " " * indent
    for j, (tok, tid) in enumerate(zip(tokens, token_ids)):
        decoded = tokenizer.decode([tid])
        # Show the token, its ID, and what AA(s) it maps to
        aa_chars = "".join(c for c in decoded.upper() if c in STANDARD_AA_SET)
        n_aa = len(aa_chars)
        print(f"{pad}[{j:2d}] {tok:<15s} id={tid:<6d} -> '{decoded}' ({n_aa} AA)")


if __name__ == "__main__":
    main()
