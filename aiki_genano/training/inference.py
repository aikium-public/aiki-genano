"""
Inference script for NanoBody generation.
Generates protein binders given peptide inputs using trained SFT checkpoints.

Usage:
    python -m aiki_genano.training.inference \
        --checkpoint_1 /path/to/checkpoint-1000 \
        --checkpoint_2 /path/to/checkpoint-4000 \
        --test_csv data/testing.csv \
        --output_dir results/generations \
        --num_samples 50 \
        --batch_size 4
"""

import argparse
import json
import os
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import math
from typing import Optional

# Import existing utilities
from aiki_genano.training.data_utils import (
    add_newline_every_60_characters,
)


def _resolve_base_model_name(checkpoint_path: str) -> str:
    """Resolve the LoRA's base_model_name_or_path against the local filesystem.

    Returns either a valid local path, a HuggingFace repo id, or — as a last
    resort — `nferruz/ProtGPT2` (with a warning, because that's NOT the
    correct backbone for our DPO/GDPO adapters).
    """
    fallback = "nferruz/ProtGPT2"
    cfg_path = os.path.join(checkpoint_path, "adapter_config.json")
    if not os.path.exists(cfg_path):
        return fallback

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            recorded = json.load(f).get("base_model_name_or_path", fallback)
    except Exception as exc:
        print(f"   WARN: could not parse adapter_config.json: {exc}")
        return fallback

    # Case 1: recorded path exists.
    if os.path.exists(recorded):
        return recorded

    # Case 2: HF repo id format ("ns/repo" or "repo"); not a leading-slash path.
    if not recorded.startswith("/") and " " not in recorded:
        return recorded

    # Case 3: stale absolute path. Try to find SFT/<basename> by walking up.
    basename = os.path.basename(recorded.rstrip("/"))
    if basename:
        current = os.path.abspath(checkpoint_path)
        for _ in range(4):
            parent = os.path.dirname(current)
            if parent == current:
                break
            candidate = os.path.join(parent, "SFT", basename)
            if os.path.isdir(candidate):
                print(f"   Resolved stale base path → {candidate}")
                return candidate
            # Also try parent directly (in case basename matches a top-level SFT)
            sibling = os.path.join(parent, "SFT")
            if os.path.isdir(sibling):
                # Pick the first subdir that looks like a model (has config.json)
                for entry in sorted(os.listdir(sibling)):
                    full = os.path.join(sibling, entry)
                    if os.path.isdir(full) and os.path.exists(os.path.join(full, "config.json")):
                        print(f"   Resolved stale base path via SFT sibling → {full}")
                        return full
            current = parent

    print(f"   WARN: could not resolve recorded base '{recorded}' on this filesystem; "
          f"falling back to {fallback}. DPO/GDPO outputs may be wrong because the "
          f"adapter was trained on top of the SFT-merged checkpoint, not raw ProtGPT2.")
    return fallback


def load_model_from_checkpoint(checkpoint_path, device="cuda"):
    """
    Load model from a training checkpoint.
    
    Handles two cases:
    1. Merged model (has model.safetensors) - load directly
    2. LoRA checkpoint (has adapter_model.safetensors) - load base + adapter
    """
    print(f"\n Loading model from: {checkpoint_path}")
    
    # Check if this is a merged model or LoRA checkpoint
    is_merged = os.path.exists(os.path.join(checkpoint_path, "model.safetensors"))
    is_lora = os.path.exists(os.path.join(checkpoint_path, "adapter_model.safetensors"))
    
    if is_merged:
        # Case 1: Full merged model - load directly
        print("   Type: Merged model (loading directly)")
        
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float16,
            device_map=device,
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        
    elif is_lora:
        # Case 2: LoRA checkpoint - need base model + adapter
        print("   Type: LoRA adapter (loading base + adapter)")

        # adapter_config.json's base_model_name_or_path was recorded at training
        # time on an absolute filesystem path that no longer exists in the
        # released bundle. Resolve here:
        #   1. If it's an existing path → use as-is.
        #   2. If it's a valid HF repo id (no leading /) → use as-is.
        #   3. If it's a stale absolute path: keep the basename and look for
        #      an SFT/<basename> subdir inside the same checkpoint root as the
        #      LoRA dir (works for both /vol/checkpoints/ on Modal and
        #      /app/checkpoints/ on Docker).
        #   4. Fall back to nferruz/ProtGPT2 with a clear warning. The DPO/GDPO
        #      LoRA adapters were trained on top of the SFT-merged model, NOT
        #      raw ProtGPT2, so this fallback yields an incorrect backbone —
        #      hence the warning.
        base_model_name = _resolve_base_model_name(checkpoint_path)
        print(f"   Base model: {base_model_name}")
        
        # Load tokenizer from checkpoint (has ChatML tokens: 50259)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        print(f"   Tokenizer vocab size: {len(tokenizer)}")
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map=device,
        )
        
        # Resize embeddings to match checkpoint tokenizer (50257 -> 50259)
        base_model.resize_token_embeddings(len(tokenizer))
        
        # Load LoRA adapter on top
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
    else:
        raise ValueError(f"Could not find model files in {checkpoint_path}")
    
    model.eval()
    
    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # IMPORTANT: For decoder-only models with batched generation, use left-padding
    # so all sequences end at the same position for proper generation
    tokenizer.padding_side = 'left'
    
    print(f"   Model loaded successfully (padding_side=left)")
    return model, tokenizer


def format_prompt(peptide_seq):
    """Format peptide sequence into ChatML prompt."""
    formatted = add_newline_every_60_characters(peptide_seq)
    prompt = f"<|im_start|>user\n{formatted}<|im_end|>\n<|im_start|>assistant\n"
    return prompt


def generate_batch(model, tokenizer, peptides, sampling_config, device="cuda"):
    """
    Generate protein binders for a batch of peptides.
    
    Args:
        model: Loaded model
        tokenizer: Tokenizer
        peptides: List of peptide sequences
        sampling_config: Dict with generation parameters
        device: Device to use
    
    Returns:
        List of generated protein sequences
    """
    # Format prompts
    prompts = [format_prompt(p) for p in peptides]
    
    # Tokenize
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=512
    ).to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
            **sampling_config
        )
    
    # Decode - only the generated part
    generated_seqs = []
    for i, output in enumerate(outputs):
        # Get only the newly generated tokens
        input_len = inputs["input_ids"][i].shape[0]
        generated_tokens = output[input_len:]
        
        # Decode
        decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        # Clean up - remove newlines and whitespace
        clean_seq = decoded.replace("\n", "").strip()
        generated_seqs.append(clean_seq)
    
    return generated_seqs


def run_inference(
    checkpoint_path,
    test_csv,
    output_path,
    num_samples=None,  # None = all samples
    batch_size=4,
    device="cuda",
    strategies=None,  # None = all strategies
    num_generations_per_epitope: int = 1,  # Repeat each input row this many times
    max_new_tokens: int = 256,
    nucleus_top_p=None,  # optional list[float]
    topk_values=None,  # optional list[int]
    temperatures=None,  # optional list[float]
):
    """
    Run inference on test set with multiple sampling strategies.
    
    Returns DataFrame with results.
    """
    # Load model
    model, tokenizer = load_model_from_checkpoint(checkpoint_path, device)
    
    # Load test data
    print(f"\nLoading test data from: {test_csv}")
    test_df = pd.read_csv(test_csv)
    
    # Support epitopes.csv (epitope column) by mapping to peptide
    if "peptide" not in test_df.columns and "epitope" in test_df.columns:
        test_df = test_df.rename(columns={"epitope": "peptide"})

    if "peptide" not in test_df.columns:
        raise ValueError("Input CSV must contain a 'peptide' column (or 'epitope' which will be renamed).")

    # Ensure protein column exists for compatibility (may be empty for epitope-only generation)
    if "protein" not in test_df.columns:
        test_df["protein"] = ""

    # Sample if needed
    if num_samples and num_samples < len(test_df):
        test_df = test_df.sample(n=num_samples, random_state=42).reset_index(drop=True)
    
    # Repeat each input row N times (to get multiple generations per epitope)
    if num_generations_per_epitope < 1:
        raise ValueError("--num_generations_per_epitope must be >= 1")

    test_df = test_df.reset_index(drop=True)
    test_df["input_row_id"] = test_df.index.astype(int)

    if num_generations_per_epitope > 1:
        test_df = test_df.loc[test_df.index.repeat(num_generations_per_epitope)].reset_index(drop=True)
        test_df["replicate_id"] = test_df.groupby("input_row_id").cumcount().astype(int)
    else:
        test_df["replicate_id"] = 0

    print(f"Processing {len(test_df)} requests (unique peptides={test_df['peptide'].nunique()}, per_input={num_generations_per_epitope})")
    
    # Default available sampling strategies (preserve backward compatibility)
    all_sampling_strategies = {
        "greedy": {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        },
        "temp_07": {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": 0.7,
        },
        "nucleus_p095": {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "top_p": 0.95,
            "temperature": 1.0,
        },
        "topk_50": {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "top_k": 50,
            "temperature": 1.0,
        },
    }

    # Optional: build custom sampling grids
    # Example:
    #   --nucleus_top_p 0.95 0.90 --temperatures 0.7 1.0 1.2
    # will produce strategies:
    #   nucleus_p095_t07, nucleus_p095_t10, nucleus_p095_t12,
    #   nucleus_p090_t07, nucleus_p090_t10, nucleus_p090_t12
    custom_sampling_strategies = {}
    if nucleus_top_p is not None:
        if temperatures is None:
            raise ValueError("If --nucleus_top_p is provided, you must also provide --temperatures.")
        for p in nucleus_top_p:
            if not (0.0 < p <= 1.0) or math.isnan(p):
                raise ValueError(f"Invalid top_p: {p}")
            for t in temperatures:
                if t <= 0.0 or math.isnan(t):
                    raise ValueError(f"Invalid temperature: {t}")
                p_str = f"{p:.2f}".replace(".", "")
                t_str = f"{t:.1f}".replace(".", "")
                name = f"nucleus_p{p_str}_t{t_str}"
                custom_sampling_strategies[name] = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": True,
                    "top_p": float(p),
                    "temperature": float(t),
                }

    if topk_values is not None:
        if temperatures is None:
            raise ValueError("If --topk_values is provided, you must also provide --temperatures.")
        for k in topk_values:
            if k <= 0:
                raise ValueError(f"Invalid top_k: {k}")
            for t in temperatures:
                if t <= 0.0 or math.isnan(t):
                    raise ValueError(f"Invalid temperature: {t}")
                t_str = f"{t:.1f}".replace(".", "")
                name = f"topk_{int(k)}_t{t_str}"
                custom_sampling_strategies[name] = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": True,
                    "top_k": int(k),
                    "temperature": float(t),
                }
    
    # Filter strategies if specified
    if strategies:
        # Always allow selecting from either default strategies or custom-generated ones (by name)
        merged = {**all_sampling_strategies, **custom_sampling_strategies}
        sampling_strategies = {k: v for k, v in merged.items() if k in strategies}
        if not sampling_strategies:
            raise ValueError(f"No strategies matched. Available: {sorted(list(merged.keys()))}")
        print(f"Using strategies: {list(sampling_strategies.keys())}")
    else:
        # If user provided custom grid args, use ONLY custom strategies (avoid accidental 4x compute)
        if custom_sampling_strategies:
            sampling_strategies = custom_sampling_strategies
            print(f"Using custom strategies: {list(sampling_strategies.keys())}")
        else:
            sampling_strategies = all_sampling_strategies
            print(f"Using ALL strategies: {list(sampling_strategies.keys())}")
    
    # Results storage
    results = {
        "peptide": test_df["peptide"].tolist(),
        "input_row_id": test_df["input_row_id"].tolist(),
        "replicate_id": test_df["replicate_id"].tolist(),
        "true_protein": test_df["protein"].tolist(),
        "true_protein_len": [len(p) for p in test_df["protein"]],
    }

    # Preserve common metadata columns if present
    for col in ["source_file", "dataset_type"]:
        if col in test_df.columns:
            results[col] = test_df[col].tolist()
    
    # Generate with each strategy
    for strategy_name, config in sampling_strategies.items():
        print(f"\nGenerating with strategy: {strategy_name}")
        
        all_generations = []
        peptides = test_df["peptide"].tolist()
        
        # Batch generation
        for i in tqdm(range(0, len(peptides), batch_size), desc=f"  {strategy_name}"):
            batch_peptides = peptides[i:i+batch_size]
            batch_generations = generate_batch(
                model, tokenizer, batch_peptides, config, device
            )
            all_generations.extend(batch_generations)
        
        # Store results
        results[f"gen_{strategy_name}"] = all_generations
        results[f"gen_{strategy_name}_len"] = [len(g) for g in all_generations]
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("GENERATION SUMMARY")
    print("="*60)
    print(f"True protein lengths:     mean={results_df['true_protein_len'].mean():.1f}, "
          f"std={results_df['true_protein_len'].std():.1f}")
    
    for strategy in sampling_strategies.keys():
        col = f"gen_{strategy}_len"
        print(f"{strategy:20s}:  mean={results_df[col].mean():.1f}, "
              f"std={results_df[col].std():.1f}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return results_df


def _format_float_list(vals) -> str:
    # Use a stable, filename-friendly representation
    return "-".join([str(v).rstrip("0").rstrip(".") for v in vals])


def _auto_run_tag(
    strategies: Optional[list],
    num_generations_per_epitope: int,
    nucleus_top_p,
    topk_values,
    temperatures,
) -> Optional[str]:
    """
    Create a compact tag to avoid overwriting outputs for common grid runs.
    Returns None if nothing special is detected.
    """
    if strategies:
        return None

    if nucleus_top_p is not None and temperatures is not None:
        return f"nucleus_p{_format_float_list(nucleus_top_p)}_t{_format_float_list(temperatures)}_n{num_generations_per_epitope}"
    if topk_values is not None and temperatures is not None:
        return f"topk_k{'-'.join(map(str, topk_values))}_t{_format_float_list(temperatures)}_n{num_generations_per_epitope}"
    return None


def compare_checkpoints(
    checkpoint_1,
    checkpoint_2,
    test_csv,
    output_dir,
    num_samples=None,
    batch_size=4,
    device="cuda",
    strategies=None,
    num_generations_per_epitope: int = 1,
    max_new_tokens: int = 256,
    nucleus_top_p=None,
    topk_values=None,
    temperatures=None,
):
    """
    Generate from two checkpoints for comparison.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract checkpoint names
    ckpt1_name = os.path.basename(checkpoint_1)
    ckpt2_name = os.path.basename(checkpoint_2)
    
    print("\n" + "="*60)
    print("COMPARING CHECKPOINTS")
    print("="*60)
    print(f"  Checkpoint 1: {ckpt1_name}")
    print(f"  Checkpoint 2: {ckpt2_name}")
    
    # Generate from checkpoint 1
    print(f"\n{'='*60}")
    print(f"CHECKPOINT 1: {ckpt1_name}")
    print("="*60)
    output_1 = os.path.join(output_dir, f"generations_{ckpt1_name}.csv")
    df1 = run_inference(
        checkpoint_1,
        test_csv,
        output_1,
        num_samples,
        batch_size,
        device,
        strategies,
        num_generations_per_epitope,
        max_new_tokens,
        nucleus_top_p,
        topk_values,
        temperatures,
    )
    
    # Generate from checkpoint 2
    print(f"\n{'='*60}")
    print(f"CHECKPOINT 2: {ckpt2_name}")
    print("="*60)
    output_2 = os.path.join(output_dir, f"generations_{ckpt2_name}.csv")
    df2 = run_inference(
        checkpoint_2,
        test_csv,
        output_2,
        num_samples,
        batch_size,
        device,
        strategies,
        num_generations_per_epitope,
        max_new_tokens,
        nucleus_top_p,
        topk_values,
        temperatures,
    )
    
    # Print comparison
    print("\n" + "="*60)
    print("CHECKPOINT COMPARISON")
    print("="*60)
    print(f"{'Metric':<30} {ckpt1_name:<15} {ckpt2_name:<15}")
    print("-"*60)
    
    # Compare only the strategies that were run
    compare_strategies = strategies if strategies else ["greedy", "temp_07", "nucleus_p095", "topk_50"]
    for strategy in compare_strategies:
        col = f"gen_{strategy}_len"
        if col in df1.columns and col in df2.columns:
            print(f"{strategy + ' mean len':<30} {df1[col].mean():<15.1f} {df2[col].mean():<15.1f}")
    
    print(f"{'true_protein mean len':<30} {df1['true_protein_len'].mean():<15.1f} {df2['true_protein_len'].mean():<15.1f}")
    
    return df1, df2


def main():
    parser = argparse.ArgumentParser(description="NanoBody Inference")
    # Single-checkpoint mode (preferred): --checkpoint
    # Backwards compatible: --checkpoint_1 is still supported (and required if --checkpoint not provided)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (single-model inference). Alias for --checkpoint_1")
    parser.add_argument("--checkpoint_1", type=str, default=None,
                        help="Path to first checkpoint (use --checkpoint for single-model inference)")
    parser.add_argument("--checkpoint_2", type=str, default=None,
                        help="Path to second checkpoint (optional, for comparison)")
    parser.add_argument("--test_csv", type=str, default="data/testing.csv",
                        help="Path to test CSV file")
    parser.add_argument("--output_dir", type=str, default="results/generations",
                        help="Output directory for results")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Number of samples to process (default: all)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--strategies", type=str, nargs="+", default=None,
                        help="Sampling strategies to use (default: all)")
    parser.add_argument("--num_generations_per_epitope", type=int, default=1,
                        help="Repeat each input row this many times to get N generations per epitope")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum number of new tokens to generate for the protein")
    parser.add_argument("--nucleus_top_p", type=float, nargs="+", default=None,
                        help="If provided, build nucleus sampling strategies for each top_p in this list")
    parser.add_argument("--topk_values", type=int, nargs="+", default=None,
                        help="If provided, build top-k sampling strategies for each k in this list")
    parser.add_argument("--temperatures", type=float, nargs="+", default=None,
                        help="Temperature list to combine with nucleus_top_p and/or topk_values")
    parser.add_argument("--run_tag", type=str, default=None,
                        help="Optional string appended to output filename to prevent overwriting (e.g., topk_k20-50_t0.8-1.0-1.5)")
    
    args = parser.parse_args()

    # Normalize checkpoint args
    checkpoint_1 = args.checkpoint if args.checkpoint is not None else args.checkpoint_1
    if checkpoint_1 is None:
        raise ValueError("You must provide --checkpoint (preferred) or --checkpoint_1.")
    
    if args.checkpoint_2:
        # Compare two checkpoints
        compare_checkpoints(
            checkpoint_1,
            args.checkpoint_2,
            args.test_csv,
            args.output_dir,
            args.num_samples,
            args.batch_size,
            args.device,
            args.strategies,
            args.num_generations_per_epitope,
            args.max_new_tokens,
            args.nucleus_top_p,
            args.topk_values,
            args.temperatures,
        )
    else:
        # Single checkpoint inference
        # Add strategy suffix to filename if specific strategies selected.
        # If running a custom grid without --strategies, auto-tag to prevent overwriting.
        auto_tag = _auto_run_tag(
            args.strategies,
            args.num_generations_per_epitope,
            args.nucleus_top_p,
            args.topk_values,
            args.temperatures,
        )
        tag = args.run_tag or auto_tag
        if args.strategies:
            strategy_suffix = f"_{'_'.join(args.strategies)}"
        elif tag:
            strategy_suffix = f"_{tag}"
        else:
            strategy_suffix = "_all"
        output_path = os.path.join(
            args.output_dir, 
            f"generations_{os.path.basename(checkpoint_1)}{strategy_suffix}.csv"
        )
        run_inference(
            checkpoint_1,
            args.test_csv,
            output_path,
            args.num_samples,
            args.batch_size,
            args.device,
            args.strategies,
            args.num_generations_per_epitope,
            args.max_new_tokens,
            args.nucleus_top_p,
            args.topk_values,
            args.temperatures,
        )


if __name__ == "__main__":
    main()
