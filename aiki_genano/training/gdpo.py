"""
GDPO Training for Protein Binder Generation
Using NVIDIA's TRL-GDPO Implementation

Paper: https://arxiv.org/abs/2601.05242
GitHub: https://github.com/NVlabs/GDPO

Key insight: GDPO normalizes each reward SEPARATELY before combining.
We pass individual property rewards (not pre-combined TDS/MFS) so GDPO
can do proper per-reward normalization.

Run with:
    python -m aiki_genano.training.gdpo --config-path=conf --config-name=config_gdpo
"""

import logging
import os
import random
from typing import List, Dict, Optional

import hydra
import torch
from accelerate import Accelerator
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# TRL imports (requires NVIDIA's trl-gdpo installation)
from trl import GRPOTrainer, GRPOConfig

# NBv1 reward functions — literature-grounded, single source of truth.
# See rewards.py for per-function references and design rationale.
from aiki_genano.rewards.rewards import (
    fr2_aggregation_reward,
    hydrophobic_patch_reward,
    liability_reward,
    expression_reward,
    vhh_hallmark_reward,
    scaffold_integrity_reward,
)

from aiki_genano.training.data_utils import add_newline_every_60_characters
from aiki_genano.training.utils import print_trainable_parameters

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


##################################################
# UTILITY FUNCTIONS
##################################################

def _find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """
    Return the latest HF Trainer checkpoint directory inside output_dir,
    e.g. output_dir/checkpoint-1200. Returns None if not found.
    """
    if not output_dir or not os.path.isdir(output_dir):
        return None
    candidates = []
    for name in os.listdir(output_dir):
        if not name.startswith("checkpoint-"):
            continue
        path = os.path.join(output_dir, name)
        if not os.path.isdir(path):
            continue
        try:
            step = int(name.split("-", 1)[1])
        except Exception:
            continue
        candidates.append((step, path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _subset_dataset_by_peptide(
    ds,
    *,
    max_unique_peptides: Optional[int],
    samples_per_peptide: Optional[int],
    seed: int,
):
    """
    Create a representative subset by sampling across *unique peptides* (targets).

    Why: this dataset has ~1.35M rows but only ~65 unique peptides, so naive "first N rows"
    can collapse to just a few repeated targets.

    Requires the dataset to have a 'peptide' column.
    """
    if max_unique_peptides is None and samples_per_peptide is None:
        return ds

    if "peptide" not in ds.column_names:
        raise ValueError("Peptide-based subsetting requires a 'peptide' column in the dataset.")

    rng = random.Random(seed)
    peptides = list(ds["peptide"])
    unique_peptides = sorted(set(peptides))
    rng.shuffle(unique_peptides)

    if max_unique_peptides is not None:
        unique_peptides = unique_peptides[: max_unique_peptides]

    selected = set(unique_peptides)

    if samples_per_peptide is not None:
        k = int(samples_per_peptide)
        if k <= 0:
            raise ValueError("samples_per_peptide must be > 0")

        reservoirs: Dict[str, List[int]] = {p: [] for p in selected}
        seen: Dict[str, int] = {p: 0 for p in selected}

        for idx, pep in enumerate(peptides):
            if pep not in selected:
                continue
            seen[pep] += 1
            r = reservoirs[pep]
            if len(r) < k:
                r.append(idx)
            else:
                j = rng.randrange(seen[pep])
                if j < k:
                    r[j] = idx

        indices: List[int] = []
        for pep in selected:
            indices.extend(reservoirs[pep])
        indices.sort()
        return ds.select(indices)

    indices = [i for i, pep in enumerate(peptides) if pep in selected]
    return ds.select(indices)


##################################################
# DATA LOADING
##################################################

def load_dataset_for_gdpo(data_dir: str):
    """Load SFT dataset and format for GDPO (prompts only)."""
    
    train_path = os.path.join(data_dir, 'training.csv')
    test_path = os.path.join(data_dir, 'testing.csv')
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found: {train_path}")
    
    data = load_dataset('csv', data_files={
        'train': train_path,
        'test': test_path
    })
    
    def format_prompt(sample):
        peptide = sample["peptide"]
        prompt = (
            "<|im_start|>user\n"
            + add_newline_every_60_characters(peptide)
            + "<|im_end|>\n"
            + "<|im_start|>assistant\n"
        )
        return {"prompt": prompt, "peptide": peptide}
    
    remove_cols = [c for c in data["train"].column_names if c != "peptide"]
    train_ds = data["train"].map(format_prompt, remove_columns=remove_cols)
    remove_cols_test = [c for c in data["test"].column_names if c != "peptide"]
    eval_ds = data["test"].map(format_prompt, remove_columns=remove_cols_test)
    
    logger.info(f"Loaded {len(train_ds)} train, {len(eval_ds)} eval samples")
    return train_ds, eval_ds


# Fixed NBv1 reward ordering (must match reward_funcs list in trainer).
NBV1_REWARD_NAMES: List[str] = [
    "fr2_aggregation",
    "hydrophobic_patch",
    "liability",
    "expression",
    "vhh_hallmark",
    "scaffold_integrity",
]
NBV1_N_REWARDS = len(NBV1_REWARD_NAMES)  # 6


##################################################
# MAIN
##################################################

@hydra.main(config_path="../../configs/gdpo", config_name="dpo_final_gated", version_base="1.1")
def main(cfg: DictConfig):
    logger.info("=" * 60)
    logger.info("GDPO Training - NVIDIA Implementation")
    logger.info("Individual rewards for per-reward normalization")
    logger.info("=" * 60)
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Extract config sections
    script_args = cfg.gdpo.script_args
    training_args_cfg = cfg.gdpo.training_args
    
    # Paths
    MODEL_PATH = script_args.sft_model_path
    DATA_DIR = script_args.data_split_dir
    OUTPUT_DIR = training_args_cfg.output_dir
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Reward weights from config (must be exactly 6, one per reward)
    REWARD_WEIGHTS = list(script_args.reward_weights)
    if len(REWARD_WEIGHTS) != NBV1_N_REWARDS:
        raise ValueError(
            f"reward_weights must have exactly {NBV1_N_REWARDS} entries "
            f"({', '.join(NBV1_REWARD_NAMES)}), got {len(REWARD_WEIGHTS)}."
        )
    
    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"Data: {DATA_DIR}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info("Reward weights:")
    for name, w in zip(NBV1_REWARD_NAMES, REWARD_WEIGHTS):
        logger.info(f"  {name:25s}: {w}")
    
    # ============================================
    # LOAD MODEL
    # ============================================
    
    logger.info("Loading model...")
    
    from trl import setup_chat_format
    
    base_model_name = script_args.get("base_model_name", "nferruz/ProtGPT2")
    
    adapter_config_path = os.path.join(MODEL_PATH, "adapter_config.json")
    is_peft_checkpoint = os.path.exists(adapter_config_path)
    
    if is_peft_checkpoint:
        logger.info(f"Detected PEFT checkpoint at {MODEL_PATH}")
        logger.info(f"Loading base model: {base_model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map={"": Accelerator().local_process_index},
            trust_remote_code=True,
        )
        
        model.resize_token_embeddings(len(tokenizer))
        
        model = PeftModel.from_pretrained(model, MODEL_PATH, is_trainable=True)
        model = model.merge_and_unload()
        logger.info("SFT adapter merged")
        
    else:
        logger.info(f"Loading merged model from: {MODEL_PATH}")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map={"": Accelerator().local_process_index},
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    model.config.use_cache = False
    print_trainable_parameters(model)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    logger.info(f"Model loaded. Vocab: {len(tokenizer)}")
    
    # ============================================
    # LOAD DATA
    # ============================================
    
    logger.info("Loading dataset...")
    train_dataset, eval_dataset = load_dataset_for_gdpo(DATA_DIR)
    
    subset_seed = int(script_args.get("subset_seed", training_args_cfg.get("seed", 42)))
    if script_args.get("max_unique_peptides_train") or script_args.get("samples_per_peptide_train"):
        logger.info("Applying peptide-based train subsetting...")
        train_dataset = _subset_dataset_by_peptide(
            train_dataset,
            max_unique_peptides=script_args.get("max_unique_peptides_train"),
            samples_per_peptide=script_args.get("samples_per_peptide_train"),
            seed=subset_seed,
        )
    elif script_args.get("max_train_samples"):
        train_dataset = train_dataset.select(range(min(script_args.max_train_samples, len(train_dataset))))

    if script_args.get("max_unique_peptides_eval") or script_args.get("samples_per_peptide_eval"):
        logger.info("Applying peptide-based eval subsetting...")
        eval_dataset = _subset_dataset_by_peptide(
            eval_dataset,
            max_unique_peptides=script_args.get("max_unique_peptides_eval"),
            samples_per_peptide=script_args.get("samples_per_peptide_eval"),
            seed=subset_seed + 1,
        )
    elif script_args.get("max_eval_samples"):
        eval_dataset = eval_dataset.select(range(min(script_args.max_eval_samples, len(eval_dataset))))
    
    # Shuffle train data to ensure all peptides are sampled throughout training
    train_dataset = train_dataset.shuffle(seed=subset_seed)
    
    logger.info(f"Using {len(train_dataset)} train, {len(eval_dataset)} eval")
    
    # ============================================
    # LORA CONFIG
    # ============================================
    
    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=list(cfg.gdpo.lora_targets),
        bias="none",
        init_lora_weights=True,
        task_type="CAUSAL_LM",
    )
    
    # ============================================
    # GDPO CONFIG
    # ============================================
    
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=training_args_cfg.get("overwrite_output_dir", False),
        
        # >>> NVIDIA GDPO FLAG <<<
        apply_gdpo=True,
        
        # Reward weights (one per reward function)
        reward_weights=REWARD_WEIGHTS,
        
        # Generation
        num_generations=script_args.num_generations,
        max_completion_length=script_args.max_completion_length,
        temperature=script_args.temperature,
        top_k=script_args.get("top_k", 0),
        top_p=script_args.get("top_p", 0.95),
        
        # KL divergence (keeps model close to SFT, handles length/validity implicitly)
        beta=script_args.beta,
        
        # PPO-style clipping
        epsilon=training_args_cfg.get("epsilon", 0.1),
        epsilon_high=training_args_cfg.get("epsilon_high", None),
        loss_type=training_args_cfg.get("loss_type", "bnpo"),
        disable_dropout=training_args_cfg.get("disable_dropout", True),
        
        # Training
        per_device_train_batch_size=training_args_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=training_args_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=training_args_cfg.gradient_accumulation_steps,
        max_steps=training_args_cfg.max_steps,
        learning_rate=training_args_cfg.learning_rate,
        
        # Optimizer
        optim=training_args_cfg.get("optim", "paged_adamw_32bit"),
        weight_decay=training_args_cfg.get("weight_decay", 0.05),
        warmup_steps=training_args_cfg.get("warmup_steps", 50),
        lr_scheduler_type=training_args_cfg.get("lr_scheduler_type", "cosine"),
        
        # Logging
        logging_steps=training_args_cfg.logging_steps,
        report_to=training_args_cfg.report_to,
        
        # Saving
        save_steps=training_args_cfg.save_steps,
        save_total_limit=training_args_cfg.get("save_total_limit", 3),
        
        # Eval
        eval_strategy=training_args_cfg.get("eval_strategy", "steps"),
        eval_steps=training_args_cfg.get("eval_steps", 100),
        
        # Memory
        gradient_checkpointing=training_args_cfg.get("gradient_checkpointing", True),
        
        # Misc
        seed=training_args_cfg.get("seed", 42),
        bf16=training_args_cfg.get("bf16", False),
        fp16=training_args_cfg.get("fp16", True),
        max_grad_norm=training_args_cfg.get("max_grad_norm", 1.0),
        remove_unused_columns=False,
    )
    
    # ============================================
    # TRAINER with INDIVIDUAL REWARDS
    # ============================================
    
    logger.info("Initializing GDPO Trainer with NBv1 literature-grounded rewards...")
    logger.info("  6 rewards (from rewards.py), validity-gated except scaffold_integrity:")
    for i, name in enumerate(NBV1_REWARD_NAMES):
        logger.info(f"    [{i}] {name}")
    
    reward_funcs = [
        fr2_aggregation_reward,     # [0] FR2 mean KD hydrophobicity (Kyte-Doolittle 1982)
        hydrophobic_patch_reward,   # [1] Consecutive hydrophobic stretches (AGGRESCAN-inspired)
        liability_reward,           # [2] Chemical liabilities (Robinson & Robinson / Geiger & Clarke)
        expression_reward,          # [3] E. coli soluble expression (Wilkinson-Harrison 1991)
        vhh_hallmark_reward,        # [4] VHH FR2 tetrad (Muyldermans 2013)
        scaffold_integrity_reward,  # [5] Huber length + linker + Cys (NOT gated)
    ]
    
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        reward_funcs=reward_funcs,
    )
    
    # ============================================
    # TRAIN
    # ============================================
    
    logger.info("=" * 60)
    logger.info("Starting GDPO Training")
    logger.info(f"  apply_gdpo: True (per-reward normalization)")
    logger.info(f"  num_generations: {script_args.num_generations}")
    logger.info(f"  reward_weights: {REWARD_WEIGHTS}")
    logger.info(f"  beta (KL): {script_args.beta}")
    logger.info("=" * 60)

    resume_from_checkpoint = training_args_cfg.get("resume_from_checkpoint", None)
    if isinstance(resume_from_checkpoint, str) and resume_from_checkpoint.lower() == "auto":
        resume_from_checkpoint = _find_latest_checkpoint(OUTPUT_DIR)
        if resume_from_checkpoint is None:
            logger.info("resume_from_checkpoint=auto, but no checkpoint found; starting fresh.")
        else:
            logger.info(f"Auto-resuming from latest checkpoint: {resume_from_checkpoint}")
    elif resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # ============================================
    # SAVE
    # ============================================
    
    logger.info("Saving model...")
    final_dir = os.path.join(OUTPUT_DIR, "final_checkpoint")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    logger.info(f"GDPO adapter saved to: {final_dir}")
    
    # ============================================
    # MERGE AND SAVE (optional)
    # ============================================
    
    should_merge = script_args.get("merge_adapter", True)
    base_model_name = script_args.get("base_model_name", "nferruz/ProtGPT2")
    
    if should_merge:
        logger.info("Merging adapter weights...")
        
        try:
            del model
            del trainer
            torch.cuda.empty_cache()
            
            from trl import setup_chat_format
            
            logger.info(f"Loading base model: {base_model_name}")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                return_dict=True,
                torch_dtype=torch.float16,
            )
            base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            
            base_model, base_tokenizer = setup_chat_format(base_model, base_tokenizer)
            
            logger.info(f"Loading GDPO adapter from: {final_dir}")
            model = PeftModel.from_pretrained(base_model, final_dir)
            model = model.merge_and_unload()
            
            merged_dir = os.path.join(OUTPUT_DIR, script_args.new_model_name)
            model.save_pretrained(merged_dir)
            base_tokenizer.save_pretrained(merged_dir)
            
            logger.info(f"Merged model saved to: {merged_dir}")
            
        except Exception as e:
            logger.warning(f"Merge failed: {e}")
            logger.warning(f"Adapter checkpoint is still available at: {final_dir}")
    else:
        logger.info("Skipping merge (merge_adapter=False)")
        logger.info(f"Use adapter at: {final_dir}")
    
    logger.info("=" * 60)
    logger.info("GDPO Training Complete!")
    logger.info(f"  Adapter: {final_dir}")
    if should_merge:
        logger.info(f"  Merged:  {os.path.join(OUTPUT_DIR, script_args.new_model_name)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
