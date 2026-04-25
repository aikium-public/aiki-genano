import logging
import os
import pdb
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

import torch
from accelerate import Accelerator
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, DPOTrainer

from aiki_genano.training.hf_arguments import ScriptArgumentsDPO, TrainingArgumentsDPO
from aiki_genano.training.data_utils import get_protein_peptide_preference_datasets
from aiki_genano.training.utils import print_trainable_parameters


@hydra.main(config_path="../../configs", config_name="dpo_developability", version_base="1.1")
def main(cfg: DictConfig):
    logger.info("Starting SFT")
    logger.info(OmegaConf.to_yaml(cfg))

    script_args = ScriptArgumentsDPO(**cfg.dpo.script_args)
    training_args = TrainingArgumentsDPO(**cfg.dpo.training_args)

    # parser = HfArgumentParser((ScriptArgumentsDPO, TrainingArgumentsDPO))
    # script_args, training_args = parser.parse_args_into_dataclasses()

    ##########################s
    # -- load models
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        script_args.sft_model_path,
        quantization_config=bnb_config,
        device_map={"": Accelerator().local_process_index},
    )
    model.config.use_cache = False
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # NOTE (trl>=0.11): when training with PEFT adapters (peft_config=...),
    # TRL expects `ref_model=None`. Passing both `ref_model` and `peft_config`
    # raises an error.

    print_trainable_parameters(model)

    # -- load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.sft_model_path)
    tokenizer.pad_token = (
        tokenizer.unk_token
    )  # this is '<|endoftext|>' so this is different than eos_token and bos_token

    # -- Load the paired dataset
    train_dataset, eval_dataset = get_protein_peptide_preference_datasets(
        data_dir=script_args.data_split_dir,
        filter_to_max_length=script_args.filter_to_max_length,
        max_length=script_args.max_length,
    )
    # pdb.set_trace()
    # --- COLLATOR for instruction finetuning, loss is only computed for response
    # instruction_template = "<|im_start|>user\n"
    # response_template = "<|im_start|>assistant\n"
    # data_collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)

    # #--- 1. COLLATOR for first stage of SFT
    # data_collator = DataCollatorForLanguageModeling(
    #     tokenizer=tokenizer,
    #     mlm=False,
    # )
    data_collator = None
    # -- LoRA
    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=list(cfg.dpo.lora_targets),  # ['c_attn', 'c_proj', 'c_fc', 'lm_head', 'wte', 'wpe'],  # ["c_attn", "c_proj", "c_fc", "lm_head"]
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model=model,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        peft_config=peft_config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        max_target_length=script_args.max_target_length,
    )

    # 6. train (resume if configured)
    dpo_trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    # dpo_trainer.save_model(training_args.output_dir)

    # 7. save
    output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)

    ###############################################
    # Save merged model and tokenizer
    ###############################################
    # -- Reload model in FP16 (instead of NF4)
    base_model_name = script_args.sft_model_path
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        return_dict=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    # -- merge adapter weights with base, call it a new model
    model = PeftModel.from_pretrained(base_model, output_dir)
    model = model.merge_and_unload()

    new_model_dir = os.path.join(training_args.output_dir, script_args.new_model_name)
    model.save_pretrained(new_model_dir)
    tokenizer.save_pretrained(new_model_dir)

    # # Push them to the HF Hub
    # model.push_to_hub(new_model_name, use_temp_dir=False, token=hf_token)
    # tokenizer.push_to_hub(new_model_name, use_temp_dir=False, token=hf_token)


if __name__ == "__main__":
    main()
