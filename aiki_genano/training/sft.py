import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

import pdb

import torch
from accelerate import Accelerator
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer, setup_chat_format
from trl.import_utils import is_npu_available, is_xpu_available

from aiki_genano.training.hf_arguments import ScriptArgumentsSFT, TrainingArgumentsSFT
from aiki_genano.training.data_utils import create_sft_instructions
from aiki_genano.training.utils import print_trainable_parameters

@hydra.main(config_path="../../configs", config_name="sft_10k", version_base="1.1")
def main(cfg: DictConfig):
    logger.info("Starting SFT")
    logger.info(OmegaConf.to_yaml(cfg))

    script_args = ScriptArgumentsSFT(**cfg.sft.script_args)
    training_args = TrainingArgumentsSFT(**cfg.sft.training_args)
    # parser = HfArgumentParser((ScriptArgumentsSFT, TrainingArgumentsSFT))
    # script_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.group_by_length and script_args.packing:
        raise ValueError("Cannot use both packing and group by length")
    # `gradient_checkpointing` was True by default until `1f3314`, but it's actually not used.
    # `gradient_checkpointing=True` will cause `Variable._execution_engine.run_backward`.
    if training_args.gradient_checkpointing:
        raise ValueError("gradient_checkpointing not supported")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    output_dir = os.path.join(training_args.output_dir, "final_checkpoint")

    # -- load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        script_args.base_model_name,
        quantization_config=bnb_config,
        device_map={"": Accelerator().local_process_index},
    )
    base_model.config.use_cache = False
    print_trainable_parameters(base_model)

    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=list(cfg.sft.lora_targets),  # ['c_attn', 'c_proj', 'c_fc', 'lm_head', 'wte', 'wpe']
        bias="none",
        task_type="CAUSAL_LM",
    )

    # -- load tokenizer
    """
    trl.setup_chat_format(): adds special tokens to the tokenizer, e.g. <|im_start|> and <|im_end|>, to indicate the start and end of a conversation.
    Resizes the model’s embedding layer to accommodate the new tokens.
    Sets the chat_template of the tokenizer, which is used to format the input data into a chat-like format. The default is chatml from OpenAI.
    optionally you can pass resize_to_multiple_of to resize the embedding layer to a multiple of the resize_to_multiple_of argument, e.g. 64. If you want to see more formats being supported in the future, please open a GitHub issue on trl

    # chat template from OpenAI chatML. injected in tokenizer by trl.setup_chat_template() dy default, if you want to modify it do it here:
    """

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.base_model_name, trust_remote_code=True
    )  # , padding_side='left')
    base_model, tokenizer = setup_chat_format(base_model, tokenizer)
    tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.pad_token = (
        tokenizer.unk_token
    )  # this is '<|endoftext|>' so this is different than eos_token and bos_token

    # -- load datasets
    train_dataset, eval_dataset = create_sft_instructions(
        tokenizer,
        data_dir=script_args.data_split_dir,
    )

    if not script_args.response_sft_stage:
        # --- 1. COLLATOR for first stage of SFT
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
    else:
        # --- 2. COLLATOR for instruction finetuning, loss is only computed for response
        instruction_template = "<|im_start|>user\n"
        response_template = "<|im_start|>assistant\n"
        data_collator = DataCollatorForCompletionOnlyLM(
            instruction_template=instruction_template,
            response_template=response_template,
            tokenizer=tokenizer,
            mlm=False,
        )

    # #--- eval metrics during training
    # from datasets import load_metric
    # import numpy as np

    # accuracy = load_metric("accuracy")
    # rouge = load_metric("rouge")
    # ppl = load_metric("perplexity")
    # bleu = load_metric("bleu")

    # def compute_metrics(eval_pred):
    #     logits, labels = eval_pred
    #     predictions = np.argmax(logits, axis=-1)
    #     return ppl.compute(predictions=predictions, references=labels)

    # -- train
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        peft_config=peft_config,
        packing=script_args.packing,
        max_seq_length=training_args.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(output_dir)
    trainer.model.save_pretrained(output_dir, save_adapter=True, save_config=True)

    # Free memory for merging weights
    del base_model
    if is_xpu_available():
        torch.xpu.empty_cache()
    elif is_npu_available():
        torch.npu.empty_cache()
    else:
        torch.cuda.empty_cache()

    #############################################
    # Save merged model and tokenizer
    #############################################
    # -- Reload model in FP16 (instead of NF4)
    base_model = AutoModelForCausalLM.from_pretrained(
        script_args.base_model_name,
        return_dict=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(script_args.base_model_name)
    base_model, tokenizer = setup_chat_format(base_model, tokenizer)
    # -- merge adapter weights with base, call it a new model
    model = PeftModel.from_pretrained(base_model, output_dir)
    model = model.merge_and_unload()

    new_model_dir = os.path.join(training_args.output_dir, script_args.new_model_name)
    model.save_pretrained(new_model_dir)
    tokenizer.save_pretrained(new_model_dir)

    # model = AutoModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
    # model = model.merge_and_unload()

    # output_merged_dir = os.path.join(training_args.output_dir, "final_merged_checkpoint")
    # model.save_pretrained(output_merged_dir, safe_serialization=True)

    # from peft import PeftModel
    # peft_model = PeftModel.from_pretrained(base_model, output_dir, torch_dtype=torch.float16, offload_folder="lora_results/lora_7/temp")

    # ---- from https://github.com/pourion/deep-learning-pytorch-huggingface/blob/main/training/fine-tune-llms-in-2024-with-trl.ipynb
    #### COMMENT IN TO MERGE PEFT AND BASE MODEL ####
    # from peft import PeftModel, PeftConfig
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # from peft import AutoPeftModelForCausalLM

    # # Load PEFT model on CPU
    # config = PeftConfig.from_pretrained(args.output_dir)
    # model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,low_cpu_mem_usage=True)
    # tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    # model.resize_token_embeddings(len(tokenizer))
    # model = PeftModel.from_pretrained(model, args.output_dir)
    # model = AutoPeftModelForCausalLM.from_pretrained(
    #     args.output_dir,
    #     torch_dtype=torch.float16,
    #     low_cpu_mem_usage=True,
    # )
    # # Merge LoRA and base model and save
    # merged_model = model.merge_and_unload()
    # merged_model.save_pretrained(args.output_dir,safe_serialization=True, max_shard_size="2GB")


if __name__ == "__main__":
    main()
