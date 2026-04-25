from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments

# TRL >= 0.11 expects DPOTrainer(args=...) to be a DPOConfig (not plain TrainingArguments).
# We keep a safe fallback so this file can still import in environments without TRL installed.
try:
    from trl import DPOConfig  # type: ignore
except Exception:  # pragma: no cover
    DPOConfig = None  # type: ignore

####################################################
## SFT
####################################################


@dataclass
class ScriptArgumentsSFT:
    base_model_name: Optional[str] = field(
        default="nferruz/ProtGPT2", metadata={"help": "initial base model name: nferruz/ProtGPT2"}
    )
    new_model_name: Optional[str] = field(default="ProtQA-sft", metadata={"help": "name of saved model"})
    response_sft_stage: bool = field(
        default=False,
        metadata={
            "help": "whether to only compute loss on the responses. SFT has 2 stages: first this should be false, then it should be true."
        },
    )
    data_split_dir: Optional[str] = field(
        default="/data/aikium/protein_peptide_2024-01-31/dpo", metadata={"help": "the data_split_dir to use"}
    )
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    size_valid_set: Optional[int] = field(default=3345, metadata={"help": "the size of the validation set"})
    streaming: Optional[bool] = field(default=True, metadata={"help": "whether to stream the dataset"})
    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})
    packing: Optional[bool] = field(default=True, metadata={"help": "whether to use packing for SFTTrainer"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=16, metadata={"help": "the lora r parameter"})


@dataclass
class TrainingArgumentsSFT(TrainingArguments):
    resume_from_checkpoint: Optional[str] = field(default=None, metadata={"help": "checkpoint to restart from"})
    max_steps: Optional[int] = field(default=20000, metadata={"help": "max number of training steps"})

    output_dir: str = field(
        default="output_trl/ProtGPT2_2024-01-31_instruction/sft", metadata={"help": "the output directory"}
    )
    save_steps: Optional[int] = field(default=500, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=500, metadata={"help": "the evaluation frequency"})
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "number of epochs for training"})
    per_device_train_batch_size: Optional[int] = field(default=2, metadata={"help": "per device train batch size"})
    per_device_eval_batch_size: Optional[int] = field(default=8, metadata={"help": "per device eval batch size"})
    warmup_steps: Optional[int] = field(default=501, metadata={"help": "warmup steps"})
    logging_dir: Optional[str] = field(default="logs", metadata={"help": "logging directory"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "logging steps"})
    evaluation_strategy: Optional[str] = field(default="steps", metadata={"help": "evaluation strategy"})
    load_best_model_at_end: Optional[bool] = field(default=True, metadata={"help": "load best model at end"})
    save_strategy: Optional[str] = field(default="steps", metadata={"help": "save strategy"})
    metric_for_best_model: str = field(default="eval_loss", metadata={"help": "metric for best model"})
    save_total_limit: Optional[int] = field(default=80, metadata={"help": "save total limit"})
    gradient_accumulation_steps: Optional[int] = field(default=2, metadata={"help": "gradient accumulation"})
    report_to: Optional[str] = field(
        default="mlflow", metadata={"help": "remote logging reporting to, wandb, mlflow, etc"}
    )
    log_freq: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    find_unused_parameters: Optional[bool] = field(default=False, metadata={"help": "ddp_find_unused_parameters"})
    max_seq_length: Optional[int] = field(default=1024, metadata={"help": "the sequence length"})


####################################################
## DPO
####################################################


@dataclass
class ScriptArgumentsDPO:
    sft_model_path: Optional[str] = field(
        default="/results/aikium/output_trl/ProtGPT2_2024-01-31_instruction/sft/ProtQA-sft",
        metadata={"help": "the location of the SFT model name or path"},
    )
    new_model_name: Optional[str] = field(default="ProtQA-dpo", metadata={"help": "name of saved model"})
    
    data_split_dir: Optional[str] = field(
        default="/data/aikium/protein_peptide_2024-01-31/dpo", metadata={"help": "the data_split_dir to use"}
    )
    beta: Optional[float] = field(default=0.5, metadata={"help": "the beta parameter for DPO loss"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=16, metadata={"help": "the lora r parameter"})
    max_prompt_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum length of the prompt. This argument is required if you want to use the default data collator."
        },
    )
    max_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator."
        },
    )
    max_target_length: Optional[int] = field(default=50, metadata={"help": "The maximum length of the target."})
    filter_to_max_length: Optional[bool] = field(
        default=True, metadata={"help": "ensure sequence lengths less than max_length"}
    )
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type, inplace operation. See https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )


@dataclass
class TrainingArgumentsDPO(DPOConfig if DPOConfig is not None else TrainingArguments):
    output_dir: Optional[str] = field(
        default="/results/aikium/output_trl/ProtGPT2_2024-01-31_instruction/dpo", metadata={"help": "the output directory"}
    )
    max_steps: Optional[int] = field(default=4000, metadata={"help": "max number of training steps"})
    save_steps: Optional[int] = field(default=500, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=500, metadata={"help": "the evaluation frequency"})

    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "number of epochs for training"})
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "per device train batch size"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "per device eval batch size"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "warmup steps"})
    logging_dir: Optional[str] = field(default="logs", metadata={"help": "logging directory"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "logging steps"})
    evaluation_strategy: Optional[str] = field(default="steps", metadata={"help": "evaluation strategy"})
    load_best_model_at_end: Optional[bool] = field(default=True, metadata={"help": "load best model at end"})
    save_strategy: Optional[str] = field(default="steps", metadata={"help": "save strategy"})
    metric_for_best_model: str = field(default="eval_loss", metadata={"help": "metric for best model"})
    save_total_limit: Optional[int] = field(default=20, metadata={"help": "save total limit"})
    gradient_accumulation_steps: Optional[int] = field(default=2, metadata={"help": "gradient accumulation"})
    report_to: Optional[str] = field(
        default="mlflow",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    resume_from_checkpoint: Optional[str] = field(default=None, metadata={"help": "checkpoint to restart from"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "whether to use gradient checkpointing"}
    )
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
