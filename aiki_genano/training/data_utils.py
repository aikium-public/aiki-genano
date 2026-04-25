import pdb
import os
from typing import Dict, Optional

from datasets import Dataset, load_dataset
from tqdm import tqdm

def custom_load_dataset(data_dir):
    # Define the mapping of data files to splits
    data_files = {
        'train': os.path.join(data_dir, 'training.csv'),
        'test': os.path.join(data_dir, 'testing.csv')
    }
    # Load the dataset with the correct file mappings
    data = load_dataset('csv', data_files=data_files)

    return data["train"], data["test"]


def add_newline_every_60_characters(input_string, start_from=0):
    """Convert the sequence to a string like this
    (note we have to introduce new line characters every 60 amino acids,
    following the FASTA file format)."""
    result = ""
    result = input_string[:start_from]
    for i in range(start_from, start_from + len(input_string), 60):
        ind = i - start_from
        result += input_string[ind : ind + 60] + "\n"
    return result[:-1]  # remove final \n character


##################################################
###### SFT DATA
##################################################
def chat_formatting_sft(tokenizer, receptors, binders):
    instructions = []
    for receptor, binder in zip(receptors, binders):
        inst = [
            {"role": "user", "content": add_newline_every_60_characters(receptor)},
            {"role": "assistant", "content": add_newline_every_60_characters(binder)},
        ]
        instructions.append(inst)
    inst_dataset = Dataset.from_dict({"chat": instructions})

    chat_dataset = inst_dataset.map(
        lambda x: {"text": tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=False)},
        keep_in_memory=True,
    )

    return chat_dataset


def create_sft_instructions(tokenizer, data_dir="/data/aikium/protein_peptide/dpo"):
    """Generate train and validation datasets from pre-splitted train/test datasets
       reference: https://huggingface.co/docs/transformers/main/en/chat_templating#templates-for-chat-models

    Args:
        tokenizer (_type_): tokenizer used
        args (_type_): script arguments

    Returns:
        train/valid datasets
    """
    
    train_data, valid_data = custom_load_dataset(data_dir)

    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
    train_dataset = chat_formatting_sft(tokenizer, train_data['peptide'], train_data['protein']) # peptide is the input, protein is the output
    valid_dataset = chat_formatting_sft(tokenizer, valid_data['peptide'], valid_data['protein']) # peptide is the input, protein is the output
    return train_dataset, valid_dataset


##################################################
###### DPO DATA
##################################################


def return_prompt_and_responses_dpo(samples):
    # Match the SFT direction used in this repo:
    #   peptide/target (prompt) -> binder/protein (chosen) vs decoy binder (rejected)
    prompt_input = samples["peptide"]
    chosen_output = samples["protein"]
    rejected_output = samples["decoy"]
    # -- format chatml:
    # "|im_start|>user\n"  + add_newline_every_60_characters(receptor)+ "<|im_end|>\n" + "<|im_start|>assistant\n" + add_newline_every_60_characters(binder) + "<|im_end|>\n"
    prompt = (
        "<|im_start|>user\n"
        + add_newline_every_60_characters(prompt_input)
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n"
    )
    chosen = add_newline_every_60_characters(chosen_output) + "<|im_end|>\n"
    rejected = add_newline_every_60_characters(rejected_output) + "<|im_end|>\n"
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def get_protein_peptide_preference_datasets(
    data_dir: str = "/data/aikium/protein_peptide/dpo",
    filter_to_max_length: bool = False,
    cache_dir: str = None,
    num_proc=24,
    max_length: int = 1024,
) -> Dataset:
    """Load the paired dataset and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }
    """
    # train_dataset = load_dataset(path=data_dir, split='train')
    # test_dataset = load_dataset(path=data_dir, split='test')

    train_dataset, test_dataset = custom_load_dataset(data_dir)

    train_processed = train_dataset.map(
        return_prompt_and_responses_dpo,
        keep_in_memory=True,
        batched=False,
#        num_proc=num_proc,
        remove_columns=train_dataset.column_names,
    )

    test_processed = test_dataset.map(
        return_prompt_and_responses_dpo,
        keep_in_memory=True,
        batched=False,
#        num_proc=num_proc,
        remove_columns=test_dataset.column_names,
    )

    return train_processed, test_processed
