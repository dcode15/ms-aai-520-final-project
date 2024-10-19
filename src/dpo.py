import json
from typing import Dict

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback, BitsAndBytesConfig, \
    PreTrainedTokenizerBase
from trl import DPOTrainer, DPOConfig

import config
from utils import set_seeds


def load_dpo_data() -> Dataset:
    """
    Loads DPO data from a JSON file and convert it to a Dataset object.

    Returns:
        Dataset: A Dataset object containing the DPO data.
    """
    with open(f"{config.DPO_DATA_PATH}/dpo_data.json", 'r', encoding="utf-8") as file:
        dpo_data = json.load(file)

    return Dataset.from_list(dpo_data)


def init_model() -> PeftModel:
    """
    Initializes the base model with quantization configuration.

    Returns:
        PeftModel: The initialized base model.
    """
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_NAME,
        quantization_config=quantization_config,
        attn_implementation="flash_attention_2",
    )
    return PeftModel.from_pretrained(model, config.FINETUNED_MODEL_PATH, is_trainable=True)


def init_tokenizer() -> PreTrainedTokenizerBase:
    """
    Initializes the tokenizer for the model.

    Returns:
        PreTrainedTokenizerBase: The initialized tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.FINETUNED_MODEL_PATH)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def prepare_dataset(dpo_data: Dataset) -> Dict[str, Dataset]:
    """
    Prepares the dataset by splitting it into train and test sets.

    Args:
        dpo_data (Dataset): The full DPO dataset.

    Returns:
        Dict[str, Dataset]: A dictionary containing 'train' and 'test' datasets.
    """
    return dpo_data.train_test_split(test_size=0.2, seed=1)


def train_dpo_model(model: PeftModel, tokenizer: PreTrainedTokenizerBase, dataset: Dict[str, Dataset]) -> None:
    """
    Trains the model using DPO (Direct Preference Optimization).

    Args:
        model (PeftModel): The initialized model.
        tokenizer (PreTrainedTokenizerBase): The initialized tokenizer.
        dataset (Dict[str, Dataset]): The prepared dataset containing 'train' and 'test' splits.
    """
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2)

    trainer = DPOTrainer(
        model=model,
        args=DPOConfig(**config.DPO_TRAINER_ARGS),
        beta=config.DPO_BETA,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        callbacks=[early_stopping_callback],
    )

    trainer.train()
    trainer.save_model(config.TRAINED_DPO_MODEL_PATH)


def main() -> None:
    set_seeds()
    dpo_data = load_dpo_data()

    model = init_model()
    tokenizer = init_tokenizer()
    model.config.pad_token_id = tokenizer.pad_token_id

    dataset = prepare_dataset(dpo_data)

    train_dpo_model(model, tokenizer, dataset)


if __name__ == "__main__":
    main()
