import json

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from trl import RewardTrainer, RewardConfig

import config
from utils import set_seeds


def load_reward_model_data() -> Dataset:
    with open(f"{config.REWARD_MODEL_DATA_PATH}/reward_model_data.json", 'r', encoding="utf-8") as file:
        reward_model_data = json.load(file)

    return Dataset.from_list(reward_model_data)


def formatting_func(examples, tokenizer):
    chosen = examples["prompt"] + "\n" + examples["chosen"]
    rejected = examples["prompt"] + "\n" + examples["rejected"]

    tokenizer_args = {
        "padding": "max_length",
        "truncation": True,
        "max_length": config.REWARD_MODEL_TRAINER_ARGS["max_length"],
        "return_tensors": "pt"
    }
    chosen_tokens = tokenizer(chosen, **tokenizer_args)
    rejected_tokens = tokenizer(rejected, **tokenizer_args)

    return {
        "input_ids_chosen": chosen_tokens["input_ids"].squeeze(),
        "attention_mask_chosen": chosen_tokens["attention_mask"].squeeze(),
        "input_ids_rejected": rejected_tokens["input_ids"].squeeze(),
        "attention_mask_rejected": rejected_tokens["attention_mask"].squeeze()
    }


def main():
    set_seeds()
    reward_model_data = load_reward_model_data()

    model = AutoModelForSequenceClassification.from_pretrained(
        config.BASE_REWARD_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_REWARD_MODEL_NAME)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.config.pad_token_id = tokenizer.pad_token_id

    dataset = reward_model_data.map(lambda examples: formatting_func(examples, tokenizer))
    dataset = dataset.train_test_split(test_size=0.2, seed=1)

    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=RewardConfig(**config.REWARD_MODEL_TRAINER_ARGS),
        peft_config=LoraConfig(**config.REWARD_MODEL_LORA_ARGS)
    )

    trainer.train()

    trainer.save_model(config.TRAINED_REWARD_MODEL_PATH)


if __name__ == "__main__":
    main()
