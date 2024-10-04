import json

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig

import config
from utils import set_seeds


def load_reward_model_data() -> Dataset:
    with open(f"{config.REWARD_MODEL_DATA_PATH}/reward_model_data.json", 'r', encoding="utf-8") as file:
        reward_model_data = json.load(file)

    return Dataset.from_list(reward_model_data)


def main():
    set_seeds()
    reward_model_data = load_reward_model_data()

    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_REWARD_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=config.DEVICE,
        attn_implementation="flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_REWARD_MODEL_NAME)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.config.pad_token_id = tokenizer.pad_token_id

    dataset = reward_model_data.train_test_split(test_size=0.2, seed=1)

    trainer = DPOTrainer(
        model=model,
        args=DPOConfig(**config.REWARD_MODEL_TRAINER_ARGS),
        beta=0.1,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        peft_config=LoraConfig(**config.REWARD_MODEL_LORA_ARGS),
    )

    trainer.train()

    trainer.save_model(config.TRAINED_REWARD_MODEL_PATH)


if __name__ == "__main__":
    main()
