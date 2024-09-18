import pickle

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import RewardTrainer, RewardConfig

import config


def load_rlhf_data() -> Dataset:
    with open(config.RLHF_DATA_PATH, 'rb') as file:
        rlhf_data = pickle.load(file)

    return Dataset.from_dict({
        'prompt': [item[0] for item in rlhf_data],
        'chosen': [item[1] for item in rlhf_data],
        'rejected': [item[2] for item in rlhf_data]
    })


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
    rlhf_data = load_rlhf_data()

    model = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME, torch_dtype=torch.float16)
    model = get_peft_model(model, LoraConfig(**config.REWARD_MODEL_LORA_ARGS))

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    dataset = rlhf_data.map(lambda examples: formatting_func(examples, tokenizer))
    dataset = dataset.train_test_split(test_size=0.2, seed=1)

    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=RewardConfig(**config.REWARD_MODEL_TRAINER_ARGS),
    )

    trainer.train()

    model.save_pretrained(config.TRAINED_REWARD_MODEL_PATH)


if __name__ == "__main__":
    main()
