import json

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig

import config
from utils import set_seeds


def load_dpo_data() -> Dataset:
    with open(f"{config.DPO_DATA_PATH}/dpo_data.json", 'r', encoding="utf-8") as file:
        dpo_data = json.load(file)

    return Dataset.from_list(dpo_data)


def main():
    set_seeds()
    dpo_data = load_dpo_data()

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
    model = PeftModel.from_pretrained(model, config.FINETUNED_MODEL_PATH, is_trainable=True)

    tokenizer = AutoTokenizer.from_pretrained(config.FINETUNED_MODEL_PATH)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.config.pad_token_id = tokenizer.pad_token_id

    dataset = dpo_data.train_test_split(test_size=0.2, seed=1)

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


if __name__ == "__main__":
    main()
