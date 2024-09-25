import json

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig, pipeline
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

import config


def load_rlhf_data() -> Dataset:
    with open(config.RLHF_DATA_PATH, 'r') as file:
        rlhf_data = json.load(file)

    dataset = Dataset.from_list(rlhf_data)
    dataset = dataset.rename_column("prompt", "query")
    dataset = dataset.remove_columns(["chosen", "rejected"])
    return dataset


def tokenize(sample, tokenizer):
    sample["input_ids"] = tokenizer.encode(sample["query"])
    return sample


def setup_model_and_tokenizer():
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.TRAINED_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=quantization_config,
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return model, tokenizer


def setup_reward_model():
    return pipeline(
        "text-classification",
        model=config.TRAINED_REWARD_MODEL_PATH,
        device=config.DEVICE
    )


def setup_ppo_trainer(model, dataset, tokenizer):
    return PPOTrainer(
        model=model,
        config=PPOConfig(**config.PPO_CONFIG),
        dataset=dataset,
        tokenizer=tokenizer,
    )


def run_rlhf_training(ppo_trainer, reward_model, tokenizer, epochs=1):
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        for batch in tqdm(ppo_trainer.dataloader, desc="Processing Batches"):
            query_tensors = batch["input_ids"]
            response_tensors = ppo_trainer.generate(query_tensors, **config.INFERENCE_PARAMS)
            batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            pipe_outputs = reward_model(texts)
            rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

    ppo_trainer.save_pretrained(config.TRAINED_RLHF_MODEL_PATH)


def main():
    model, tokenizer = setup_model_and_tokenizer()
    reward_model = setup_reward_model()

    dataset = load_rlhf_data()
    dataset = dataset.map(lambda sample: tokenize(sample, tokenizer), batched=False)

    ppo_trainer = setup_ppo_trainer(model, dataset, tokenizer)
    run_rlhf_training(ppo_trainer, reward_model, tokenizer)


if __name__ == "__main__":
    main()
