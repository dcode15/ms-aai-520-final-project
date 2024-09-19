import json

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import pipeline
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

import config


def load_rlhf_data() -> Dataset:
    with open(config.RLHF_DATA_PATH, 'r') as file:
        rlhf_data = json.load(file)

    dataset = Dataset.from_list(rlhf_data)
    dataset = dataset.rename_column("prompt", "query")
    dataset = dataset.remove_columns(["chosen", "rejected"])
    return dataset


model = AutoModelForCausalLMWithValueHead.from_pretrained(config.TRAINED_MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(config.TRAINED_MODEL_PATH)

tokenizer.pad_token = tokenizer.eos_token

reward_model = pipeline("text-classification", model=config.TRAINED_REWARD_MODEL_PATH, device=config.DEVICE)


def tokenize(sample):
    sample["input_ids"] = tokenizer.encode(
        sample["query"],
        padding="max_length",
        truncation=True,
        max_length=config.REWARD_MODEL_TRAINER_ARGS["max_length"],
    )
    return sample


dataset = load_rlhf_data()
dataset = dataset.map(tokenize, batched=False)

ppo_trainer = PPOTrainer(
    model=model,
    config=PPOConfig(**config.PPO_CONFIG),
    dataset=dataset,
    tokenizer=tokenizer,
)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}

epochs = 10
for epoch in tqdm(range(epochs), "epoch: "):
    for batch in tqdm(ppo_trainer.dataloader):
        query_tensors = batch["input_ids"]

        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = reward_model(texts)
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

ppo_trainer.save_pretrained(config.TRAINED_RLHF_MODEL_PATH)
