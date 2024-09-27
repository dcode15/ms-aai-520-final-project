import json

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

import config
from utils import set_seeds

set_seeds()

with open(f"{config.RLAIF_DATA_PATH}/query_data.json", "r") as file:
    query_data = json.load(file)

with open(f"{config.RLAIF_DATA_PATH}/response_data.json", "r") as file:
    response_data = json.load(file)

with open(f"{config.RLAIF_DATA_PATH}/reward_data.json", "r") as file:
    reward_data = json.load(file)

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.FINETUNED_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=quantization_config,
    attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

ppo_trainer = PPOTrainer(
    model=model,
    config=PPOConfig(**config.PPO_CONFIG),
    tokenizer=tokenizer,
)

epochs = 1
num_records = len(query_data)
for epoch in tqdm(range(config.RLAIF_EPOCHS), desc="Training Epochs"):
    for index in tqdm(range(0, num_records, config.PPO_CONFIG["batch_size"]), desc="Processing Batches"):
        batch_queries = query_data[index:min(index + config.PPO_CONFIG["batch_size"], num_records)]
        query_tensors = tokenizer.batch_encode_plus(batch_queries, return_tensors="pt", padding=True, truncation=True)[
            'input_ids']
        query_tensors = [tensor for tensor in query_tensors]

        batch_responses = response_data[index:min(index + config.PPO_CONFIG["batch_size"], num_records)]
        response_tensors = \
        tokenizer.batch_encode_plus(batch_responses, return_tensors="pt", padding=True, truncation=True)['input_ids']
        response_tensors = [tensor for tensor in response_tensors]

        batch_rewards = reward_data[index:min(index + config.PPO_CONFIG["batch_size"], num_records)]
        batch_rewards = [torch.tensor([reward]) for reward in batch_rewards]

        stats = ppo_trainer.step(query_tensors, response_tensors, batch_rewards)
        ppo_trainer.log_stats(stats, {"query": batch_queries, "response": batch_responses}, batch_rewards)

ppo_trainer.save_pretrained(config.TRAINED_RLAIF_MODEL_PATH)
