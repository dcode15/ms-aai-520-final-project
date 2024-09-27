import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
from inference import Chatbot
from rlaif_queries import queries
from utils import set_seeds
import json
from tqdm import tqdm

set_seeds()
NUM_REPEATS = 3

reward_model = AutoModelForSequenceClassification.from_pretrained(config.TRAINED_REWARD_MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

chatbot = Chatbot()
all_queries = []
responses = []
rewards = []

for i in range(NUM_REPEATS):
    for query in tqdm(queries):
        chatbot.start_conversation()
        response = chatbot.generate_response(query)
        all_queries.append(query)
        responses.append(response)

        inputs = tokenizer(query + "\n" + response, return_tensors="pt")
        with torch.no_grad():
            outputs = reward_model(**inputs)
        rewards.append(outputs.logits[0][1].item())

with open(f"{config.RLAIF_DATA_PATH}/query_data.json", 'w', encoding="utf-8") as file:
    json.dump(all_queries, file, ensure_ascii=False, indent=4)

with open(f"{config.RLAIF_DATA_PATH}/response_data.json", 'w', encoding="utf-8") as file:
    json.dump(responses, file, ensure_ascii=False, indent=4)

with open(f"{config.RLAIF_DATA_PATH}/reward_data.json", 'w', encoding="utf-8") as file:
    json.dump(rewards, file, ensure_ascii=False, indent=4)
