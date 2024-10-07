import json
import os
import random
from pathlib import Path
from typing import Optional

import instructor
from openai import OpenAI
from tqdm import tqdm

import config
from dpo_prompts import dpo_prompts
from inference import Chatbot

llm_client = instructor.from_openai(OpenAI())
data_file_path = f"{config.DPO_DATA_PATH}/dpo_data.json"


def load_data():
    if os.path.exists(data_file_path):
        with open(data_file_path, 'r', encoding="utf-8") as file:
            return json.load(file)
    else:
        return []


def write_data_file(data):
    with open(data_file_path, 'w', encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def create_dpo_data(num_samples: int = 1500) -> list[dict]:
    data = load_data()
    chatbot = Chatbot()
    for _ in tqdm(range(num_samples), desc="Creating DPO dataset"):
        if len(data) % 50 == 0:
            write_data_file(data)

        query = random.choice(dpo_prompts)

        chatbot.start_conversation()
        response1 = chatbot.generate_response(query, max_length=config.INFERENCE_MAX_LENGTH, config_override={
            "do_sample": True,
            "temperature": random.uniform(0.7, 1.0),
            "top_k": random.randint(20, 50),
            "top_p": random.uniform(0.9, 1.0),
            "no_repeat_ngram_size": 2
        })

        chatbot.start_conversation()
        response2 = chatbot.generate_response(query, max_length=config.INFERENCE_MAX_LENGTH, config_override={
            "do_sample": True,
            "temperature": random.uniform(0.7, 1.0),
            "top_k": random.randint(20, 50),
            "top_p": random.uniform(0.9, 1.0),
            "no_repeat_ngram_size": 2
        })

        preference = get_response_preference(query, response1, response2)
        if preference is not None:
            chosen_response = response1 if preference == 1 else response2
            rejected_response = response2 if preference == 1 else response1

            data.append({
                "prompt": query,
                "chosen": chosen_response,
                "rejected": rejected_response
            })

    return data


def get_response_preference(query: str, response1: str, response2: str) -> Optional[int]:
    prompt = f"""
Given the following query and two possible responses, select the response that is most coherent and works best as dialogue from a movie. 
Answer with either 1 or 2 and nothing else.

Query: {query}

Response 1: {response1}

Response 2: {response2}"""

    response = llm_client.chat.completions.create(
        **config.DPO_LLM_CONFIG,
        messages=[{"role": "user", "content": prompt}],
        response_model=str
    )

    try:
        return int(response.strip())
    except ValueError:
        print(f"Invalid LLM response: {response}")
        return None


if __name__ == "__main__":
    Path(config.DPO_DATA_PATH).mkdir(parents=True, exist_ok=True)
    data = create_dpo_data()
    write_data_file(data)
