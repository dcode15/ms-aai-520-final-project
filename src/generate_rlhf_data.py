import os
import pickle
import random
from pathlib import Path

import instructor
from openai import OpenAI
from tqdm import tqdm

import config
from inference import generate_response
from rlhf_queries import queries

llm_client = instructor.from_openai(OpenAI())


def create_rlhf_dataset(num_samples: int = 250) -> list[tuple]:
    data = []
    for _ in tqdm(range(num_samples), desc="Creating RLHF dataset"):
        query = random.choice(queries)
        response1 = generate_response(query, max_length=config.INFERENCE_MAX_LENGTH, config_override={
            "do_sample": True,
            "temperature": random.uniform(0.7, 1.0),
            "top_k": random.randint(20, 50),
            "top_p": random.uniform(0.9, 1.0),
            "no_repeat_ngram_size": 2
        })
        response2 = generate_response(query, max_length=config.INFERENCE_MAX_LENGTH, config_override={
            "do_sample": True,
            "temperature": random.uniform(0.7, 1.0),
            "top_k": random.randint(20, 50),
            "top_p": random.uniform(0.9, 1.0),
            "no_repeat_ngram_size": 2
        })

        preference = get_claude_preference(query, response1, response2)
        chosen_response = response1 if preference == 1 else response2
        rejected_response = response2 if preference == 1 else response1

        data.append((
            query,
            chosen_response,
            rejected_response,
            1.0,
            -1.0
        ))

    return data


def get_claude_preference(query: str, response1: str, response2: str) -> int:
    prompt = f"""
Given the following query and two possible responses, select the response that is more concise, appropriate, accurate, and engaging. 

Query: {query}

Response 1: {response1}

Response 2: {response2}

Which response is better? Answer with either 1 or 2 and nothing else."""

    response = llm_client.chat.completions.create(
        **config.RLHF_LLM_CONFIG,
        messages=[{"role": "user", "content": prompt}],
        response_model=str
    )
    return int(response.strip())


if __name__ == "__main__":
    if os.path.exists(config.RLHF_DATA_PATH):
        print(f"Loading existing RLHF dataset from {config.RLHF_DATA_PATH}")
        with open(config.RLHF_DATA_PATH, 'rb') as file:
            rlhf_data = pickle.load(file)
    else:
        print("Creating new RLHF dataset...")
        rlhf_data = create_rlhf_dataset()
        Path(config.RLHF_DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(config.RLHF_DATA_PATH, 'wb') as file:
            pickle.dump(rlhf_data, file)
