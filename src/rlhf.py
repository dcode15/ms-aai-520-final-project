import os
import random
from typing import List, Tuple

import instructor
from datasets import Dataset
from openai import OpenAI
from tqdm import tqdm

import config
from inference import generate_response
from rlhf_queries import queries

llm_client = instructor.from_openai(OpenAI())


def create_rlhf_dataset(num_samples: int = 5) -> Dataset:
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

        data.append({
            "query": query,
            "response1": response1,
            "response2": response2,
        })

    dataset = Dataset.from_dict({
        "query": [item["query"] for item in data],
        "response1": [item["response1"] for item in data],
        "response2": [item["response2"] for item in data],
    })

    dataset.save_to_disk(config.RLHF_DATASET_PATH)
    return dataset


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


def prepare_rlhf_data(dataset: Dataset) -> List[Tuple[str, str, str, float, float]]:
    rlhf_data = []
    for item in tqdm(dataset, desc="Preparing RLHF data"):
        preference = get_claude_preference(item["query"], item["response1"], item["response2"])

        chosen = item["response1"] if preference == 1 else item["response2"]
        rejected = item["response2"] if preference == 1 else item["response1"]

        rlhf_data.append((
            item["query"],
            chosen,
            rejected,
            1.0,
            -1.0
        ))

    return rlhf_data


if __name__ == "__main__":
    if os.path.exists(config.RLHF_DATASET_PATH):
        print(f"Loading existing RLHF dataset from {config.RLHF_DATASET_PATH}")
        rlhf_dataset = Dataset.load_from_disk(config.RLHF_DATASET_PATH)
    else:
        print("Creating new RLHF dataset...")
        rlhf_dataset = create_rlhf_dataset()

    print("Getting Claude preferences...")
    rlhf_data = prepare_rlhf_data(rlhf_dataset)
