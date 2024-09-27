import json
import random
from pathlib import Path
from typing import Optional

import instructor
from openai import OpenAI
from tqdm import tqdm

import config
from inference import Chatbot
from rlaif_queries import queries

llm_client = instructor.from_openai(OpenAI())


def create_rlaif_dataset(num_samples: int = 1000) -> list[dict]:
    data = []
    chatbot = Chatbot()
    for _ in tqdm(range(num_samples), desc="Creating RLAIF dataset"):
        query = random.choice(queries)

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
        **config.RLAIF_LLM_CONFIG,
        messages=[{"role": "user", "content": prompt}],
        response_model=str
    )

    try:
        return int(response.strip())
    except ValueError:
        print(f"Invalid LLM response: {response}")
        return None


if __name__ == "__main__":
    rlaif_data = create_rlaif_dataset()
    Path(config.RLAIF_DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(config.RLAIF_DATA_PATH, 'w', encoding="utf-8") as file:
        json.dump(rlaif_data, file, ensure_ascii=False, indent=4)
