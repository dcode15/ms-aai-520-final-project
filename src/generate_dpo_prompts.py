import os
from typing import List, Set

import instructor
from openai import OpenAI
from pydantic import BaseModel

import config

llm_client = instructor.from_openai(OpenAI())


class MoviePromptBatch(BaseModel):
    """
    A model representing a batch of movie dialogue prompts for return from the LLM.
    """
    prompts: List[str] = []


def load_existing_prompts() -> Set[str]:
    """
    Loads existing prompts from the dpo_prompts.py file if it exists.

    Returns:
        Set[str]: A set of existing prompts.
    """
    prompts_file = "dpo_prompts.py"
    prompts = set()
    if os.path.exists(prompts_file):
        from dpo_prompts import dpo_prompts
        prompts = set(dpo_prompts)
    return prompts


def generate_prompts(num_prompts: int, batch_size: int) -> Set[str]:
    """
    Generates movie dialogue prompts using the OpenAI API.

    Args:
        num_prompts (int): The total number of prompts to generate.
        batch_size (int): The number of prompts to generate in each batch.

    Returns:
        Set[str]: A set of generated prompts.
    """
    prompts = load_existing_prompts()
    batch_counter = 0
    written_prompts = len(prompts)

    while len(prompts) < num_prompts:
        batch_counter += 1
        print(f"Processing batch {batch_counter}. {len(prompts)} prompts generated so far.")

        try:
            result = llm_client.chat.completions.create(
                **config.DPO_LLM_CONFIG,
                response_model=MoviePromptBatch,
                messages=[
                    {"role": "system",
                     "content": "You are a screenwriter tasked with creating dialogue prompts for various movie genres."},
                    {"role": "user",
                     "content": f"""
                     Generate {batch_size} pieces of dialogue using the following criteria: 
                     1. Use a variety of styles and genres. 
                     2. Vary lengths from one to three sentences. 
                     3. Each prompt should feel natural and conversational, as if it's part of a movie script.
                     4. Focus primarily on 'realistic' lines rather than highly dramatized one. However, do include occasional dramatic lines.
                     4. Do not repeat prompts."""}
                ]
            )

            for prompt in result.prompts:
                prompts.add(prompt)
                if len(prompts) >= num_prompts:
                    break
        except Exception as e:
            print(f"Error generating prompts: {e}")

        if len(prompts) > (written_prompts + 100):
            write_prompts_to_file(prompts)
            written_prompts = len(prompts)

    return prompts


def write_prompts_to_file(prompts: Set[str]) -> None:
    """
    Writes the generated prompts to a file.

    Args:
        prompts (Set[str]): The set of prompts to write to the file.
    """
    with open("dpo_prompts.py", "w", encoding="utf-8") as f:
        f.write("queries = [\n")
        for prompt in list(prompts):
            f.write(f"    {prompt},\n")
        f.write("]\n")


def main() -> None:
    num_prompts = 10000
    batch_size = 100
    generate_prompts(num_prompts, batch_size)


if __name__ == "__main__":
    main()
