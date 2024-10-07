import os
from typing import List

import instructor
from openai import OpenAI
from pydantic import BaseModel

import config

llm_client = instructor.from_openai(OpenAI())


class MoviePromptBatch(BaseModel):
    prompts: List[str] = []


prompts_file = "dpo_prompts.py"
prompts = set()
if os.path.exists(prompts_file):
    from dpo_prompts import dpo_prompts

    prompts = set(dpo_prompts)

num_prompts = 10000
batch_size = 100
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
        with open(prompts_file, "w", encoding="utf-8") as f:
            f.write("queries = [\n")
            for prompt in list(prompts):
                f.write(f"    {prompt},\n")
            f.write("]\n")

        written_prompts = len(prompts)
