import json
import os
import time

from openai import OpenAI

import config

client = OpenAI()
criteria = ["coherence", "consistency", "fluency", "relevance"]
prompts = {
    "coherence": """
        You will be given a line of movie dialogue and an AI-generated response. Your task is to rate the AI's response on coherence.
        
        Evaluation Criteria:
        Coherence (1-5) - The degree to which the AI's response logically connects to the given dialogue line and maintains a plausible flow of conversation within a movie context. A score of 1 indicates a completely incoherent response that doesn't fit the scene at all, while a score of 5 indicates a perfectly coherent response that naturally continues the dialogue in a way that could believably appear in a movie script.
        
        Evaluation Steps:
        1. Read the given movie dialogue line carefully, considering its potential context, tone, and implied setting.
        2. Read the AI's response and assess how well it follows from the given line in a cinematic context.
        3. Consider whether the response maintains the implied tone, setting, and character dynamics of the original line.
        4. Evaluate how well the response could continue or advance a potential movie scene.
        5. Assess whether the response introduces any abrupt or illogical shifts that would be jarring in a film dialogue.
        6. Assign a score from 1 to 5 based on the overall coherence of the AI's response in the context of movie dialogue.
    """,
    "consistency": """
        You will be given a line of movie dialogue and an AI-generated response. Your task is to rate the AI's response on consistency.

        Evaluation Criteria:
        Consistency (1-5) - The degree to which the AI's response maintains consistent information, tone, and character voice with the given dialogue line. A score of 1 indicates highly inconsistent responses with clear contradictions or tonal mismatches, while a score of 5 indicates perfectly consistent responses that maintain the established context and character voice.
        
        Evaluation Steps:
        1. Carefully read the given movie dialogue line, noting any implied information about the character, setting, or situation.
        2. Read the AI's response and check if it's consistent with the information, tone, and character voice implied by the original line.
        3. Look for any contradictions in facts, emotions, or character traits between the original line and the response.
        4. Assess whether the response maintains a consistent level of formality, emotion, or genre-appropriate language.
        5. Check if the response's tone and style are consistent with what one would expect in a movie dialogue continuation.
        6. Assign a score from 1 to 5 based on the overall consistency of the AI's response with the given movie dialogue line.
    """,
    "fluency": """
        You will be given a line of movie dialogue and an AI-generated response. Your task is to rate the AI's response on fluency.

        Evaluation Criteria:
        Fluency (1-5) - The quality of the AI's language in terms of grammar, vocabulary, and natural flow, specifically in the context of movie dialogue. A score of 1 indicates poor fluency with many errors and unnatural language that would be jarring in a film, while a score of 5 indicates perfect fluency that sounds natural and believable as movie dialogue.
        
        Evaluation Steps:
        1. Read the AI's response carefully, focusing on the language quality in the context of movie dialogue.
        2. Check for any grammatical errors, including issues with verb tenses, subject-verb agreement, and sentence structure.
        3. Assess the vocabulary use, looking for appropriate word choice and variety that fits well in a movie script.
        4. Evaluate the natural flow of language, checking if it sounds like realistic spoken dialogue.
        5. Look for any awkward phrasings or unnatural expressions that would sound out of place in a film.
        6. Consider the appropriate use of idioms, colloquialisms, or character-specific language that enhances the dialogue's authenticity.
        7. Assign a score from 1 to 5 based on the overall fluency of the AI's response as movie dialogue.
    """,
    "relevance": """
        You will be given a line of movie dialogue and an AI-generated response. Your task is to rate the AI's response on relevance.

        Evaluation Criteria:
        Relevance (1-5) - The degree to which the AI's response appropriately addresses or follows up on the given dialogue line in a way that makes sense for a movie scene. A score of 1 indicates a completely irrelevant response that doesn't fit the context of the dialogue at all, while a score of 5 indicates a highly relevant response that perfectly continues or responds to the given line in a cinematically appropriate way.
        
        Evaluation Steps:
        1. Carefully read the given movie dialogue line, identifying the main points, emotions, or subtext it conveys.
        2. Assess how directly the AI's response addresses or builds upon the given line in a way that makes sense for a movie scene.
        3. Check if the response acknowledges all important aspects of the original line, or if it misses any crucial points.
        4. Evaluate whether the response contributes to advancing a potential scene or character development in a relevant way.
        5. Assess if the response stays on topic or if it introduces irrelevant information that doesn't fit the implied movie context.
        6. Consider the depth and specificity of the AI's response in relation to the given line and its potential place in a larger movie narrative.
        7. Assign a score from 1 to 5 based on the overall relevance of the AI's response to the given movie dialogue line.
    """,
}


def prepare_batch_file(data, criteria_type):
    batch_requests = []
    for i, record in enumerate(data):
        request = {
            "custom_id": f"{criteria_type}_{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": prompts[criteria_type]},
                    {"role": "user", "content": json.dumps(record)}
                ],
                "max_completion_tokens": 10,
                "logprobs": True,
                "top_logprobs": 10,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response_score",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "score": {"type": "integer"},
                            },
                            "required": ["score"],
                            "additionalProperties": False
                        }
                    }
                }
            }
        }
        batch_requests.append(json.dumps(request))

    batch_file_path = f"{config.LOGS_DIR}/batch_input_{criteria_type}.jsonl"
    with open(batch_file_path, 'w') as f:
        for request in batch_requests:
            f.write(request + '\n')

    return batch_file_path


def calculate_adjusted_score(probabilities):
    return sum(prob * score for score, prob in zip(range(1, 6), probabilities))


def submit_all_batches(all_data, criteria):
    batch_jobs = {}
    for criteria_type in criteria:
        for model in all_data.keys():
            batch_file_path = prepare_batch_file(all_data[model], criteria_type)
            with open(batch_file_path, 'rb') as f:
                file = client.files.create(file=f, purpose="batch")

            batch = client.batches.create(
                input_file_id=file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            batch_jobs[(model, criteria_type)] = batch.id

    return batch_jobs


def wait_for_batches(batch_jobs):
    completed_jobs = set()
    while len(completed_jobs) < len(batch_jobs):
        for (model, criteria_type), batch_id in batch_jobs.items():
            if (model, criteria_type) not in completed_jobs:
                batch_status = client.batches.retrieve(batch_id)
                if batch_status.status == "completed":
                    completed_jobs.add((model, criteria_type))
                    print(f"Batch for {model} - {criteria_type} completed")
        time.sleep(60)


def load_existing_results(model):
    output_file = f"{config.LOGS_DIR}/model_evaluation_results_{model}_scored.json"
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def process_batch_results(all_data, batch_jobs, existing_results):
    for (model, criteria_type), batch_id in batch_jobs.items():
        batch_status = client.batches.retrieve(batch_id)
        output_file = client.files.retrieve_content(batch_status.output_file_id)
        batch_results = [json.loads(line) for line in output_file.split('\n') if line]

        for record, result in zip(all_data[model], batch_results):
            try:
                response_body = result['response']['body']
                choices = response_body['choices']
                if choices:
                    top_choice = choices[0]
                    score = int(json.loads(top_choice['message']['content'])['score'])
                    score_probabilities = [None] * 5
                    token_probabilities = top_choice['logprobs']['content']
                    for token_probability in token_probabilities:
                        if token_probability["token"] == str(score):
                            for item in token_probability['top_logprobs']:
                                if item['token'] in ["1", "2", "3", "4", "5"]:
                                    score_probabilities[int(item['token']) - 1] = item['logprob']

                    adjusted_score = calculate_adjusted_score(score_probabilities)
                    record[f"{criteria_type}_adjusted"] = adjusted_score
                    record[criteria_type] = score
                else:
                    record[criteria_type] = None
                    record[f"{criteria_type}_adjusted"] = None
            except Exception as e:
                print(f"Error processing result: \n\n{e}")

        existing_results[model].extend(all_data[model])

    return existing_results


# Load data
data_files = {
    'dpo': f"{config.LOGS_DIR}/model_evaluation_results.json",
    'ft': f"{config.LOGS_DIR}/model_evaluation_results_ft.json",
    'base': f"{config.LOGS_DIR}/model_evaluation_results_base.json"
}

all_data = {}
existing_results = {}
for model, file_path in data_files.items():
    with open(file_path, 'r', encoding='utf-8') as f:
        all_data[model] = json.load(f)[750:1000]
    existing_results[model] = load_existing_results(model)

print("Submitting all batches...")
batch_jobs = submit_all_batches(all_data, criteria)

print("Waiting for batches to complete...")
wait_for_batches(batch_jobs)

print("Processing batch results...")
updated_results = process_batch_results(all_data, batch_jobs, existing_results)

for model, data in updated_results.items():
    output_file = f"{config.LOGS_DIR}/model_evaluation_results_{model}_scored.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

print("Evaluation complete. Results appended and saved.")
