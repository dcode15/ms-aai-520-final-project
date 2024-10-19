import json
import os
import time
from typing import Dict, List, Any

from openai import OpenAI

import config

client = OpenAI()
criteria = ["coherence", "consistency", "fluency", "relevance"]


def prepare_batch_file(data: List[Dict[str, Any]], criteria_type: str) -> str:
    """
    Prepares a batch file request for scoring.

    Args:
        data (List[Dict[str, Any]]): The responses to be scored.
        criteria_type (str): The name of the criteria being evaluated.

    Returns:
        str: The path to the created batch input file.
    """
    batch_requests = []
    for i, record in enumerate(data):
        request = {
            "custom_id": f"{criteria_type}_{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": config.SCORING_PROMPTS[criteria_type]},
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

    batch_file_path = f"{config.G_EVAL_DIR}/batch_input_{criteria_type}.jsonl"
    with open(batch_file_path, 'w') as f:
        for request in batch_requests:
            f.write(request + '\n')

    return batch_file_path


def calculate_adjusted_score(probabilities: List[float]) -> float:
    """
    Calculates the adjusted score based on token probabilities.

    Args:
        probabilities (List[float]): The token probabilities for each score.

    Returns:
        float: The calculated adjusted score.
    """
    return sum(prob * score for score, prob in zip(range(1, 6), probabilities))


def submit_all_batches(all_data: Dict[str, List[Dict[str, Any]]], criteria: List[str]) -> Dict[tuple, str]:
    """
    Submits all batches for scoring.

    Args:
        all_data (Dict[str, List[Dict[str, Any]]]): The responses to be scored for each model.
        criteria (List[str]): The list of criteria to be evaluated.

    Returns:
        Dict[tuple, str]: A dictionary mapping (model, criteria_type) to batch job IDs.
    """
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


def wait_for_batches(batch_jobs: Dict[tuple, str]) -> None:
    """
    Waits for all batch jobs to complete.

    Args:
        batch_jobs (Dict[tuple, str]): A dictionary of batch jobs to wait for.
    """
    completed_jobs = set()
    while len(completed_jobs) < len(batch_jobs):
        for (model, criteria_type), batch_id in batch_jobs.items():
            if (model, criteria_type) not in completed_jobs:
                batch_status = client.batches.retrieve(batch_id)
                if batch_status.status == "completed":
                    completed_jobs.add((model, criteria_type))
                    print(f"Batch for {model} - {criteria_type} completed")
        time.sleep(60)


def load_existing_results(model: str) -> List[Dict[str, Any]]:
    """
    Loads existing results for a given model.

    Args:
        model (str): The model name.

    Returns:
        List[Dict[str, Any]]: The list of existing results.
    """
    output_file = f"{config.G_EVAL_DIR}/model_evaluation_results_{model}_scored.json"
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def process_batch_results(all_data: Dict[str, List[Dict[str, Any]]], batch_jobs: Dict[tuple, str],
                          existing_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Processes the results of all batch jobs.

    Args:
        all_data (Dict[str, List[Dict[str, Any]]]): The original responses for each model.
        batch_jobs (Dict[tuple, str]): The batch jobs to process.
        existing_results (Dict[str, List[Dict[str, Any]]]): Existing results for each model.

    Returns:
        Dict[str, List[Dict[str, Any]]]: Updated results including new scores.
    """
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


def load_data(model: str) -> List[Dict[str, Any]]:
    """
    Loads data for a given model.

    Args:
        model (str): The model name.

    Returns:
        List[Dict[str, Any]]: The loaded data for the model.
    """
    file_path = f"{config.G_EVAL_DIR}/model_evaluation_results_{model}.json"
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)[:2000]


def save_results(updated_results: Dict[str, List[Dict[str, Any]]]) -> None:
    """
    Saves the updated results for each model.

    Args:
        updated_results (Dict[str, List[Dict[str, Any]]]): The updated results for each model.
    """
    for model, data in updated_results.items():
        output_file = f"{config.G_EVAL_DIR}/model_evaluation_results_{model}_scored.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    data_files = {
        'dpo': 'model_evaluation_results.json',
        'ft': 'model_evaluation_results_ft.json',
        'base': 'model_evaluation_results_base.json'
    }

    all_data = {model: load_data(model) for model in data_files.keys()}
    existing_results = {model: load_existing_results(model) for model in data_files.keys()}
    batch_jobs = submit_all_batches(all_data, criteria)
    wait_for_batches(batch_jobs)

    updated_results = process_batch_results(all_data, batch_jobs, existing_results)
    save_results(updated_results)


if __name__ == "__main__":
    main()
