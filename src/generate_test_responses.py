import json
import os
from typing import List, Dict

from datasets import load_from_disk, Dataset
from tqdm import tqdm

import config
from inference import Chatbot


def extract_prompt(formatted_text: str) -> str:
    """
    Extracts the user prompt from the formatted conversation text.

    Args:
        formatted_text (str): The formatted conversation text.

    Returns:
        str: The extracted user prompt.
    """
    start = formatted_text.index("<|im_start|>user\n") + len("<|im_start|>user\n")
    end = formatted_text.index("<|im_end|>", start)
    return formatted_text[start:end].strip()


def load_dataset() -> Dataset:
    """
    Loads the preprocessed dataset and return the test split.

    Returns:
        Dataset: The test dataset.
    """
    dataset = load_from_disk(config.PREPROCESSED_DATA_PATH)
    return dataset['test']


def load_existing_results(output_file: str) -> List[Dict[str, str]]:
    """
    Loads existing results from the output file if it exists.

    Args:
        output_file (str): The path to the output JSON file.

    Returns:
        List[Dict[str, str]]: The list of existing results, or an empty list if the file doesn't exist.
    """
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_results(output_file: str, results: List[Dict[str, str]]) -> None:
    """
    Saves the evaluation results to a JSON file.

    Args:
        output_file (str): The path to the output JSON file.
        results (List[Dict[str, str]]): The list of evaluation results.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def generate_responses(chatbot: Chatbot, test_dataset: Dataset, start_index: int) -> List[Dict[str, str]]:
    """
    Generates responses for the test dataset using the chatbot.

    Args:
        chatbot (Chatbot): The initialized chatbot.
        test_dataset (Dataset): The test dataset.
        start_index (int): The index to start generating responses from.

    Returns:
        List[Dict[str, str]]: The list of generated responses.
    """
    results = []
    for i, example in enumerate(tqdm(test_dataset)):
        if i < start_index:
            continue

        prompt = extract_prompt(example['text'])

        chatbot.start_conversation()
        response = chatbot.generate_response(prompt)
        result = {
            "prompt": prompt,
            "model_response": response
        }
        results.append(result)

        if (i + 1) % 100 == 0:
            save_results(config.G_EVAL_DIR, results)

    return results


def main() -> None:
    os.makedirs(config.G_EVAL_DIR, exist_ok=True)
    output_file = config.G_EVAL_DIR

    test_dataset = load_dataset()
    existing_results = load_existing_results(output_file)
    start_index = len(existing_results)

    chatbot = Chatbot(config.FINETUNED_MODEL_PATH)

    new_results = generate_responses(chatbot, test_dataset, start_index)
    all_results = existing_results + new_results

    save_results(output_file, all_results)

    print(f"Evaluation complete. Results saved to {output_file}")


if __name__ == "__main__":
    main()
