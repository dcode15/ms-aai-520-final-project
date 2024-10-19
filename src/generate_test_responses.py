import json
import os

from datasets import load_from_disk
from tqdm import tqdm

import config
from inference import Chatbot


def extract_prompt(formatted_text):
    start = formatted_text.index("<|im_start|>user\n") + len("<|im_start|>user\n")
    end = formatted_text.index("<|im_end|>", start)
    return formatted_text[start:end].strip()


def evaluate_model():
    dataset = load_from_disk(config.PREPROCESSED_DATA_PATH)
    test_dataset = dataset['test']

    chatbot = Chatbot(config.FINETUNED_MODEL_PATH)

    os.makedirs(config.G_EVAL_DIR, exist_ok=True)
    output_file = os.path.join(config.G_EVAL_DIR, "model_evaluation_results_ft.json")

    results = []
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    start_index = len(results)

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
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Evaluation complete. Results saved to {output_file}")


if __name__ == "__main__":
    evaluate_model()
