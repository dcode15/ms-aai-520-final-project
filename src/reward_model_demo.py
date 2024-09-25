import random

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import config

torch.manual_seed(1)
random.seed(1)
np.random.seed(1)

def load_reward_model():
    model = AutoModelForSequenceClassification.from_pretrained(config.TRAINED_REWARD_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_reward_score(model, tokenizer, prompt, response):
    inputs = tokenizer(prompt + "\n" + response, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits[0][1].item()


def main():
    model, tokenizer = load_reward_model()

    examples = [
        ("How was your day?",
         "It was great! I had a productive meeting at work and then enjoyed a nice dinner with friends."),
        ("What's your favorite movie?",
         "A mammal is a type of furry creature that lays eggs."),
        ("Can you explain quantum computing?",
         "Quantum computing uses quantum bits or qubits, which can exist in multiple states simultaneously, allowing for complex calculations."),
    ]

    print("Testing reward model on examples:\n")
    for prompt, response in examples:
        score = get_reward_score(model, tokenizer, prompt, response)
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print(f"Reward Score: {score:.4f}\n")


if __name__ == "__main__":
    main()
