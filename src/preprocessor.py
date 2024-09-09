import os
import random
import re

from datasets import Dataset, load_from_disk
from transformers import GPT2Tokenizer


class Preprocessor:
    @staticmethod
    def _load_cornell_corpus(data_path: str) -> list:
        movie_lines_path = os.path.join(data_path, "movie_lines.txt")
        movie_conversations_path = os.path.join(data_path, "movie_conversations.txt")

        lines_data = {}
        with open(movie_lines_path, "r", encoding="iso-8859-1") as f:
            for line in f:
                parts = line.strip().split(" +++$+++ ")
                if len(parts) == 5:
                    lines_data[parts[0]] = parts[4]

        conversations = []
        with open(movie_conversations_path, "r", encoding="iso-8859-1") as f:
            for line in f:
                parts = line.strip().split(" +++$+++ ")
                if len(parts) == 4:
                    conversation = eval(parts[3])
                    conversations.append(conversation)

        conversation_pairs = []
        for conversation in conversations:
            for i in range(len(conversation) - 1):
                try:
                    context = lines_data[conversation[i]]
                    response = lines_data[conversation[i + 1]]
                    conversation_pairs.append((context, response))
                except KeyError as e:
                    print(f"Unrecognized line key: {e}")

        return conversation_pairs

    @staticmethod
    def _preprocess_text(text: str) -> str:
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.lower()

    @staticmethod
    def preprocess_data(data_path: str, tokenizer: GPT2Tokenizer, max_length,
                        save_path: str, subset_proportion: float = 1.0) -> Dataset:
        if os.path.exists(save_path):
            print(f"Loading preprocessed dataset from {save_path}")
            return load_from_disk(save_path)

        print("Preprocessed dataset not found. Starting preprocessing...")
        conversation_pairs = Preprocessor._load_cornell_corpus(data_path)

        if subset_proportion < 1.0:
            num_pairs = int(len(conversation_pairs) * subset_proportion)
            conversation_pairs = random.sample(conversation_pairs, num_pairs)

        preprocessed_data = []
        for context, response in conversation_pairs:
            context = Preprocessor._preprocess_text(context)
            response = Preprocessor._preprocess_text(response)

            full_text = f"{context}{tokenizer.eos_token}{response}{tokenizer.eos_token}"

            encodings = tokenizer.encode_plus(
                full_text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            preprocessed_data.append({
                "input_ids": encodings["input_ids"].squeeze().tolist(),
                "attention_mask": encodings["attention_mask"].squeeze().tolist()
            })

        dataset = Dataset.from_list(preprocessed_data)

        print(f"Saving preprocessed dataset to {save_path}")
        dataset.save_to_disk(save_path)

        return dataset
