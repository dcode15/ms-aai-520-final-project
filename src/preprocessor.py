import os

from datasets import Dataset, load_from_disk

import config


class Preprocessor:
    @staticmethod
    def _load_cornell_corpus(data_path: str) -> list:
        movie_lines_path = os.path.join(data_path, "movie_lines.txt")
        movie_conversations_path = os.path.join(data_path, "movie_conversations.txt")

        lines = {}
        with open(movie_lines_path, "r") as f:
            for line in f:
                parts = line.strip().split(" +++$+++ ")
                if len(parts) == 5:
                    lines[parts[0]] = parts[4]

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
                    context = lines[conversation[i]]
                    response = lines[conversation[i + 1]]
                    conversation_pairs.append((context, response))
                except KeyError as e:
                    print(f"Unrecognized line key: {e}")

        return conversation_pairs

    @staticmethod
    def _format_conversation(conversation):
        user_prompt = f"<|im_start|>user\n{conversation[0]}<|im_end|>"
        output = f"<|im_start|>assistant\n{conversation[1]}<|im_end|>"
        return f"{user_prompt}\n{output}<|endoftext|>"

    @staticmethod
    def prepare_dataset():
        if os.path.exists(config.PREPROCESSED_DATA_PATH):
            print(f"Loading preprocessed dataset from {config.PREPROCESSED_DATA_PATH}")
            dataset = load_from_disk(config.PREPROCESSED_DATA_PATH)
        else:
            conversations = Preprocessor._load_cornell_corpus(config.DATA_PATH)
            formatted_conversations = [Preprocessor._format_conversation(conversation) for conversation in
                                       conversations]
            dataset = Dataset.from_dict({"text": formatted_conversations})

            print(f"Saving preprocessed dataset to {config.PREPROCESSED_DATA_PATH}")
            dataset.save_to_disk(config.PREPROCESSED_DATA_PATH)

        subset_size = int(len(dataset) * config.DATA_SUBSET_PROPORTION)
        return dataset.select(range(subset_size))
