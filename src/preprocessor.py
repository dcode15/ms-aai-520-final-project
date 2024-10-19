import os
from typing import List, Tuple

from datasets import Dataset, load_from_disk, DatasetDict

import config


class Preprocessor:
    @staticmethod
    def _load_cornell_corpus(data_path: str) -> List[Tuple[str, str]]:
        """
        Loads the Cornell Movie-Dialog Corpus into context/response pairs.

        Args:
            data_path (str): Path to the directory containing the corpus files.

        Returns:
            List[Tuple[str, str]]: A list of conversation pairs (context, response).
        """
        movie_lines_path = os.path.join(data_path, "movie_lines.txt")
        movie_conversations_path = os.path.join(data_path, "movie_conversations.txt")

        lines = {}
        with open(movie_lines_path, "r", encoding="iso-8859-1") as f:
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
    def _format_conversation(conversation: Tuple[str, str]) -> str:
        """
        Formats a conversation pair into the required format for the Qwen 2.5 model.

        Args:
            conversation (Tuple[str, str]): A tuple containing (context, response).

        Returns:
            str: Formatted conversation string.
        """
        user_prompt = f"<|im_start|>user\n{conversation[0]}<|im_end|>"
        output = f"<|im_start|>assistant\n{conversation[1]}<|im_end|>"
        return f"{user_prompt}\n{output}<|endoftext|>"

    @staticmethod
    def prepare_dataset() -> DatasetDict:
        """
        Prepares the dataset by loading or creating it from the Cornell Movie-Dialog Corpus.

        Returns:
            DatasetDict: A dataset dictionary containing 'train', 'validation', and 'test' splits.
        """
        if os.path.exists(config.PREPROCESSED_DATA_PATH):
            print(f"Loading preprocessed dataset from {config.PREPROCESSED_DATA_PATH}")
            dataset = load_from_disk(config.PREPROCESSED_DATA_PATH)
        else:
            conversations = Preprocessor._load_cornell_corpus(config.DATA_PATH)
            formatted_conversations = [Preprocessor._format_conversation(conversation) for conversation in
                                       conversations]
            dataset = Dataset.from_dict({"text": formatted_conversations})

            train_val_test = dataset.train_test_split(test_size=0.2, seed=1)
            train_val = train_val_test['train'].train_test_split(test_size=0.2, seed=1)

            dataset = DatasetDict({
                'train': train_val['train'],
                'validation': train_val['test'],
                'test': train_val_test['test']
            })

            print(f"Saving preprocessed dataset to {config.PREPROCESSED_DATA_PATH}")
            dataset.save_to_disk(config.PREPROCESSED_DATA_PATH)

        return dataset
