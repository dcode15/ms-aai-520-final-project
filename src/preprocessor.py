import os
import re

import pandas as pd
from transformers import BartTokenizer

from src.coversation_dataset import ConversationDataset


class Preprocessor:

    @staticmethod
    def _load_cornell_corpus(data_path: str) -> pd.DataFrame:
        movie_lines_path = os.path.join(data_path, 'movie_lines.txt')
        movie_conversations_path = os.path.join(data_path, 'movie_conversations.txt')

        lines_data = {}
        with open(movie_lines_path, 'r', encoding='iso-8859-1') as f:
            for line in f:
                parts = line.strip().split(' +++$+++ ')
                if len(parts) == 5:
                    lines_data[parts[0]] = parts[4]

        conversations = []
        with open(movie_conversations_path, 'r', encoding='iso-8859-1') as f:
            for line in f:
                parts = line.strip().split(' +++$+++ ')
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

        return pd.DataFrame(conversation_pairs, columns=['context', 'response'])

    @staticmethod
    def _preprocess_text(text: str) -> str:
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()

    @staticmethod
    def preprocess_data(data_path: str, tokenizer: BartTokenizer, max_length: int = 512) -> ConversationDataset:
        df = Preprocessor._load_cornell_corpus(data_path)
        df['context'] = df['context'].apply(Preprocessor._preprocess_text)
        df['response'] = df['response'].apply(Preprocessor._preprocess_text)

        conversations = list(zip(df['context'], df['response']))
        dataset = ConversationDataset(conversations, tokenizer, max_length)

        return dataset
