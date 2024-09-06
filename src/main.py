from transformers import BartTokenizer

from src.coversation_dataset import ConversationDataset
from src.preprocessor import Preprocessor

data_path = "../data/"
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

dataset: ConversationDataset = Preprocessor.preprocess_data(data_path, tokenizer)
print(f"Sample item: {dataset[0]}")
