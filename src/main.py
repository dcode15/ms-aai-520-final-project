import torch
from datasets import DatasetDict
from transformers import GPT2Config, GPT2Tokenizer, Trainer, DataCollatorForLanguageModeling

from src.chatbot_model import ChatbotModel
from src.preprocessor import Preprocessor
from src.trainer_config import get_training_args

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_name = "microsoft/DialoGPT-small"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

data_path = "../data/"
preprocessed_data_path = "./preprocessed_dataset"
dataset = Preprocessor.preprocess_data(
    data_path,
    tokenizer,
    save_path=preprocessed_data_path,
    subset_proportion=0.0001
)

train_testvalid = dataset.train_test_split(test_size=0.3)
test_valid = train_testvalid["test"].train_test_split(test_size=0.5)
dataset_dict = DatasetDict({
    "train": train_testvalid["train"],
    "validation": test_valid["train"],
    "test": test_valid["test"]
})

print(f"Train set size: {len(dataset_dict["train"])}")
print(f"Validation set size: {len(dataset_dict["validation"])}")
print(f"Test set size: {len(dataset_dict["test"])}")

config = GPT2Config.from_pretrained(model_name)
model = ChatbotModel(config)

training_args = get_training_args("./chatbot_output")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["validation"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()
trainer.save_model("./chatbot_trained_model")
tokenizer.save_pretrained("./chatbot_trained_model")

eval_results = trainer.evaluate(eval_dataset=dataset_dict["test"])
print(f"Evaluation results: {eval_results}")
