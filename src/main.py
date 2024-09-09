from datasets import DatasetDict
from transformers import GPT2Config, GPT2Tokenizer, Trainer, DataCollatorForLanguageModeling, TrainingArguments

import config
from src.chatbot_model import ChatbotModel
from src.preprocessor import Preprocessor

print(f"Using device: {config.DEVICE}")

tokenizer = GPT2Tokenizer.from_pretrained(config.MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'additional_special_tokens': ['<start_of_turn>', '<end_of_turn>']})

dataset = Preprocessor.preprocess_data(
    config.DATA_PATH,
    tokenizer,
    save_path=config.PREPROCESSED_DATA_PATH,
    subset_proportion=config.DATA_SUBSET_PROPORTION,
    max_length=config.TRAINING_MAX_LENGTH
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

model = ChatbotModel(GPT2Config.from_pretrained(config.MODEL_NAME))
model.resize_token_embeddings(len(tokenizer))

trainer = Trainer(
    model=model,
    args=TrainingArguments(**config.TRAINER_ARGS),
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["validation"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()
trainer.save_model(config.TRAINED_MODEL_PATH)
tokenizer.save_pretrained(config.TRAINED_MODEL_PATH)

eval_results = trainer.evaluate(eval_dataset=dataset_dict["test"])
print(f"Evaluation results: {eval_results}")
