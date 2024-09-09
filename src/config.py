import torch

# Paths
DATA_PATH = "../data/"
PREPROCESSED_DATA_PATH = "./preprocessed_dataset"
MODEL_OUTPUT_DIR = "./chatbot_output"
TRAINED_MODEL_PATH = "./chatbot_trained_model"
LOGS_DIR = "./logs"

# Model configuration
MODEL_NAME = "microsoft/DialoGPT-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training parameters
TRAINING_MAX_LENGTH = 512
DATA_SUBSET_PROPORTION = 0.0001
TRAINER_ARGS = {
    "output_dir": MODEL_OUTPUT_DIR,
    "num_train_epochs": 2,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "logging_dir": "./logs",
    "logging_steps": 10,
    "eval_strategy": "steps",
    "eval_steps": 500,
    "save_steps": 1000,
    "load_best_model_at_end": True,
}

# Inference parameters
INFERENCE_MAX_LENGTH = 20
INFERENCE_PARAMS = {
    "no_repeat_ngram_size": 2,
    "temperature": 0.7,
    "do_sample": True
}
