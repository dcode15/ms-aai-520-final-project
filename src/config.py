import torch
from peft import TaskType

# Paths
DATA_PATH = "../data/"
PREPROCESSED_DATA_PATH = "./preprocessed_dataset"
MODEL_OUTPUT_DIR = "./chatbot_output"
TRAINED_MODEL_PATH = "./chatbot_trained_model"
LOGS_DIR = "./logs"

# Model configuration
MODEL_NAME = "Qwen/Qwen2-0.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SYSTEM_PROMPT = "You are a friendly conversation partner. Match the user's conversational tone for all responses."

# Training parameters
DATA_SUBSET_PROPORTION = 1
LORA_ARGS = {
    "task_type": TaskType.CAUSAL_LM
}
TRAINER_ARGS = {
    "output_dir": MODEL_OUTPUT_DIR,
    "num_train_epochs": 2,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "load_best_model_at_end": True,
    "eval_strategy": "steps",
    "dataset_text_field": "text",
    "packing": True,
    "max_seq_length": 512
}

# Inference parameters
USE_BASE_MODEL = False
INFERENCE_MAX_LENGTH = 64
INFERENCE_PARAMS = {
    "no_repeat_ngram_size": 2,
    "temperature": 0.7,
    "do_sample": True
}

# RLHF parameters
RLHF_LLM_CONFIG = {
    "model": "gpt-4o-mini",
    "temperature": 0,
    "max_retries": 3
}
RLHF_DATASET_PATH = "./rlhf_dataset"
RLHF_MODEL_OUTPUT_DIR = "./rlhf_trained_model"
