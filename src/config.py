import torch
from peft import TaskType

# Paths
DATA_PATH = "../data/"
PREPROCESSED_DATA_PATH = "../out/preprocessed_dataset"
MODEL_OUTPUT_DIR = "../out/chatbot_model_output"
TRAINED_MODEL_PATH = "../out/trained_chatbot_model"
RLHF_DATA_PATH = "../out/rlhf_dataset/rlhf_data.pkl"
RLHF_MODEL_OUTPUT_DIR = "../out/reward_model_output"
RLHF_TRAINED_MODEL_PATH = "../out/trained_reward_model"

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
    "max_seq_length": 512,
    "neftune_noise_alpha": 5
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
RLHF_TRAINER_ARGS = {
    "output_dir": RLHF_MODEL_OUTPUT_DIR,
    "num_train_epochs": 2,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "load_best_model_at_end": True,
    "eval_strategy": "steps",
    "remove_unused_columns": False,
    "max_length": TRAINER_ARGS["max_seq_length"]
}
