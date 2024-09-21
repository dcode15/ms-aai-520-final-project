import torch
from peft import TaskType

# Paths
DATA_PATH = "../data/"
PREPROCESSED_DATA_PATH = "../out/preprocessed_dataset"
MODEL_OUTPUT_DIR = "../out/chatbot_model_output"
TRAINED_MODEL_PATH = "../out/trained_chatbot_model"
RLHF_DATA_PATH = "../out/rlhf_dataset/rlhf_data.json"
REWARD_MODEL_OUTPUT_DIR = "../out/reward_model_output"
TRAINED_REWARD_MODEL_PATH = "../out/trained_reward_model"
TRAINED_RLHF_MODEL_PATH = "../out/trained_rlhf_model"

LOGS_DIR = "./logs"

# Model configuration
BASE_MODEL_NAME = "Qwen/Qwen2-1.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SYSTEM_PROMPT = "You are a dialogue partner in a movie. Respond naturally and conversationally."

# Training parameters
TUNING_DATA_SUBSET_PROPORTION = 1
TUNING_LORA_ARGS = {
    "task_type": TaskType.CAUSAL_LM
}
TUNING_TRAINER_ARGS = {
    "output_dir": MODEL_OUTPUT_DIR,
    "num_train_epochs": 2,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "load_best_model_at_end": True,
    "eval_strategy": "steps",
    "dataset_text_field": "text",
    "packing": True,
    "max_seq_length": 512,
    "fp16": True,
    "neftune_noise_alpha": 5,
    "gradient_accumulation_steps": 8
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
BASE_REWARD_MODEL_NAME = "Qwen/Qwen2-0.5B"
RLHF_LLM_CONFIG = {
    "model": "gpt-4o-mini",
    "temperature": 0,
    "max_retries": 3
}
REWARD_MODEL_LORA_ARGS = {
    "task_type": TaskType.CAUSAL_LM
}
REWARD_MODEL_TRAINER_ARGS = {
    "output_dir": REWARD_MODEL_OUTPUT_DIR,
    "num_train_epochs": 5,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "load_best_model_at_end": True,
    "eval_strategy": "steps",
    "remove_unused_columns": False,
    "max_length": TUNING_TRAINER_ARGS["max_seq_length"],
    "fp16": True,
    "gradient_accumulation_steps": 2
}
PPO_CONFIG = {
    "mini_batch_size": 1,
    "batch_size": 1,
    "gradient_accumulation_steps": 1
}
