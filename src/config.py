import torch
from peft import TaskType

# Paths
DATA_PATH = "../data/"
PREPROCESSED_DATA_PATH = "./out/preprocessed_dataset"

FINETUNING_OUTPUT_DIR = "./out/chatbot_model_output"
FINETUNED_MODEL_PATH = "./finetuned_chatbot_model"

DPO_DATA_PATH = "./out/dpo_data"
DPO_OUTPUT_DIR = "./out/dpo_output"
TRAINED_DPO_MODEL_PATH = "./out/trained_dpo_model"

LOGS_DIR = "./logs"

# Model configuration
BASE_MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SYSTEM_PROMPT = "You are a dialogue partner in a movie. Respond naturally and conversationally."

# Training parameters
TUNING_DATA_SUBSET_PROPORTION = 1
TUNING_LORA_ARGS = {
    "task_type": TaskType.CAUSAL_LM,
    "r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "target_modules": ["q_proj", "v_proj"],
    "inference_mode": False,
}
TUNING_TRAINER_ARGS = {
    "output_dir": FINETUNING_OUTPUT_DIR,
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
    "gradient_accumulation_steps": 16,
    "optim": "adamw_bnb_8bit"
}

# Inference parameters
USE_BASE_MODEL = False
INFERENCE_MAX_LENGTH = 48
INFERENCE_PARAMS = {
    "no_repeat_ngram_size": 2,
    "temperature": 0.9,
    "do_sample": True,
    "top_k": 25,
    "top_p": 0.95,
    "num_beams": 3,
    "remove_invalid_values": True
}

# DPO parameters
DPO_LLM_CONFIG = {
    "model": "gpt-4o-mini",
    "temperature": 0,
    "max_retries": 3
}

DPO_LORA_ARGS = {
    "task_type": TaskType.SEQ_CLS,
    "r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "inference_mode": False,
}
DPO_TRAINER_ARGS = {
    "output_dir": DPO_OUTPUT_DIR,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "load_best_model_at_end": True,
    "eval_strategy": "steps",
    "remove_unused_columns": False,
    "max_length": TUNING_TRAINER_ARGS["max_seq_length"],
    "fp16": True,
    "gradient_accumulation_steps": 8,
    "max_prompt_length": 256,
}
