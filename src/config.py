import torch
from peft import TaskType

# Paths
DATA_PATH = "../data/"
PREPROCESSED_DATA_PATH = "./out/preprocessed_dataset"

FINETUNING_OUTPUT_DIR = "./out/chatbot_model_output"
FINETUNED_MODEL_PATH = "./finetuned_chatbot_model"

DPO_DATA_PATH = "./out/dpo_data"
DPO_OUTPUT_DIR = "./out/dpo_output"
TRAINED_DPO_MODEL_PATH = "./trained_dpo_model"
G_EVAL_DIR = "./out/g_eval"

# Model configuration
BASE_MODEL_NAME = "Qwen/Qwen2.5-3B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training parameters
TUNING_DATA_SUBSET_PROPORTION = 1
TUNING_LORA_ARGS = {
    "task_type": TaskType.CAUSAL_LM,
    "r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "target_modules": ["gate_proj", "down_proj", "up_proj", "q_proj", "v_proj", "k_proj", "o_proj"],
    "inference_mode": False,
}
TUNING_TRAINER_ARGS = {
    "output_dir": FINETUNING_OUTPUT_DIR,
    "num_train_epochs": 10,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "load_best_model_at_end": True,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "dataset_text_field": "text",
    "packing": True,
    "max_seq_length": 512,
    "fp16": True,
    "neftune_noise_alpha": 5,
    "gradient_accumulation_steps": 16,
    "optim": "adamw_bnb_8bit",
    "learning_rate": 5e-5,
    "lr_scheduler_type": "cosine_with_restarts",
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
}

# Inference parameters
SYSTEM_PROMPT = "You are a character in a movie. Respond with engaging dialogue."
INFERENCE_MAX_LENGTH = 64
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
    "model": "gpt-4o",
    "temperature": 0,
    "max_retries": 3
}

DPO_BETA = 0.3
DPO_LORA_ARGS = {
    "task_type": TaskType.SEQ_CLS,
    "r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "target_modules": ["gate_proj", "down_proj", "up_proj", "q_proj", "v_proj", "k_proj", "o_proj"],
    "inference_mode": False,
}
DPO_TRAINER_ARGS = {
    "output_dir": DPO_OUTPUT_DIR,
    "num_train_epochs": 10,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "load_best_model_at_end": True,
    "eval_strategy": "steps",
    "eval_steps": 500,
    "save_strategy": "steps",
    "save_steps": 500,
    "remove_unused_columns": False,
    "max_length": TUNING_TRAINER_ARGS["max_seq_length"],
    "fp16": True,
    "gradient_accumulation_steps": 8,
    "max_prompt_length": 256,
    "learning_rate": 1e-5,
    "lr_scheduler_type": "cosine_with_restarts",
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
}

# G-Eval prompts
SCORING_PROMPTS = {
    "coherence": """
        You will be given a line of movie dialogue and an AI-generated response. Your task is to rate the AI's response on coherence.
        
        Evaluation Criteria:
        Coherence (1-5) - The degree to which the AI's response logically connects to the given dialogue line and maintains a plausible flow of conversation within a movie context. A score of 1 indicates a completely incoherent response that doesn't fit the scene at all, while a score of 5 indicates a perfectly coherent response that naturally continues the dialogue in a way that could believably appear in a movie script.
        
        Evaluation Steps:
        1. Read the given movie dialogue line carefully, considering its potential context, tone, and implied setting.
        2. Read the AI's response and assess how well it follows from the given line in a cinematic context.
        3. Consider whether the response maintains the implied tone, setting, and character dynamics of the original line.
        4. Evaluate how well the response could continue or advance a potential movie scene.
        5. Assess whether the response introduces any abrupt or illogical shifts that would be jarring in a film dialogue.
        6. Assign a score from 1 to 5 based on the overall coherence of the AI's response in the context of movie dialogue.
    """,
    "consistency": """
        You will be given a line of movie dialogue and an AI-generated response. Your task is to rate the AI's response on consistency.

        Evaluation Criteria:
        Consistency (1-5) - The degree to which the AI's response maintains consistent information, tone, and character voice with the given dialogue line. A score of 1 indicates highly inconsistent responses with clear contradictions or tonal mismatches, while a score of 5 indicates perfectly consistent responses that maintain the established context and character voice.
        
        Evaluation Steps:
        1. Carefully read the given movie dialogue line, noting any implied information about the character, setting, or situation.
        2. Read the AI's response and check if it's consistent with the information, tone, and character voice implied by the original line.
        3. Look for any contradictions in facts, emotions, or character traits between the original line and the response.
        4. Assess whether the response maintains a consistent level of formality, emotion, or genre-appropriate language.
        5. Check if the response's tone and style are consistent with what one would expect in a movie dialogue continuation.
        6. Assign a score from 1 to 5 based on the overall consistency of the AI's response with the given movie dialogue line.
    """,
    "fluency": """
        You will be given a line of movie dialogue and an AI-generated response. Your task is to rate the AI's response on fluency.

        Evaluation Criteria:
        Fluency (1-5) - The quality of the AI's language in terms of grammar, vocabulary, and natural flow, specifically in the context of movie dialogue. A score of 1 indicates poor fluency with many errors and unnatural language that would be jarring in a film, while a score of 5 indicates perfect fluency that sounds natural and believable as movie dialogue.
        
        Evaluation Steps:
        1. Read the AI's response carefully, focusing on the language quality in the context of movie dialogue.
        2. Check for any grammatical errors, including issues with verb tenses, subject-verb agreement, and sentence structure.
        3. Assess the vocabulary use, looking for appropriate word choice and variety that fits well in a movie script.
        4. Evaluate the natural flow of language, checking if it sounds like realistic spoken dialogue.
        5. Look for any awkward phrasings or unnatural expressions that would sound out of place in a film.
        6. Consider the appropriate use of idioms, colloquialisms, or character-specific language that enhances the dialogue's authenticity.
        7. Assign a score from 1 to 5 based on the overall fluency of the AI's response as movie dialogue.
    """,
    "relevance": """
        You will be given a line of movie dialogue and an AI-generated response. Your task is to rate the AI's response on relevance.

        Evaluation Criteria:
        Relevance (1-5) - The degree to which the AI's response appropriately addresses or follows up on the given dialogue line in a way that makes sense for a movie scene. A score of 1 indicates a completely irrelevant response that doesn't fit the context of the dialogue at all, while a score of 5 indicates a highly relevant response that perfectly continues or responds to the given line in a cinematically appropriate way.
        
        Evaluation Steps:
        1. Carefully read the given movie dialogue line, identifying the main points, emotions, or subtext it conveys.
        2. Assess how directly the AI's response addresses or builds upon the given line in a way that makes sense for a movie scene.
        3. Check if the response acknowledges all important aspects of the original line, or if it misses any crucial points.
        4. Evaluate whether the response contributes to advancing a potential scene or character development in a relevant way.
        5. Assess if the response stays on topic or if it introduces irrelevant information that doesn't fit the implied movie context.
        6. Consider the depth and specificity of the AI's response in relation to the given line and its potential place in a larger movie narrative.
        7. Assign a score from 1 to 5 based on the overall relevance of the AI's response to the given movie dialogue line.
    """
}