import torch
from datasets import DatasetDict
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback, \
    PreTrainedTokenizerBase, PreTrainedModel
from trl.trainer import SFTTrainer, SFTConfig

import config
from preprocessor import Preprocessor
from utils import set_seeds


def init_model() -> PreTrainedModel:
    """
    Initializes the base model with quantization configuration.

    Returns:
        PreTrainedModel: The initialized base model.
    """
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    return AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_NAME,
        quantization_config=quantization_config,
        attn_implementation="flash_attention_2"
    )


def init_tokenizer() -> PreTrainedTokenizerBase:
    """
    Initializes the tokenizer for the base model.

    Returns:
        PreTrainedTokenizerBase: The initialized tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def prepare_model_for_training(model: PreTrainedModel) -> PeftModel:
    """
    Prepares the model for quantized training and applies LoRA configuration.

    Args:
        model (PreTrainedModel): The base model to be prepared.

    Returns:
        PeftModel: The prepared model with LoRA configuration.
    """
    model = prepare_model_for_kbit_training(model)
    return get_peft_model(model, LoraConfig(**config.TUNING_LORA_ARGS))


def train_model(model: PeftModel, tokenizer: PreTrainedTokenizerBase, dataset: DatasetDict) -> None:
    """
    Trains the model using the SFTTrainer.

    Args:
        model (PeftModel): The prepared model for training.
        tokenizer (PreTrainedTokenizerBase): The tokenizer for the model.
        dataset (DatasetDict): The dataset containing train and validation splits.
    """
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=SFTConfig(**config.TUNING_TRAINER_ARGS),
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        callbacks=[early_stopping_callback]
    )

    trainer.train()
    trainer.save_model(config.FINETUNED_MODEL_PATH)


def main() -> None:
    set_seeds()
    dataset = Preprocessor.prepare_dataset()

    model = init_model()
    tokenizer = init_tokenizer()
    model = prepare_model_for_training(model)

    train_model(model, tokenizer, dataset)


if __name__ == "__main__":
    main()
