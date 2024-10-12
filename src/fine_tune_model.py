import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback
from trl.trainer import SFTTrainer, SFTConfig

import config
from preprocessor import Preprocessor
from utils import set_seeds

set_seeds()
dataset = Preprocessor.prepare_dataset()

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    config.BASE_MODEL_NAME,
    quantization_config=quantization_config,
    attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(**config.TUNING_LORA_ARGS))

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
