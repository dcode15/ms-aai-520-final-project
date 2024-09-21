import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer import SFTTrainer, SFTConfig

import config
from preprocessor import Preprocessor

dataset = Preprocessor.prepare_dataset()

train_test = dataset.train_test_split(test_size=0.2, seed=1)

model = AutoModelForCausalLM.from_pretrained(config.BASE_MODEL_NAME, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)

model = get_peft_model(model, LoraConfig(**config.TUNING_LORA_ARGS))

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=SFTConfig(**config.TUNING_TRAINER_ARGS),
    train_dataset=train_test['train'],
    eval_dataset=train_test['test']
)

trainer.train()

trainer.save_model(config.TRAINED_MODEL_PATH)
