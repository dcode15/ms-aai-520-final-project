import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl.trainer import SFTTrainer, SFTConfig

import config
from preprocessor import Preprocessor
from utils import set_seeds

set_seeds()
dataset = Preprocessor.prepare_dataset()

train_test = dataset.train_test_split(test_size=0.2, seed=1)

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    config.BASE_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=quantization_config,
    attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

model = get_peft_model(model, LoraConfig(**config.TUNING_LORA_ARGS))

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=SFTConfig(**config.TUNING_TRAINER_ARGS),
    train_dataset=train_test['train'],
    eval_dataset=train_test['test']
)

trainer.train()

trainer.save_model(config.FINETUNED_MODEL_PATH)
