import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import config


class Chatbot:

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = config.FINETUNED_MODEL_PATH

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        if not model_path == config.BASE_MODEL_NAME:
            self.model = PeftModel.from_pretrained(self.model, model_path)

        self.model.eval()
        self.conversation = []
        self.start_conversation()

    def start_conversation(self, previous_conversation: list[dict] = None):
        self.conversation = []
        if previous_conversation is not None:
            self.conversation.append(previous_conversation)

    def generate_response(self, input_text: str, max_length: int = config.INFERENCE_MAX_LENGTH,
                          config_override: dict = None) -> str:
        self.conversation.append({
            "role": "user",
            "content": input_text
        })
        text = self.tokenizer.apply_chat_template(
            self.conversation,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(config.DEVICE)

        params = config_override if config_override is not None else config.INFERENCE_PARAMS
        with torch.inference_mode():
            output = self.model.generate(
                model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                max_new_tokens=max_length,
                **params
            )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, output)
        ]
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        reply = self._clean_text(generated_text.split("\n")[0])
        self.conversation.append({"role": "assistant", "content": reply})
        return reply

    def _clean_text(self, text: str) -> str:
        # Remove non-ASCII characters
        text = re.sub(r'[^\x00-\x7F]+', '', text)

        last_sentence_end = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
        return text[:last_sentence_end + 1] if last_sentence_end != -1 else text


if __name__ == "__main__":
    chatbot = Chatbot()
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit']:
            break
        elif user_input.lower() in ['start new conversation']:
            chatbot.start_conversation()
        else:
            response = chatbot.generate_response(user_input, max_length=config.INFERENCE_MAX_LENGTH)
            print("Chatbot:", response)
