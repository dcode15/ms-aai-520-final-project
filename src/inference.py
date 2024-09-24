from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import config


class Chatbot:

    def __init__(self, model_override: str = None):
        if model_override is not None:
            model_path = model_override
        elif config.USE_BASE_MODEL:
            model_path = config.BASE_MODEL_NAME
        else:
            model_path = config.TRAINED_MODEL_PATH

        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            quantization_config=quantization_config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.conversation = []
        self.start_conversation()

    def start_conversation(self):
        self.conversation = [{"role": "system", "content": config.SYSTEM_PROMPT}]

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
        model_inputs = self.tokenizer([text], return_tensors="pt").to(config.DEVICE)
        attention_mask = model_inputs.input_ids.ne(self.tokenizer.pad_token_id).long().to(config.DEVICE)

        params = config_override if config_override is not None else config.INFERENCE_PARAMS
        output = self.model.generate(
            model_inputs.input_ids,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=max_length,
            **params
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, output)
        ]
        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        reply = generated_text.split("\n")[0]
        self.conversation.append({"role": "assistant", "content": reply})
        return reply


if __name__ == "__main__":
    print(f"Chatbot: Hello! I'm your AI assistant. How can I help you today?")
    chatbot = Chatbot()
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Chatbot: Goodbye! Have a great day!")
            break
        response = chatbot.generate_response(user_input, max_length=config.INFERENCE_MAX_LENGTH)
        print("Chatbot:", response)
