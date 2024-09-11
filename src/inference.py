from transformers import AutoModelForCausalLM, AutoTokenizer

import config

model_path = config.MODEL_NAME if config.USE_BASE_MODEL else config.TRAINED_MODEL_PATH
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)


def generate_response(input_text: str, max_length: int = 50) -> str:
    messages = [
        {"role": "system", "content": "You are a friendly chatbot."},
        {"role": "user", "content": input_text},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(config.DEVICE)
    attention_mask = model_inputs.input_ids.ne(tokenizer.pad_token_id).long().to(config.DEVICE)

    output = model.generate(
        model_inputs.input_ids,
        attention_mask=attention_mask,
        max_new_tokens=128
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, output)
    ]
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    parts = generated_text.split("<endheader>")
    if len(parts) > 2:
        return parts[1].strip()
    else:
        return generated_text.strip()


# Example usage
if __name__ == "__main__":
    model_type = "base" if config.USE_BASE_MODEL else "fine-tuned"
    print(f"Chatbot: Hello! I'm your AI assistant using the {model_type} model. How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Chatbot: Goodbye! Have a great day!")
            break
        response = generate_response(user_input, max_length=config.INFERENCE_MAX_LENGTH)
        print("Chatbot:", response)
