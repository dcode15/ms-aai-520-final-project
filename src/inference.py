from transformers import GPT2Tokenizer, GPT2LMHeadModel

import config

model = GPT2LMHeadModel.from_pretrained(config.TRAINED_MODEL_PATH).to(config.DEVICE)
tokenizer = GPT2Tokenizer.from_pretrained(config.TRAINED_MODEL_PATH)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def generate_response(input_text: str, max_length: int = 50) -> str:
    input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt").to(config.DEVICE)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long().to(config.DEVICE)

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=len(input_ids[0]) + max_length,
        num_return_sequences=1,
        **config.INFERENCE_PARAMS
    )

    full_response = tokenizer.decode(output[0], skip_special_tokens=True)
    response = full_response[len(input_text):].strip()
    return response


# Example usage
if __name__ == "__main__":
    print("Chatbot: Hello! I'm your AI assistant. How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Chatbot: Goodbye! Have a great day!")
            break
        response = generate_response(user_input, max_length=config.INFERENCE_MAX_LENGTH)
        print("Chatbot:", response)
