from beam import asgi, Image

import config
from inference import Chatbot
from models import ChatInput, Message


def init_models():
    return Chatbot(config.BASE_MODEL_NAME), Chatbot(config.FINETUNED_MODEL_PATH), Chatbot(config.TRAINED_DPO_MODEL_PATH)


@asgi(
    name="chatbot",
    image=Image(
        python_version="python3.12",
        python_packages=["transformers", "torch", "fastapi", "pydantic", "peft", "bitsandbytes"]
    ),
    on_start=init_models,
    authorized=False,
    gpu="RTX4090"
)
def web_server(context):
    from fastapi import FastAPI

    app = FastAPI()

    base_chatbot, finetuned_chatbot, dpo_chatbot = context.on_start_value

    @app.post("/new-conversation")
    async def new_conversation():
        base_chatbot.start_conversation()
        finetuned_chatbot.start_conversation()
        dpo_chatbot.start_conversation()

    @app.post("/chat")
    async def generate_text(chat_input: ChatInput):
        if chat_input.mode == "Base":
            chatbot = base_chatbot
        elif chat_input.mode == "Fine-tuned":
            chatbot = finetuned_chatbot
        else:
            chatbot = dpo_chatbot

        new_message: Message = chat_input.messages.pop()
        chatbot.start_conversation(list(map(lambda message: dict(message), chat_input.messages)))
        if chat_input.config is not None:
            return chatbot.generate_response(new_message.content,
                                             config_override={**config.INFERENCE_PARAMS, **dict(chat_input.config)})
        else:
            return chatbot.generate_response(new_message.content)

    return app
