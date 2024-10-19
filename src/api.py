from typing import Any

from beam import asgi, Image

import config
from inference import Chatbot
from models import ChatInput, Message
from fastapi import FastAPI


def init_models() -> tuple[Chatbot, Chatbot, Chatbot]:
    """
    Initializes three versions of the Chatbot: base, fine-tuned, and DPO-trained.

    Returns:
        tuple[Chatbot, Chatbot, Chatbot]: A tuple containing the base, fine-tuned, and DPO-trained Chatbots.
    """
    return Chatbot(config.BASE_MODEL_NAME), Chatbot(config.FINETUNED_MODEL_PATH), Chatbot(config.TRAINED_DPO_MODEL_PATH)


@asgi(
    name="chatbot",
    image=Image(
        python_version="python3.12",
        python_packages=["transformers", "torch", "fastapi", "pydantic", "peft", "bitsandbytes"]
    ),
    on_start=init_models,
    authorized=False,
    gpu="T4"
)
def web_server(context: Any) -> FastAPI:
    """
    Creates and configures the FastAPI web server for the chatbot.

    Args:
        context (Any): The context provided by the Beam framework.

    Returns:
        FastAPI: The configured FastAPI application.
    """
    from fastapi import FastAPI

    app = FastAPI()

    base_chatbot, finetuned_chatbot, dpo_chatbot = context.on_start_value

    @app.post("/chat")
    async def generate_text(chat_input: ChatInput) -> str:
        """
        Generates a response based on the chat input.

        Args:
            chat_input (ChatInput): The input for the chat, including messages and configuration.

        Returns:
            str: The generated response from the selected chatbot model.
        """
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
