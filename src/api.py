from beam import asgi, Image
from pydantic import BaseModel

from inference import Chatbot


class ChatInput(BaseModel):
    message: str


def init_models():
    return Chatbot()


@asgi(
    name="chatbot",
    image=Image(
        python_version="python3.12",
        python_packages=["transformers", "torch", "fastapi", "pydantic", "peft"]
    ),
    on_start=init_models,
)
def web_server(context):
    from fastapi import FastAPI

    app = FastAPI()

    chatbot = context.on_start_value

    @app.post("/chat")
    async def generate_text(message: ChatInput):
        chatbot.start_conversation()
        return chatbot.generate_response(message.message)

    return app
