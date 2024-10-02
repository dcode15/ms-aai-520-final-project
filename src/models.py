from enum import Enum
from typing import Optional

from pydantic import BaseModel


class ChatRole(str, Enum):
    user = "user"
    assistant = "assistant"


class Message(BaseModel):
    role: ChatRole
    content: str


class ChatConfig(BaseModel):
    temperature: float
    top_p: float
    top_k: int


class ChatInput(BaseModel):
    messages: list[Message]
    config: Optional[ChatConfig] = None
