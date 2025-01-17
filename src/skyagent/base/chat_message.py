from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class ChatMessageRole(str, Enum):
    system = "system"
    assistant = "assistant"
    user = "user"
    tool = "tool"


class BaseChatMessage(BaseModel):
    role: ChatMessageRole

    def __str__(self) -> str:

        data = self.model_dump()

        max_length = 124
        for key, value in data.items():
            if isinstance(value, str) and len(value) > max_length:
                data[key] = value[:max_length] + "... [truncated]"

        fields_str = ", ".join(f"{k}={v!r}" for k, v in data.items())
        return f"{self.__class__.__name__}({fields_str})"


class UserChatMessage(BaseChatMessage):
    "Represents a user message in a chat history."
    role: ChatMessageRole = ChatMessageRole.user
    content: str


class SystemChatMessage(BaseChatMessage):
    "Represents a system message in a chat history."
    role: ChatMessageRole = ChatMessageRole.system
    content: str


class AssistantChatMessage(BaseChatMessage):
    "Represents an assistant message in a chat history."
    role: ChatMessageRole = ChatMessageRole.assistant
    content: str


class ToolCallOutgoingMessage(BaseChatMessage):
    role: ChatMessageRole = ChatMessageRole.tool
    content: str
    tool_call_id: str
