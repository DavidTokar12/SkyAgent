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


class UserChatMessage(BaseModel):
    "Represents a user message in a chat history."
    role: ChatMessageRole = ChatMessageRole.user
    content: str


class SystemChatMessage(BaseModel):
    "Represents a system message in a chat history."
    role: ChatMessageRole = ChatMessageRole.system
    content: str


class AssistantChatMessage(BaseModel):
    "Represents an assistant message in a chat history."
    role: ChatMessageRole = ChatMessageRole.assistant
    content: str


class ToolCallOutgoingMessage(BaseModel):
    role: ChatMessageRole = ChatMessageRole.tool
    content: str
    tool_call_id: str
