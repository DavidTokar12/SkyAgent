from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from enum import Enum
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal


if TYPE_CHECKING:
    from pydantic import BaseModel


class MessagePartType(str, Enum):
    system_prompt = "system_prompt"
    user_prompt = "user_prompt"

    image_attachment = "image_attachment"
    image_url = "image_url"

    document_attachment = "document_attachment"
    document_url = "document_url"

    retry_prompt = "retry_prompt"

    text_response = "text_response"
    structured_response = "structured_response"

    tool_call = "tool_call"
    tool_result = "tool_result"


###################### USER MESSAGES ######################


@dataclass
class SystemPrompt:
    content: str

    part_type: Literal[MessagePartType.system_prompt] = MessagePartType.system_prompt
    timestamp: datetime = datetime.now(tz=timezone.utc)


@dataclass
class UserPrompt:
    content: str

    part_type: Literal[MessagePartType.user_prompt] = MessagePartType.user_prompt
    timestamp: datetime = datetime.now(tz=timezone.utc)


@dataclass
class ImageAttachment:
    base_64: str
    file_name: str

    part_type: Literal[MessagePartType.image_attachment] = (
        MessagePartType.image_attachment
    )
    timestamp: datetime = datetime.now(tz=timezone.utc)


@dataclass
class ImageUrl:
    url: str

    part_type: Literal[MessagePartType.image_url] = MessagePartType.image_url
    timestamp: datetime = datetime.now(tz=timezone.utc)


@dataclass
class DocumentAttachment:
    base_64: str
    file_name: str

    part_type: Literal[MessagePartType.document_attachment] = (
        MessagePartType.document_attachment
    )
    timestamp: datetime = datetime.now(tz=timezone.utc)


@dataclass
class DocumentUrl:
    url: str

    part_type: Literal[MessagePartType.document_url] = MessagePartType.document_url
    timestamp: datetime = datetime.now(tz=timezone.utc)


@dataclass
class RetryPrompt:
    content: str

    part_type: Literal[MessagePartType.retry_prompt] = MessagePartType.retry_prompt
    timestamp: datetime = datetime.now(tz=timezone.utc)


@dataclass
class ModelInput:
    message_parts: list[
        SystemPrompt
        | UserPrompt
        | ImageAttachment
        | ImageUrl
        | DocumentAttachment
        | DocumentUrl
        | RetryPrompt
    ]


###################### USER MESSAGES ######################

###################### MODEL MESSAGES ######################


@dataclass
class TextResponse:
    content: str

    part_type: Literal[MessagePartType.text_response] = MessagePartType.text_response
    timestamp: datetime = datetime.now(tz=timezone.utc)


@dataclass
class StructuredResponse:
    content: BaseModel

    part_type: Literal[MessagePartType.structured_response] = (
        MessagePartType.structured_response
    )
    timestamp: datetime = datetime.now(tz=timezone.utc)


@dataclass
class ToolCall:
    tool_name: str
    args: dict[str, Any]
    tool_call_id: str

    part_type: Literal[MessagePartType.tool_call] = MessagePartType.tool_call
    timestamp: datetime = datetime.now(tz=timezone.utc)


@dataclass
class ToolResult:
    tool_name: str
    content: Any
    tool_call_id: str

    part_type: Literal[MessagePartType.tool_result] = MessagePartType.tool_result
    timestamp: datetime = datetime.now(tz=timezone.utc)


@dataclass
class ModelOutput:
    model_name: str
    agent_name: str
    message_parts: list[TextResponse | StructuredResponse | ToolCall | ToolResult]


###################### MODEL MESSAGES ######################

###################### STREAMING MESSAGES ######################


class EventType(str, Enum):
    text_delta = "text_delta"
    final_text_result = "final_text_result"

    structured_delta = "structured_delta"
    final_structured_result = "final_structured_result"

    tool_call_start = "tool_call_start"
    tool_call_end = "tool_call_end"


# @dataclass
# class BaseStreamEvent:
#     event_type: EventType
#     timestamp: datetime = field(default_factory=datetime.now(tz=timezone.utc))


@dataclass
class TextDelta:
    content: str

    event_type: Literal[EventType.text_delta] = EventType.text_delta
    timestamp: datetime = datetime.now(tz=timezone.utc)


@dataclass
class StructuredDelta:
    content: BaseModel

    event_type: Literal[EventType.structured_delta] = EventType.structured_delta
    timestamp: datetime = datetime.now(tz=timezone.utc)


@dataclass
class ToolCallStart:
    tool_name: str
    tool_call_id: str
    args: dict[str, Any]

    event_type: Literal[EventType.tool_call_start] = EventType.tool_call_start
    timestamp: datetime = datetime.now(tz=timezone.utc)


@dataclass
class ToolCallEnd:
    tool_name: str
    tool_call_id: str
    content: Any

    event_type: Literal[EventType.tool_call_end] = EventType
    timestamp: datetime = datetime.now(tz=timezone.utc)


@dataclass
class FinalTextResult:
    content: str

    event_type: Literal[EventType.final_text_result] = EventType.final_text_result
    timestamp: datetime = datetime.now(tz=timezone.utc)


@dataclass
class FinalStructuredResult:
    content: BaseModel

    event_type: Literal[EventType.final_structured_result] = (
        EventType.final_structured_result
    )
    timestamp: datetime = datetime.now(tz=timezone.utc)


###################### STREAMING MESSAGES ######################
