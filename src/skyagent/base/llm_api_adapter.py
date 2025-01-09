from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from skyagent.base.tools import ToolCall  # noqa: TCH001


if TYPE_CHECKING:
    from skyagent.base.chat_message import BaseChatMessage
    from skyagent.base.tools import BaseTool


class LlmUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int


class CompletionResponse(BaseModel):
    content: str | None
    tool_calls: list[ToolCall] | None
    usage: LlmUsage


class LlmApiAdapter:

    def __init__(
        self,
        model: str,
        token: str | None,
        temperature: float = 0.0,
        timeout: int = 3,
    ):

        self.model = model
        self.token = token
        self.temperature = temperature
        self.timeout = timeout

    def get_completion(
        self,
        message_history: list[BaseChatMessage],
        response_format: BaseModel | None = None,
        tools: list[BaseTool] | None = None,
    ) -> CompletionResponse:
        raise NotImplementedError("The get_completion method must be implemented!")
