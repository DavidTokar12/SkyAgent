from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from skyagent.base.tools import ToolCall  # noqa: TCH001


if TYPE_CHECKING:
    from skyagent.base.chat_message import BaseChatMessage
    from skyagent.base.chat_message import ToolCallOutgoingMessage
    from skyagent.base.tools import BaseTool
    from skyagent.base.tools import ToolCallResult


class LlmUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int


class CompletionResponse(BaseModel):
    content: str | BaseModel | None
    tool_calls: list[ToolCall] | None
    usage: LlmUsage


class LlmApiAdapter:

    def __init__(
        self,
        model: str,
        token: str | None = None,
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

    def convert_tool_result_answer(
        self, tool_call_result: ToolCallResult
    ) -> ToolCallOutgoingMessage:
        raise NotImplementedError(
            "The generate_tool_result_answer method must be implemented!"
        )
