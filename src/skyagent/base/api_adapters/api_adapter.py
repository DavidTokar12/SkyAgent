from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from skyagent.base.tools import ToolCall  # noqa: TCH001


if TYPE_CHECKING:
    from skyagent.base.chat_message import _BaseMessage
    from skyagent.base.chat_message import ToolCallOutgoingMessage
    from skyagent.base.tools import Tool
    from skyagent.base.tools import ToolCallResult


class ApiUsage(BaseModel):
    output_tokens: int
    input_tokens: int


class CompletionResponse(BaseModel):
    content: str | BaseModel | None
    tool_calls: list[ToolCall] | None
    usage: ApiUsage


class ApiAdapter:

    def __init__(
        self,
        model: str,
        token: str | None = None,
        timeout: int = 10,
        model_extra_args: dict | None = None,
        client_extra_args: dict | None = None,
    ):

        self.model = model
        self.token = token
        self.timeout = timeout
        self.model_extra_args = model_extra_args
        self.client_extra_args = client_extra_args

    def get_completion(
        self,
        message_history: list[_BaseMessage],
        response_format: BaseModel | None = None,
        tools: list[Tool] | None = None,
    ) -> CompletionResponse:
        raise NotImplementedError("The get_completion method must be implemented!")

    def convert_tool_result_answer(
        self, tool_call_result: ToolCallResult
    ) -> ToolCallOutgoingMessage:
        raise NotImplementedError(
            "The convert_tool_result_answer method must be implemented!"
        )

    def tool_to_dict(self, tool: Tool) -> dict:
        raise NotImplementedError("The tool_to_dict method must be implemented!")
