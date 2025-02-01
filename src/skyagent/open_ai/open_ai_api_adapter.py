from __future__ import annotations

import ast
import json

from typing import TYPE_CHECKING
from typing import Any

from openai import OpenAI

from skyagent.base.chat_message import AssistantChatMessage
from skyagent.base.chat_message import BaseChatMessage
from skyagent.base.chat_message import SystemChatMessage
from skyagent.base.chat_message import ToolCallOutgoingMessage
from skyagent.base.chat_message import UserChatMessage
from skyagent.base.exceptions import SkyAgentContextWindowSaturatedError
from skyagent.base.exceptions import SkyAgentCopyrightError
from skyagent.base.exceptions import SkyAgentDetrimentalError
from skyagent.base.llm_api_adapter import CompletionResponse
from skyagent.base.llm_api_adapter import LlmApiAdapter
from skyagent.base.llm_api_adapter import LlmUsage
from skyagent.base.tools import ToolCall
from skyagent.base.tools import ToolCallResult


if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

    from skyagent.open_ai.open_ai_tool import OpenAITool


class OpenAiApiAdapter(LlmApiAdapter):

    def __init__(
        self,
        model: str,
        token: str | None = None,
        temperature: float = 0.0,
        timeout: int = 3,
    ):
        super().__init__(
            model=model,
            token=token,
            temperature=temperature,
            timeout=timeout,
        )

        self.client = OpenAI(api_key=self.token, timeout=self.timeout)

    def get_completion(
        self,
        chat_history: list[BaseChatMessage],
        response_format: Any | None = None,
        tools: list[OpenAITool] | None = None,
    ) -> CompletionResponse:

        if len(chat_history) == 0:
            raise SkyAgentDetrimentalError("message_history cannot be an empty array!")

        try:

            messages = []
            for message in chat_history:
                if isinstance(
                    message, UserChatMessage | AssistantChatMessage | SystemChatMessage
                ):
                    messages.append(
                        {
                            "role": message.role.value,
                            "content": message.content,
                        }
                    )
                elif isinstance(message, ToolCallOutgoingMessage):
                    messages.append(
                        {
                            "role": message.role.value,
                            "content": message.content,
                            "tool_call_id": message.tool_call_id,
                        }
                    )
                else:
                    # we keep incoming tool calls
                    messages.append(message)

            if response_format is None:
                response: ChatCompletion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=[tool.to_dict() for tool in tools] if tools else None,
                    timeout=self.timeout,
                    temperature=self.temperature,
                )
            else:
                response: ChatCompletion = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    tools=[tool.to_dict() for tool in tools],
                    response_format=response_format,
                    timeout=self.timeout,
                    temperature=self.temperature,
                )
        except Exception as e:
            raise SkyAgentDetrimentalError(f"Chat completion failed: '{e}'")

        finish_reason = response.choices[0].finish_reason

        if finish_reason == "length":
            raise SkyAgentContextWindowSaturatedError("Context window exceeded!")

        if finish_reason == "content_filter":
            raise SkyAgentCopyrightError("Query was filtered due to copyright reasons!")

        usage = LlmUsage(
            completion_tokens=response.usage.completion_tokens,
            prompt_tokens=response.usage.prompt_tokens,
        )

        if finish_reason == "tool_calls":

            # gpt requires the tool call sent by him to be appended
            chat_history.append(response.choices[0].message.to_dict())

            parsed_tool_calls = [
                ToolCall(
                    id=tool_call.id,
                    function_name=tool_call.function.name,
                    arguments=ast.literal_eval(tool_call.function.arguments),
                )
                for tool_call in response.choices[0].message.tool_calls
            ]

            return CompletionResponse(
                content=None, tool_calls=parsed_tool_calls, usage=usage
            )

        elif finish_reason == "stop":
            return CompletionResponse(
                content=response.choices[0].message.content,
                tool_calls=None,
                usage=usage,
            )
        else:
            raise SkyAgentDetrimentalError(
                f"OpenAI API returned with an unexpected finish reason: '{
                    finish_reason}'"
            )

    def convert_tool_result_answer(
        self, tool_call_result: ToolCallResult
    ) -> ToolCallOutgoingMessage:

        content_dict = {
            "input": tool_call_result.arguments,
            "output": tool_call_result.result,
        }

        return ToolCallOutgoingMessage(
            content=json.dumps(content_dict), tool_call_id=tool_call_result.id
        )
