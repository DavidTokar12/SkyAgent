from __future__ import annotations

from itertools import groupby
from typing import TYPE_CHECKING
from typing import Any

from anthropic import Anthropic

from skyagent.base.chat_message import AssistantChatMessage
from skyagent.base.chat_message import BaseChatMessage
from skyagent.base.chat_message import SystemChatMessage
from skyagent.base.chat_message import ToolCallOutgoingMessage
from skyagent.base.chat_message import UserChatMessage
from skyagent.base.exceptions import SkyAgentContextWindowSaturatedError
from skyagent.base.exceptions import SkyAgentDetrimentalError
from skyagent.base.llm_api_adapter import CompletionResponse
from skyagent.base.llm_api_adapter import LlmApiAdapter
from skyagent.base.llm_api_adapter import LlmUsage
from skyagent.base.tools import ToolCall
from skyagent.base.tools import ToolCallResult
from skyagent.utils import model_to_string


if TYPE_CHECKING:
    from anthropic.types import Message

    from skyagent.anthropic.anthropic_tool import AnthropicTool


class AnthropicApiAdapter(LlmApiAdapter):

    def __init__(
        self,
        model: str,
        token: str | None = None,
        temperature: float = 0.0,
        timeout: int = 30,
        max_token: int = 4096,
    ):
        super().__init__(model, token, temperature, timeout)

        self.max_token = max_token

        self.client = Anthropic(api_key=self.token, timeout=self.timeout)

    def get_completion(
        self,
        chat_history: list[BaseChatMessage],
        response_format: Any | None = None,
        tools: list[AnthropicTool] | None = None,
    ) -> CompletionResponse:

        if len(chat_history) == 0:
            raise SkyAgentDetrimentalError(
                "message_history cannot be an empty array!")

        try:
            messages = []

            grouped_history = []
            for is_tool_call, group in groupby(
                chat_history, key=lambda msg: isinstance(
                    msg, ToolCallOutgoingMessage)
            ):
                if is_tool_call:
                    # Group tool call outgoing messages
                    grouped_history.append(
                        [
                            {
                                "type": "tool_result",
                                "tool_use_id": message.tool_call_id,
                                "content": message.content,
                            }
                            for message in group
                        ]
                    )
                else:
                    # Collect other messages as-is
                    grouped_history.extend(group)

            messages = []
            for item in grouped_history:
                if isinstance(
                    item, UserChatMessage | AssistantChatMessage | SystemChatMessage
                ):
                    messages.append(
                        {
                            "role": item.role.value,
                            "content": item.content,
                        }
                    )
                elif isinstance(item, list):  # This is a grouped tool result
                    messages.append(
                        {
                            "role": "user",
                            "content": item,
                        }
                    )
                else:
                    # Handle incoming tool use messages
                    messages.append(item)

            system_message = messages[0]["content"]

            if response_format is not None:
                system_message += f"\n Your output must be a valid JSON object with the following schema:\n {
                    model_to_string(response_format)}"

            messages = messages[1:]

            response: Message = self.client.messages.create(
                model=self.model,
                system=system_message,
                messages=messages,
                max_tokens=self.max_token,
                tools=[tool.to_dict() for tool in tools] if tools else [],
                timeout=self.timeout,
                temperature=self.temperature,
            )

        except Exception as e:
            raise SkyAgentDetrimentalError(f"Chat completion failed: '{e}'")

        finish_reason = response.stop_reason

        if finish_reason == "max_tokens":
            raise SkyAgentContextWindowSaturatedError(
                "Context window exceeded!")

        usage = LlmUsage(
            completion_tokens=response.usage.output_tokens,
            prompt_tokens=response.usage.input_tokens,
        )

        if finish_reason == "tool_use":

            chat_history.append(
                {"role": "assistant", "content": response.content})

            parsed_tool_calls = []

            for block in response.content:
                if block.type == "tool_use":
                    tool_call = block

                    tool_call_function_name = tool_call.name
                    tool_call_input = tool_call.input
                    tool_call_id = tool_call.id

                    parsed_tool_call = ToolCall(
                        id=tool_call_id,
                        function_name=tool_call_function_name,
                        arguments=tool_call_input,
                    )

                    parsed_tool_calls.append(parsed_tool_call)

            return CompletionResponse(
                content=None, tool_calls=parsed_tool_calls, usage=usage
            )

        elif finish_reason == "end_turn":

            final_response = response.content[0].text

            if response_format is not None:
                try:
                    response_format.model_validate_json(final_response)
                except Exception as e:
                    raise SkyAgentDetrimentalError(
                        f"Failed to validate response format: {e}"
                    )

            return CompletionResponse(
                content=final_response,
                tool_calls=None,
                usage=usage,
            )
        else:
            raise SkyAgentDetrimentalError(
                f"Anthropic API returned with an unexpected finish reason: '{
                    finish_reason}'"
            )

    def convert_tool_result_answer(
        self, tool_call_result: ToolCallResult
    ) -> ToolCallOutgoingMessage:

        return ToolCallOutgoingMessage(
            content=str(tool_call_result.result), tool_call_id=tool_call_result.id
        )
