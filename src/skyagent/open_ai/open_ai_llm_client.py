from __future__ import annotations

import ast
import json

from typing import TYPE_CHECKING
from typing import Any

from openai import OpenAI

from skyagent.base_classes import AgentConversationToLongError
from skyagent.base_classes import AgentCopyrightError
from skyagent.base_classes import AgentDetrimentalError
from skyagent.base_classes import AssistantChatMessage
from skyagent.base_classes import CompletionResponse
from skyagent.base_classes import ImageChatMessage
from skyagent.base_classes import LLMClient
from skyagent.base_classes import LLMUsage
from skyagent.base_classes import ToolCall
from skyagent.base_classes import ToolCallOutgoingMessage
from skyagent.base_classes import UserChatMessage


if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

    from skyagent.open_ai.open_ai_tool import OpenAITool


class OpenAILLMClient(LLMClient):

    def __init__(
        self,
        model: str,
        token: str | None = None,
        temperature: float = 0.0,
        timeout: int = 3,
    ):
        super().__init__(model, token, temperature, timeout)

        self.client = OpenAI(api_key=self.token, timeout=self.timeout)

    def get_completion(
        self,
        chat_history: list[
            UserChatMessage | ImageChatMessage | ToolCallOutgoingMessage
        ],
        response_format: Any | None = None,
        tools: list[OpenAITool] | None = None,
    ) -> CompletionResponse:

        if len(chat_history) == 0:
            raise AgentDetrimentalError("message_history cannot be an empty array!")

        try:

            messages = []
            for message in chat_history:
                if isinstance(message, UserChatMessage | AssistantChatMessage):
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
                elif isinstance(message, ImageChatMessage):
                    messages.append(
                        {"role": message.role.value, "image_url": message.image_url}
                    )
                else:
                    # we keep incoming tool calls
                    messages.append(message)

            response: ChatCompletion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=[tool.to_dict() for tool in tools],
                response_format=response_format,
                timeout=self.timeout,
                temperature=self.temperature,
            )
        except Exception as e:
            raise AgentDetrimentalError("Chat completion failed: '%s'", e)

        finish_reason = response.choices[0].finish_reason

        if finish_reason == "length":
            raise AgentConversationToLongError("Context window exceeded!")

        if finish_reason == "content_filter":
            raise AgentCopyrightError("Query was filtered due to copyright reasons!")

        usage = LLMUsage(
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
            raise AgentDetrimentalError(
                "OpenAI API returned with an unexpected finish reason: '%s'",
                finish_reason,
            )

    def generate_tool_result_answer(
        self, tool_call: ToolCall, result: Any
    ) -> ToolCallOutgoingMessage:
        """
        Generates a message dict containing the result of a tool call.
        This message is appended to the conversation history so that
        the model can consume the tool's result.
        """

        content_dict = {
            "input": tool_call.arguments,
            "output": result,
        }

        return ToolCallOutgoingMessage(
            content=json.dumps(content_dict), tool_call_id=tool_call.id
        )