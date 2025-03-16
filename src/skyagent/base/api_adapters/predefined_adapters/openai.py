from __future__ import annotations

import ast
import json

from typing import TYPE_CHECKING

from openai import OpenAI

from skyagent.base.api_adapters.api_adapter import ApiAdapter
from skyagent.base.api_adapters.api_adapter import ApiUsage
from skyagent.base.api_adapters.api_adapter import CompletionResponse
from skyagent.base.chat_message import _BaseMessage
from skyagent.base.chat_message import SystemMessage
from skyagent.base.chat_message import ToolCallOutgoingMessage
from skyagent.base.chat_message import UserMessage
from skyagent.base.exceptions import SkyAgentContextWindowSaturatedError
from skyagent.base.exceptions import SkyAgentCopyrightError
from skyagent.base.exceptions import SkyAgentDetrimentalError
from skyagent.base.tools import ToolCall
from skyagent.base.tools import ToolCallResult


if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion
    from pydantic import BaseModel

    from skyagent.base.tools import Tool


class OpenAiApiAdapter(ApiAdapter):

    def __init__(
        self,
        model: str,
        token: str | None = None,
        timeout: int = 10,
        model_extra_args: dict | None = None,
        client_extra_args: dict | None = None,
    ):
        super().__init__(
            model=model,
            token=token,
            timeout=timeout,
            model_extra_args=model_extra_args,
            client_extra_args=client_extra_args,
        )

        self.client = OpenAI(
            api_key=self.token,
            timeout=self.timeout,
            **(client_extra_args if client_extra_args is not None else {}),
        )

    def get_completion(
        self,
        chat_history: list[_BaseMessage],
        response_format: BaseModel | None = None,
        tools: list[Tool] | None = None,
    ) -> CompletionResponse:

        if len(chat_history) == 0:
            raise SkyAgentDetrimentalError("chat_history cannot be an empty array!")

        try:

            messages = []
            for message in chat_history:
                if isinstance(message, UserMessage):

                    if message.attached_images:
                        content = []

                        if message.content:
                            content.append({"type": "text", "text": message.content})

                        for image in message.attached_images:
                            content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image.base_64}"
                                    },
                                }
                            )

                        messages.append(
                            {"role": message.role.value, "content": content}
                        )
                    else:
                        messages.append(
                            {"role": message.role.value, "content": message.content}
                        )
                elif isinstance(message, SystemMessage):
                    messages.append(
                        {"role": message.role.value, "content": message.content}
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

            completion_args = self._build_completion_arguments(
                messages, tools, response_format
            )

            if response_format is None:
                response: ChatCompletion = self.client.chat.completions.create(
                    **completion_args
                )
            else:
                response: ChatCompletion = self.client.beta.chat.completions.parse(
                    **completion_args
                )

        except Exception as e:
            raise SkyAgentDetrimentalError(f"Chat completion failed: '{e}'")

        finish_reason = response.choices[0].finish_reason

        if finish_reason == "length":
            raise SkyAgentContextWindowSaturatedError("Context window exceeded!")

        if finish_reason == "content_filter":
            raise SkyAgentCopyrightError("Query was filtered due to copyright reasons!")

        usage = ApiUsage(
            output_tokens=response.usage.completion_tokens,
            input_tokens=response.usage.prompt_tokens,
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

    def tool_to_dict(self, tool: Tool) -> dict:

        properties_dict = {
            param.name: {
                "type": param.type,
                "description": param.description,
            }
            for param in tool.parameters
        }

        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": properties_dict,
                    "required": tool.required_properties,
                    "additionalProperties": tool.additional_properties,
                },
            },
        }

    def _build_completion_arguments(
        self,
        messages: list,
        tools: list[Tool] | None = None,
        response_format: BaseModel | None = None,
    ) -> dict:
        """
        Build completion arguments for OpenAI API.

        Args:
            messages: Formatted chat messages
            tools: Optional list of tools
            response_format: Optional response format specification

        Returns:
            dict: Arguments for OpenAI API completion call
        """

        completion_args = {
            "model": self.model,
            "messages": messages,
        }

        if tools:
            completion_args["tools"] = [self.tool_to_dict(tool) for tool in tools]

        if response_format is not None:
            completion_args["response_format"] = response_format

        if self.model_extra_args:
            completion_args.update(self.model_extra_args)

        return completion_args
