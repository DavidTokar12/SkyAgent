from __future__ import annotations

from itertools import groupby
from typing import TYPE_CHECKING
from typing import Any

from anthropic import Anthropic

from skyagent.base.api_adapters.api_adapter import ApiAdapter
from skyagent.base.api_adapters.api_adapter import ApiUsage
from skyagent.base.api_adapters.api_adapter import CompletionResponse
from skyagent.base.chat_message import AssistantMessage
from skyagent.base.chat_message import _BaseMessage
from skyagent.base.chat_message import SystemMessage
from skyagent.base.chat_message import ToolCallOutgoingMessage
from skyagent.base.chat_message import UserMessage
from skyagent.base.exceptions import SkyAgentContextWindowSaturatedError
from skyagent.base.exceptions import SkyAgentDetrimentalError
from skyagent.base.tools import ToolCall
from skyagent.base.tools import ToolCallResult
from skyagent.utils import _model_to_string


if TYPE_CHECKING:
    from anthropic.types import Message

    from skyagent.base.tools import Tool


class AnthropicApiAdapter(ApiAdapter):

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

        self.client = Anthropic(
            api_key=self.token,
            timeout=self.timeout,
            **(self.client_extra_args if self.client_extra_args is not None else {}),
        )

    def get_completion(
        self,
        chat_history: list[_BaseMessage],
        response_format: Any | None = None,
        tools: list[Tool] | None = None,
    ) -> CompletionResponse:

        if len(chat_history) == 0:
            raise SkyAgentDetrimentalError("message_history cannot be an empty array!")

        try:
            messages = []

            grouped_history = []
            for is_tool_call, group in groupby(
                chat_history, key=lambda msg: isinstance(msg, ToolCallOutgoingMessage)
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
            system_message = None

            for item in grouped_history:
                if isinstance(item, UserMessage):
                    pass
                    # text_and_image_messages = item.to_text_and_image_messages()
                    # if item.attached_images:
                    #     content = []

                    #     if item.content:
                    #         content.append(
                    #             {"type": "text", "text": item.content})

                    #     for file in item.attached_files:
                    #         content.append(
                    #             {
                    #                 "type": "text",
                    #                 "text": f"File: {file.file_name}",
                    #             }
                    #         )

                    #     for image in item.attached_images:
                    #         content.append(
                    #             {
                    #                 "type": "image",
                    #                 "source": {
                    #                     "type": "base64",
                    #                     "media_type": "image/jpeg",
                    #                     "data": image.base_64,
                    #                 },
                    #             }
                    #         )

                    #     messages.append(
                    #         {"role": item.role.value, "content": content})
                    # else:
                    #     # Handle regular text messages
                    #     messages.append(
                    #         {
                    #             "role": item.role.value,
                    #             "content": item.content,
                    #         }
                    #     )
                elif isinstance(item, AssistantMessage):
                    messages.append(
                        {
                            "role": item.role.value,
                            "content": item.content,
                        }
                    )
                elif isinstance(item, SystemMessage):
                    system_message = item.content
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

            if response_format is not None:
                messages.append(
                    {
                        "role": "user",
                        "content": f"Your output must be a valid JSON object with the following schema:\n{_model_to_string(response_format)}",
                    }
                )

            completion_args = self._build_completion_arguments(
                messages, system_message, tools
            )

            response: Message = self.client.messages.create(**completion_args)

        except Exception as e:
            raise SkyAgentDetrimentalError(f"Chat completion failed: '{e}'")

        finish_reason = response.stop_reason

        if finish_reason == "max_tokens":
            raise SkyAgentContextWindowSaturatedError("Context window exceeded!")

        usage = ApiUsage(
            output_tokens=response.usage.output_tokens,
            input_tokens=response.usage.input_tokens,
        )

        if finish_reason == "tool_use":

            chat_history.append({"role": "assistant", "content": response.content})

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

    def tool_to_dict(self, tool: Tool) -> dict:

        properties_dict = {
            param.name: {
                "type": param.type,
                "description": param.description,
            }
            for param in tool.parameters
        }

        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": {
                "type": "object",
                "properties": properties_dict,
                "required": tool.required_properties,
            },
        }

    def _build_completion_arguments(
        self,
        messages: list,
        system_message: str | None = None,
        tools: list[Tool] | None = None,
    ) -> dict:
        """
        Build completion arguments for Anthropic API.

        Args:
            messages: Formatted chat messages
            system_message: Optional system message
            tools: Optional list of tools

        Returns:
            dict: Arguments for Anthropic API completion call
        """
        completion_args = {
            "model": self.model,
            "messages": messages,
        }

        if system_message:
            completion_args["system"] = system_message

        if tools:
            completion_args["tools"] = [self.tool_to_dict(tool) for tool in tools]

        if completion_args.get("max_tokens") is None:
            completion_args["max_tokens"] = 4096

        if self.model_extra_args:
            completion_args.update(self.model_extra_args)

        return completion_args
