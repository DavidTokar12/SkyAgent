from __future__ import annotations

import ast

from datetime import datetime
from datetime import timezone
from typing import TYPE_CHECKING
from typing import Any

import pydantic

from openai import NOT_GIVEN
from openai import OpenAI

from skyagent.exceptions import SkyAgentContextWindowSaturatedError
from skyagent.exceptions import SkyAgentCopyrightError
from skyagent.providers.model_settings.openai import OpenAIClientSettings
from skyagent.providers.model_settings.openai import OpenAIModelSettings
from skyagent.providers.provider import Provider
from skyagent.response import IterationResponse
from skyagent.tool import ToolCall
from skyagent.usage import Usage
from skyagent.utils import to_strict_json_schema


# from openai.types.chat.chat_completion_content_part_image_param import ImageURL
# from openai.types.chat.chat_completion_content_part_input_audio_param import InputAudio


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from openai.types.chat import ChatCompletion
    from openai.types.chat import ChatCompletionChunk
    from pydantic import BaseModel

    from skyagent.messages import ModelInput
    from skyagent.messages import ModelOutput
    from skyagent.tool import Tool


class OpenAiProvider(Provider):

    def __init__(
        self,
        model: str,
        model_settings: OpenAIModelSettings | None = None,
        client_settings: OpenAIClientSettings | None = None,
    ):

        if model_settings is None:
            model_settings = OpenAIModelSettings()

        if client_settings is None:
            client_settings = OpenAIClientSettings()

        super().__init__(
            model=model,
            model_settings=model_settings,
            client_settings=client_settings,
        )

        self.client = OpenAI(
            timeout=self.client_settings.timeout,
            api_key=self.client_settings.api_key,
            organization=self.client_settings.organization,
            project=self.client_settings.project,
            base_url=self.client_settings.base_url,
            websocket_base_url=self.client_settings.websocket_base_url,
            max_retries=self.client_settings.max_retries,
            default_headers=self.client_settings.default_headers,
            default_query=self.client_settings.default_query,
            http_client=self.client_settings.http_client,
        )

    async def run_iteration(
        self,
        chat_history: list[ModelInput | ModelOutput],
        result_format: BaseModel | None = None,
        tools: list[Tool] | None = None,
    ) -> IterationResponse:

        mapped_tools = [self._map_tool(tool) for tool in tools] if tools else None

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": "What is 12341234234 % 3",
                },
            ],
            n=1,
            tool_choice="auto" if tools else NOT_GIVEN,
            tools=mapped_tools or NOT_GIVEN,
            response_format=(
                self._map_result_format(result_format) if result_format else NOT_GIVEN
            ),
            stream=False,
            stream_options=NOT_GIVEN,
            parallel_tool_calls=self.model_settings.parallel_tool_calls,
            max_completion_tokens=self.model_settings.max_completion_tokens,
            temperature=self.model_settings.temperature,
            top_p=self.model_settings.top_p,
            timeout=self.model_settings.timeout,
            seed=self.model_settings.seed,
            presence_penalty=self.model_settings.presence_penalty,
            frequency_penalty=self.model_settings.frequency_penalty,
            logit_bias=self.model_settings.logit_bias,
            reasoning_effort=self.model_settings.reasoning_effort,
        )

        finish_reason = response.choices[0].finish_reason

        if finish_reason == "length":
            raise SkyAgentContextWindowSaturatedError("Context window exceeded!")

        if finish_reason == "content_filter":
            raise SkyAgentCopyrightError("Query was filtered due to copyright reasons!")

        usage = self._map_usage(response)

        choice = response.choices[0]
        content = None
        tool_calls = None
        timestamp = datetime.fromtimestamp(response.created, tz=timezone.utc)

        if choice.message.content is not None:
            content = choice.message.content

        if choice.message.tool_calls is not None:
            tool_calls = [
                ToolCall(
                    call_id=tool_call.id,
                    function_name=tool_call.function.name,
                    arguments=ast.literal_eval(tool_call.function.arguments),
                )
                for tool_call in choice.message.tool_calls
            ]

        return IterationResponse(
            usage=usage, timestamp=timestamp, content=content, tool_calls=tool_calls
        )

    async def run_iteration_stream(
        self,
        chat_history: list[ModelInput | ModelOutput],
        result_format: BaseModel | None = None,
        tools: list[Tool] | None = None,
    ) -> AsyncIterator:
        raise NotImplementedError(
            "The run_iteration_stream method must be implemented!"
        )

    @staticmethod
    def _map_tool(tool: Tool) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.param_schema,
            },
        }

    @staticmethod
    def _map_result_format(response_format: BaseModel) -> dict[str, Any]:
        json_schema = pydantic.TypeAdapter(response_format).json_schema()
        return {
            "type": "json_schema",
            "json_schema": {
                "schema": to_strict_json_schema(json_schema, path=(), root=json_schema),
                "name": response_format.__name__,
                "strict": True,
            },
        }

    @staticmethod
    def _map_usage(response: ChatCompletion | ChatCompletionChunk) -> Usage:

        if response.usage is None:
            return Usage()
        else:
            details: dict[str, int] = {}

            if response.usage.completion_tokens_details is not None:
                details.update(
                    response.usage.completion_tokens_details.model_dump(
                        exclude_none=True
                    )
                )

            if response.usage.prompt_tokens_details is not None:
                details.update(
                    response.usage.prompt_tokens_details.model_dump(exclude_none=True)
                )

            return Usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                details=details,
            )

    # def get_completion(
    #     self,
    #     chat_history: list[BaseMessagePart],
    #     response_format: BaseModel | None = None,
    #     tools: list[Tool] | None = None,
    # ) -> IterationResponse:

    #     if len(chat_history) == 0:
    #         raise SkyAgentDetrimentalError("chat_history cannot be an empty array!")

    #     try:

    #         messages = []
    #         for message in chat_history:
    #             if isinstance(message, OutgoingMessage):

    #                 if message.attached_images:
    #                     content = []

    #                     if message.content:
    #                         content.append({"type": "text", "text": message.content})

    #                     for image in message.attached_images:
    #                         content.append(
    #                             {
    #                                 "type": "image_url",
    #                                 "image_url": {
    #                                     "url": f"data:image/jpeg;base64,{image.base_64}"
    #                                 },
    #                             }
    #                         )

    #                     messages.append(
    #                         {"role": message.role.value, "content": content}
    #                     )
    #                 else:
    #                     messages.append(
    #                         {"role": message.role.value, "content": message.content}
    #                     )
    #             elif isinstance(message, SystemMessage):
    #                 messages.append(
    #                     {"role": message.role.value, "content": message.content}
    #                 )
    #             elif isinstance(message, ToolCallOutgoingMessage):
    #                 messages.append(
    #                     {
    #                         "role": message.role.value,
    #                         "content": message.content,
    #                         "tool_call_id": message.tool_call_id,
    #                     }
    #                 )
    #             else:
    #                 # we keep incoming tool calls
    #                 messages.append(message)

    #         completion_args = self._build_completion_arguments(
    #             messages, tools, response_format
    #         )

    #         if response_format is None:
    #             response: ChatCompletion = self.client.chat.completions.create(
    #                 **completion_args
    #             )
    #         else:
    #             response: ChatCompletion = self.client.beta.chat.completions.parse(
    #                 **completion_args
    #             )

    #     except Exception as e:
    #         raise SkyAgentDetrimentalError(f"Chat completion failed: '{e}'")

    #     finish_reason = response.choices[0].finish_reason

    #     if finish_reason == "length":
    #         raise SkyAgentContextWindowSaturatedError("Context window exceeded!")

    #     if finish_reason == "content_filter":
    #         raise SkyAgentCopyrightError("Query was filtered due to copyright reasons!")

    #     usage = ApiUsage(
    #         output_tokens=response.usage.completion_tokens,
    #         input_tokens=response.usage.prompt_tokens,
    #     )

    #     if finish_reason == "tool_calls":

    #         # gpt requires the tool call sent by him to be appended
    #         chat_history.append(response.choices[0].message.to_dict())

    #         parsed_tool_calls = [
    #             ToolCall(
    #                 call_id=tool_call.id,
    #                 function_name=tool_call.function.name,
    #                 arguments=ast.literal_eval(tool_call.function.arguments),
    #             )
    #             for tool_call in response.choices[0].message.tool_calls
    #         ]

    #         return IterationResponse(
    #             content=None, tool_calls=parsed_tool_calls, usage=usage
    #         )

    #     elif finish_reason == "stop":
    #         return IterationResponse(
    #             content=response.choices[0].message.content,
    #             tool_calls=None,
    #             usage=usage,
    #         )
    #     else:
    #         raise SkyAgentDetrimentalError(
    #             f"OpenAI API returned with an unexpected finish reason: '{
    #                 finish_reason}'"
    #         )

    # def convert_tool_result_answer(
    #     self, tool_call_result: ToolCallResult
    # ) -> ToolCallOutgoingMessage:

    #     content_dict = {
    #         "input": tool_call_result.arguments,
    #         "output": tool_call_result.result,
    #     }

    #     return ToolCallOutgoingMessage(
    #         content=json.dumps(content_dict), tool_call_id=tool_call_result.id
    #     )

    # def tool_to_dict(self, tool: Tool) -> dict:

    #     properties_dict = {
    #         param.param_name: {
    #             "type": param.param_type,
    #             "description": param.param_description,
    #         }
    #         for param in tool.parameters
    #     }

    #     return {
    #         "type": "function",
    #         "function": {
    #             "name": tool._tool_name,
    #             "description": tool.description,
    #             "strict": True,
    #             "parameters": {
    #                 "type": "object",
    #                 "properties": properties_dict,
    #                 "required": tool.required_properties,
    #                 "additionalProperties": tool._additional_properties,
    #             },
    #         },
    #     }

    # def _build_completion_arguments(
    #     self,
    #     messages: list,
    #     tools: list[Tool] | None = None,
    #     response_format: BaseModel | None = None,
    # ) -> dict:
    #     """
    #     Build completion arguments for OpenAI API.

    #     Args:
    #         messages: Formatted chat messages
    #         tools: Optional list of tools
    #         response_format: Optional response format specification

    #     Returns:
    #         dict: Arguments for OpenAI API completion call
    #     """

    #     completion_args = {
    #         "model": self.model,
    #         "messages": messages,
    #     }

    #     if tools:
    #         completion_args["tools"] = [self.tool_to_dict(tool) for tool in tools]

    #     if response_format is not None:
    #         completion_args["response_format"] = response_format

    #     if self.model_extra_args:
    #         completion_args.update(self.model_extra_args)

    #     return completion_args
