from __future__ import annotations

import asyncio
import time
import uuid

from typing import TYPE_CHECKING

from skyagent.exceptions import SkyAgentDetrimentalError
from skyagent.exceptions import SkyAgentTypeError
from skyagent.function_executor import FunctionCall
from skyagent.function_executor import FunctionExecutor
from skyagent.function_executor import FunctionResult
from skyagent.messages import ModelInput
from skyagent.messages import ModelOutput
from skyagent.messages import SystemPrompt
from skyagent.messages import UserPrompt
from skyagent.providers.provider import Provider

# from skyagent.providers.provider import IterationResponse
from skyagent.providers.provider_registry import ProviderRegistry
from skyagent.usage import Usage


if TYPE_CHECKING:
    from typing import dataclass

    from pydantic import BaseModel

    from skyagent.tool import Tool


class Agent:

    def __init__(
        self,
        model: str,
        provider: str | type[Provider],
        name: str | None = None,
        system_prompt: SystemPrompt | None = None,
        tools: list[Tool] | None = None,
        max_iterations: int = 10,
        max_retries: int = 1,
        model_settings: dataclass | None = None,
        client_settings: dataclass | None = None,
    ):
        """
        Initialize an Agent that manages tool-using interactions with an LLM.

        Args:
            model: Identifier for the specific LLM to use (e.g., "gpt-4-turbo", "claude-3-opus")
            provider: API provider to use - either a string identifier for built-in providers ("openai", "anthropic", etc.) or a custom ApiAdapter class
            name: Human-readable name for the agent (used in logging and debugging)
            system_prompt: Initial system instructions for the LLM.
            tools: List of Tool instances the agent can call during execution
            max_iterations: Maximum number of tool-calling iterations before terminating (prevents infinite loops)
            max_retries: Number of retries when the LLM fails to follow the expected output format
            model_settings: Additional configuration parameters passed directly to the LLM.
            client_kwargs: Additional configuration for the API client connection.
        """

        self._agent_id = str(uuid.uuid4())

        self._name = name
        self._model = model
        self._system_prompt = system_prompt

        self._tools = tools
        self._tool_name_mapping = {tool.name: tool for tool in tools} if tools else {}

        self._max_iterations = max_iterations
        self._max_retries = max_retries
        self._model_settings = model_settings
        self._client_settings = client_settings

        self._function_executor = FunctionExecutor()

        if isinstance(provider, str):
            provider_class = ProviderRegistry.get_provider_class(provider)
        elif isinstance(provider, type) and issubclass(provider, Provider):
            provider_class = provider
        else:
            raise SkyAgentTypeError(
                "Provider must be either a string identifier for a registered provider (openai, anthropic...) "
                "or a subclass of Provider (not an instance)"
            )

        self._provider = provider_class(
            model=self._model,
            model_settings=self._model_settings,
            client_settings=self._client_settings,
        )

        self._chat_history: list[ModelInput | ModelOutput] | None = None

    async def run(
        self,
        query: str | None = None,
        input_chat_history: list[ModelInput | ModelOutput] | None = None,
        result_format: BaseModel | None = None,
        deps: dataclass | None = None,
        usage_limits: Usage | None = None,
    ):
        _usage = Usage()

        _start_time = time.time()

        self._chat_history = self._prepare_chat_history(query, input_chat_history)

        try:

            for _current_turn in range(1, self._max_iterations + 1):

                iteration_response = await self._provider.run_iteration(
                    chat_history=self._chat_history,
                    result_format=result_format,
                    tools=self._tools,
                )

                _usage.add(iteration_response.usage)

                print(iteration_response)

                if iteration_response.tool_calls:

                    parsed_calls = []

                    for tool_call in iteration_response.tool_calls:

                        if tool_call.function_name not in self._tool_name_mapping:
                            # TODO - handle with correction message
                            raise SkyAgentDetrimentalError(
                                f"Tool '{tool_call.function_name}' not found in agent's tool list."
                            )

                        tool_result = None

                        try:
                            tool = self._tool_name_mapping[tool_call.function_name]
                            tool.validate_args(tool_call.arguments)

                            function_call = FunctionCall(
                                function=tool._tool_function,
                                function_name=tool_call.function_name,
                                arguments=tool_call.arguments,
                                call_id=tool_call.call_id,
                                compute_heavy=tool._is_compute_heavy,
                            )

                            parsed_calls.append(function_call)

                        except Exception as e:
                            # TODO - handle with correction message
                            raise SkyAgentDetrimentalError(
                                f"Tool '{tool_call.function_name}' failed to validate arguments: {e}"
                            )

                    function_results = await self._function_executor.execute_all(
                        parsed_calls
                    )

                    print(function_results)

                if iteration_response.content:
                    # append to chat history
                    pass

                break

        except SkyAgentDetrimentalError as e:
            # self._logger.log_error(e)
            raise e

    def run_sync(
        self,
        input_chat_history: list[ModelInput | ModelOutput],
        result_format: BaseModel | None = None,
        dependencies: BaseModel | None = None,
        usage: None = None,
        usage_limits: None = None,
    ):
        return asyncio.get_event_loop().run_until_complete(
            self.run(
                input_chat_history=input_chat_history,
                result_format=result_format,
                dependencies=dependencies,
                usage=usage,
                usage_limits=usage_limits,
            )
        )

    async def run_stream(
        self,
        input_chat_history: list[ModelInput | ModelOutput],
        result_format: BaseModel | None = None,
        dependencies: BaseModel | None = None,
        usage: None = None,
        usage_limits: None = None,
    ):
        pass

    def _prepare_chat_history(
        self,
        query: str | None,
        input_chat_history: list[ModelInput | ModelOutput] | None,
    ) -> list[ModelInput | ModelOutput]:
        """
        Prepare chat history ensuring proper formatting with system prompt if available.

        Args:
            query: User query string if no chat history is provided
            input_chat_history: Optional existing chat history

        Returns:
            Properly formatted chat history
        """

        if query is None and input_chat_history is None:
            raise SkyAgentDetrimentalError(
                "One of 'input' or 'input_chat_history' has to be provided."
            )

        if query is not None and input_chat_history is not None:
            raise SkyAgentDetrimentalError(
                "Only one of 'input' or 'input_chat_history' can be provided."
            )

        if input_chat_history is not None:
            if len(input_chat_history) == 0:
                raise SkyAgentDetrimentalError(
                    "input_chat_history cannot be an empty array!"
                )
            if not all(
                isinstance(msg, ModelInput | ModelOutput) for msg in input_chat_history
            ):
                raise SkyAgentTypeError(
                    "input_chat_history can only contain OutgoingMessage or IncomingMessage instances."
                )

        # If no system prompt available
        if self._system_prompt is None:
            if input_chat_history is not None:
                return list(input_chat_history)
            else:
                return [ModelInput(message_parts=[UserPrompt(content=query)])]

        # System prompt exists, handle chat history
        if input_chat_history is not None:
            chat_history = list(input_chat_history)

            if isinstance(chat_history[0], ModelInput):
                # Check if already has system prompt
                if len(chat_history[0].message_parts) > 0 and isinstance(
                    chat_history[0].message_parts[0], SystemPrompt
                ):
                    return chat_history

                # No system prompt, add it
                chat_history[0].message_parts.insert(0, self._system_prompt)
                return chat_history
            else:
                # First message is not OutgoingMessage, prepend one with system prompt
                return [
                    ModelInput(message_parts=[self._system_prompt]),
                    *chat_history,
                ]

        # Handle query-only case with system prompt
        return [
            ModelInput(message_parts=[self._system_prompt, UserPrompt(content=query)])
        ]
