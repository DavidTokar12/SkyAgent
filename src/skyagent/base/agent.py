from __future__ import annotations

import asyncio
import uuid

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path
from typing import TYPE_CHECKING

from skyagent.base.chat_message import SystemChatMessage
from skyagent.base.exceptions import SkyAgentDetrimentalError
from skyagent.base.logger import AgentLogger
from skyagent.base.tools import ToolCallResult
from skyagent.base.utils import get_or_create_event_loop


if TYPE_CHECKING:
    from skyagent.base.chat_message import BaseChatMessage
    from skyagent.base.llm_api_adapter import CompletionResponse
    from skyagent.base.tools import TOOL_ARG_TYPES
    from skyagent.base.tools import BaseTool
    from skyagent.base.tools import ToolCall


def run_async_in_task(async_func, *args, **kwargs):
    """Wrapper to run an async function in a synchronous way."""

    return asyncio.run(async_func(*args, **kwargs))


class BaseAgent:

    def __init__(
        self,
        name: str,
        model: str,
        system_prompt: str | Path,
        tools: list[BaseTool] | None,
        max_turns: int = 10,
        token: str | None = None,
        parallelize: bool = True,
        num_processes: int = 4,
        temperature: float = 0.0,
        timeout: int = 3,
        log_file_path: Path | None = None,
        log_server: str | None = None,
        enable_live_display: bool = False,
    ):
        self.agent_id = str(uuid.uuid4())
        self.name = name
        self.model = model
        self.token = token
        self.temperature = temperature
        self.timeout = timeout
        self.log_file_path = log_file_path
        self.log_server = log_server
        self.enable_live_display = enable_live_display

        if isinstance(system_prompt, Path):
            if not system_prompt.exists():
                raise FileNotFoundError(
                    "System prompt file not found: '%s'", system_prompt
                )
            with open(system_prompt) as f:
                self.system_prompt = f.read()
        else:
            self.system_prompt = system_prompt

        self.tools_array = tools
        self.tools_dict = {tool.name: tool for tool in tools} if tools else None

        self.max_turns = max_turns

        self.chat_history: list[BaseChatMessage] = [
            SystemChatMessage(content=self.system_prompt)
        ]

        # Concurrency-related
        self.parallelize = parallelize
        self.num_processes = num_processes

        self.process_executor = ProcessPoolExecutor(
            max_workers=num_processes if parallelize else 1
        )

        self.logger = AgentLogger(
            agent_id=self.agent_id,
            agent_name=self.name,
            agent_model=self.model,
            agent_parallelize=self.parallelize,
            agent_chat_history=self.chat_history,
            agent_tools=self.tools_array,
            log_file_path=self.log_file_path,
            log_server=self.log_server,
            enable_live_display=self.enable_live_display,
        )

        self.logger.initialized_agent()

    def call(self, query: str) -> CompletionResponse:
        if self.enable_live_display:
            with self.logger.live_dashboard():
                return self._call_implementation(query=query)
        return self._call_implementation(query=query)

    def _call_implementation(self, query: str) -> CompletionResponse:
        raise NotImplementedError(
            "The _call_implementation method must be implemented!"
        )

    def execute_tool_calls(self, tool_calls: list[ToolCall]) -> list[TOOL_ARG_TYPES]:

        inline_tool_calls = []
        async_tool_calls = []
        compute_heavy_tool_calls = []

        for tool_call in tool_calls:
            function_name = tool_call.function_name

            tool_wrapper = self.tools_dict.get(function_name)

            if not tool_wrapper:
                raise SkyAgentDetrimentalError(
                    f"Tool '{tool_call.function_name}' not found in the agent named '{
                        self.name}' tool definitions.",
                )

            if tool_wrapper.is_compute_heavy:
                compute_heavy_tool_calls.append(tool_call)
            elif tool_wrapper.is_async:
                async_tool_calls.append(tool_call)
            else:
                inline_tool_calls.append(tool_call)

        inline_results = self.execute_inline_tool_calls(tool_calls=inline_tool_calls)
        compute_heavy_results = self.execute_compute_heavy_tool_calls(
            tool_calls=compute_heavy_tool_calls
        )
        async_results = self.execute_async_tool_calls(tool_calls=async_tool_calls)

        return inline_results + compute_heavy_results + async_results

    def parse_tool_call_arguments(
        self, tool_call: ToolCall
    ) -> dict[str, TOOL_ARG_TYPES]:
        tool_wrapper = self.tools_dict.get(tool_call.function_name)

        if not tool_wrapper:
            raise SkyAgentDetrimentalError(
                "Tool '{tool_call.function_name}' not found in the agent named '{self.name}' tool definitions.",
            )

        converted_args = {}
        for param_name, param_value in tool_call.arguments.items():
            converted_args[param_name] = self.tools_dict[
                tool_call.function_name
            ].validate_and_convert_input_param(
                input_param_name=param_name, input_param_value=param_value
            )

        return converted_args

    def execute_inline_tool_calls(
        self, tool_calls: list[ToolCall]
    ) -> list[ToolCallResult]:

        results = []

        for tool_call in tool_calls:

            self.logger.started_executing_tool_call(
                tool_call, is_async=False, is_compute_heavy=False
            )

            tool_wrapper = self.tools_dict.get(tool_call.function_name)
            parsed_arguments = self.parse_tool_call_arguments(tool_call=tool_call)

            try:

                result_value = tool_wrapper.tool_function(**parsed_arguments)
                tool_call_result = ToolCallResult(
                    id=tool_call.id,
                    function_name=tool_call.function_name,
                    arguments=tool_call.arguments,
                    result=result_value,
                )

                self.logger.finished_executing_tool_call(
                    tool_call_result=tool_call_result
                )

                results.append(tool_call_result)

            except Exception as e:
                raise SkyAgentDetrimentalError(
                    f"Error executing tool '{tool_call.function_name}': '{e}'"
                )

        return results

    def execute_compute_heavy_tool_calls(
        self, tool_calls: list[ToolCall]
    ) -> list[ToolCallResult]:

        results = []

        if not tool_calls:
            return results

        with self.process_executor as executor:

            futures = {}
            for tool_call in tool_calls:

                tool_wrapper = self.tools_dict.get(tool_call.function_name)

                self.logger.started_executing_tool_call(
                    tool_call,
                    is_async=tool_wrapper.is_async,
                    is_compute_heavy=True,
                )

                if not tool_wrapper:
                    raise SkyAgentDetrimentalError(
                        f"Tool {tool_call.function_name} not found in the agent named {
                            self.name} tool definitions.",
                    )

                parsed_arguments = self.parse_tool_call_arguments(tool_call=tool_call)

                if tool_wrapper.is_async:
                    futures[
                        executor.submit(
                            run_async_in_task,
                            tool_wrapper.tool_function,
                            **parsed_arguments,
                        )
                    ] = tool_call
                else:
                    futures[
                        executor.submit(tool_wrapper.tool_function, **parsed_arguments)
                    ] = tool_call

            for future in as_completed(futures):

                tool_call = futures[future]

                try:
                    result_value = future.result()
                    tool_call_result = ToolCallResult(
                        id=tool_call.id,
                        function_name=tool_call.function_name,
                        arguments=tool_call.arguments,
                        result=result_value,
                    )

                    self.logger.finished_executing_tool_call(
                        tool_call_result=tool_call_result
                    )

                    results.append(tool_call_result)
                except Exception as e:
                    raise SkyAgentDetrimentalError(
                        f"Tool execution failed with error: {e}"
                    )

        return results

    def execute_async_tool_calls(
        self, tool_calls: list[ToolCall]
    ) -> list[ToolCallResult]:
        results = []
        if not tool_calls:
            return results

        loop = get_or_create_event_loop()

        if self.parallelize:
            tasks = []
            for tool_call in tool_calls:
                tool_wrapper = self.tools_dict.get(tool_call.function_name)

                if not tool_wrapper:
                    raise SkyAgentDetrimentalError(
                        f"Tool '{tool_call.function_name}' not found in agent '{
                            self.name}' definitions."
                    )

                self.logger.started_executing_tool_call(
                    tool_call, is_async=True, is_compute_heavy=False
                )

                parsed_arguments = self.parse_tool_call_arguments(tool_call=tool_call)
                tasks.append(
                    loop.create_task(tool_wrapper.tool_function(**parsed_arguments))
                )

            all_results = loop.run_until_complete(
                asyncio.gather(*tasks, return_exceptions=True)
            )

            for tool_call, result_value_or_exc in zip(tool_calls, all_results):
                if isinstance(result_value_or_exc, Exception):
                    raise SkyAgentDetrimentalError(
                        f"Async tool '{tool_call.function_name}' execution failed: {
                            result_value_or_exc}"
                    )

                result = ToolCallResult(
                    id=tool_call.id,
                    function_name=tool_call.function_name,
                    arguments=tool_call.arguments,
                    result=result_value_or_exc,
                )

                self.logger.finished_executing_tool_call(result)

                results.append(result)
        else:
            for tool_call in tool_calls:
                tool_wrapper = self.tools_dict.get(tool_call.function_name)

                if not tool_wrapper:
                    raise SkyAgentDetrimentalError(
                        f"Tool '{tool_call.function_name}' not found in agent '{
                            self.name}' definitions."
                    )

                self.logger.started_executing_tool_call(
                    tool_call, is_async=True, is_compute_heavy=False
                )

                parsed_arguments = self.parse_tool_call_arguments(tool_call=tool_call)
                try:
                    result = loop.run_until_complete(
                        tool_wrapper.tool_function(**parsed_arguments)
                    )
                    result = ToolCallResult(
                        id=tool_call.id,
                        function_name=tool_call.function_name,
                        arguments=tool_call.arguments,
                        result=result,
                    )

                    self.logger.finished_executing_tool_call(result)

                    results.append(result)
                except Exception as e:
                    raise SkyAgentDetrimentalError(
                        f"Async tool '{
                            tool_call.function_name}' execution failed: {e}"
                    )

        return results
