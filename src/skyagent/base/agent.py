from __future__ import annotations

import asyncio
import time
import uuid

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path
from typing import TYPE_CHECKING

from skyagent.base.chat_message import AssistantChatMessage
from skyagent.base.chat_message import ChatMessageRole
from skyagent.base.chat_message import SystemChatMessage
from skyagent.base.chat_message import UserChatMessage
from skyagent.base.exceptions import SkyAgentDetrimentalError
from skyagent.base.logger import AgentLogger
from skyagent.base.tools import ToolCallResult
from skyagent.base.utils import get_or_create_event_loop


if TYPE_CHECKING:
    from pydantic import BaseModel

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
        tools: list[BaseTool] | None = None,
        max_turns: int = 10,
        token: str | None = None,
        parallelize: bool = True,
        num_processes: int = 4,
        temperature: float = 0.0,
        timeout: int = 10,
        log_file_path: Path | None = None,
        log_server: str | None = None,
        enable_live_display: bool = False,
    ):
        """
        Initializes a new instance of Agent..

        Args:
            name (str): Human-readable name for the agent.
            model (str): The LLM model name/identifier.
            system_prompt (str | Path): The system-level prompt or path to file containing it.
            tools (list[BaseTool] | None): Optional list of tools the agent can use.
            max_turns (int, optional): Max conversation turns before completion. Defaults to 10.
            token (str | None, optional): API token or authentication credentials for the LLM. Defaults to None.
            parallelize (bool, optional): If True, allows concurrent processing of tool calls. Defaults to True.
            num_processes (int, optional): Number of worker processes for heavy or concurrent tasks. Defaults to 4.
            temperature (float, optional): LLM temperature setting. Defaults to 0.0.
            timeout (int, optional): Timeout in seconds for LLM queries. Defaults to 3.
            log_file_path (Path | None, optional): Path for log file output. Defaults to None.
            log_server (str | None, optional): Server address for HTTP logging. Defaults to None.
            enable_live_display (bool, optional): If True, shows a Rich live logging dashboard. Defaults to False.
        """
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
                    f"System prompt file not found: '{system_prompt}'"
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

        self._initialize_client()

        self.logger.log_agent_initialized()

    def _initialize_client(self):
        """
        Initializes or configures the LLM client.

        Raises:
            NotImplementedError: Must be overridden by child classes for specific LLM setup.
        """
        raise NotImplementedError(
            "Client initialization must be implemented by agent child classes."
        )

    def call_agent(
        self, query: str, response_format: BaseModel | None = None
    ) -> CompletionResponse:
        """
        Entry point to invoke the agent with a given query.

        Args:
            query (str): The user-provided query or request.
            response_format (BaseModel | None, optional): Optional for validated structured output. Defaults to None.

        Returns:
            CompletionResponse: The final LLM response, if successfully completed.
        """
        if self.enable_live_display:
            with self.logger.live_dashboard_context():
                return self._execute_agent_call(
                    query=query, response_format=response_format
                )
        return self._execute_agent_call(query=query, response_format=response_format)

    def _execute_agent_call(
        self, query: str, response_format: BaseModel | None = None
    ) -> CompletionResponse:
        """
        Handles the main call logic, orchestrating the conversation loop and tool usage.

        Args:
            query (str): The user or external request to process.
            response_format (BaseModel | None, optional): Optional for validated structured output. Defaults to None.

        Returns:
            CompletionResponse: The final LLM completion, if available.

        Raises:
            SkyAgentDetrimentalError: If there is a fatal error during execution.
        """
        try:
            self.logger.log_query_received(query=query)

            start_time = time.time()

            self.chat_history.append(
                UserChatMessage(role=ChatMessageRole.user, content=query)
            )

            for current_turn in range(1, self.max_turns + 1, 1):

                self.logger.log_chat_loop_started(turn=current_turn)

                completion = self.client.get_completion(
                    chat_history=self.chat_history,
                    tools=self.tools_array,
                    response_format=response_format,
                )

                if completion.tool_calls:

                    self.logger.log_tool_calls_received(completion.tool_calls)

                    tool_call_results = self._execute_all_tool_calls(
                        completion.tool_calls
                    )

                    for tool_call_result in tool_call_results:
                        tool_result_answer = self.client.convert_tool_result_answer(
                            tool_call_result=tool_call_result
                        )
                        self.chat_history.append(tool_result_answer)

                else:
                    self.chat_history.append(
                        AssistantChatMessage(content=completion.content)
                    )

                    execution_time = time.time() - start_time

                    self.logger.log_final_completion(
                        completion=completion, execution_time=execution_time
                    )

                    return completion

            self.logger.log_error(
                SkyAgentDetrimentalError("Max turns reached with no final completion.")
            )
            # TODO also return a completion, with content None, and usage information
            return None
        except SkyAgentDetrimentalError as e:
            self.logger.log_error(e)

    def _execute_all_tool_calls(
        self, tool_calls: list[ToolCall]
    ) -> list[ToolCallResult]:
        """
        Routes tool calls into inline, async, or compute-heavy methods and collects the results.

        Args:
            tool_calls (list[ToolCall]): A batch of tool calls requested by the model.

        Returns:
            list[ToolCallResult]: The aggregated results of all tool calls.
        """
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

        inline_results = self._execute_inline_tool_calls(tool_calls=inline_tool_calls)
        compute_heavy_results = self._execute_compute_heavy_tool_calls(
            tool_calls=compute_heavy_tool_calls
        )
        async_results = self._execute_async_tool_calls(tool_calls=async_tool_calls)

        return inline_results + compute_heavy_results + async_results

    def _parse_tool_call_args(self, tool_call: ToolCall) -> dict[str, TOOL_ARG_TYPES]:
        """
        Converts and validates raw tool call arguments into typed forms required by the tool.

        Args:
            tool_call (ToolCall): The requested tool call details.

        Returns:
            dict[str, TOOL_ARG_TYPES]: A dictionary of validated and converted argument values.

        Raises:
            SkyAgentDetrimentalError: If the tool does not exist in the agent's tool dictionary.
        """

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

    def _execute_inline_tool_calls(
        self, tool_calls: list[ToolCall]
    ) -> list[ToolCallResult]:
        """
        Executes a list of inline (synchronous) tool calls sequentially.

        Args:
            tool_calls (list[ToolCall]): Tool calls that are neither async nor compute-heavy.

        Returns:
            list[ToolCallResult]: A list of results from inline tool execution.
        """

        results = []

        for tool_call in tool_calls:

            self.logger.started_executing_tool_call(
                tool_call, is_async=False, is_compute_heavy=False
            )

            tool_wrapper = self.tools_dict.get(tool_call.function_name)
            parsed_arguments = self._parse_tool_call_args(tool_call=tool_call)

            try:

                result_value = tool_wrapper.tool_function(**parsed_arguments)
                tool_call_result = ToolCallResult(
                    id=tool_call.id,
                    function_name=tool_call.function_name,
                    arguments=tool_call.arguments,
                    result=result_value,
                )

                self.logger.log_tool_call_finished(tool_call_result=tool_call_result)

                results.append(tool_call_result)

            except Exception as e:
                raise SkyAgentDetrimentalError(
                    f"Error executing tool '{tool_call.function_name}': '{e}'"
                )

        return results

    def _execute_compute_heavy_tool_calls(
        self, tool_calls: list[ToolCall]
    ) -> list[ToolCallResult]:
        """
        Executes compute-heavy tool calls in a ProcessPoolExecutor for parallelism.

        Args:
            tool_calls (list[ToolCall]): Tool calls marked as compute-heavy.

        Returns:
            list[ToolCallResult]: A list of results from compute-heavy executions.

        Raises:
            SkyAgentDetrimentalError: If a tool call fails or a tool is not found.
        """
        results = []

        if not tool_calls:
            return results

        with ProcessPoolExecutor(
            max_workers=self.num_processes if self.parallelize else 1
        ) as executor:

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

                parsed_arguments = self._parse_tool_call_args(tool_call=tool_call)

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

                    self.logger.log_tool_call_finished(
                        tool_call_result=tool_call_result
                    )

                    results.append(tool_call_result)
                except Exception as e:
                    raise SkyAgentDetrimentalError(
                        f"Tool execution failed with error: {e}"
                    )

        return results

    def _execute_async_tool_calls(
        self, tool_calls: list[ToolCall]
    ) -> list[ToolCallResult]:
        """
        Executes asynchronous tool calls using an event loop.

        Args:
            tool_calls (list[ToolCall]): Tool calls requiring async execution.

        Returns:
            list[ToolCallResult]: A list of results from async tool execution.

        Raises:
            SkyAgentDetrimentalError: If a tool call fails or a tool is not found.
        """
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

                parsed_arguments = self._parse_tool_call_args(tool_call=tool_call)
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

                self.logger.log_tool_call_finished(result)

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

                parsed_arguments = self._parse_tool_call_args(tool_call=tool_call)
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

                    self.logger.log_tool_call_finished(result)

                    results.append(result)
                except Exception as e:
                    raise SkyAgentDetrimentalError(
                        f"Async tool '{
                            tool_call.function_name}' execution failed: {e}"
                    )

        return results
