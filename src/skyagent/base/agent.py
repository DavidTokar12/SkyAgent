from __future__ import annotations

import asyncio
import time
import uuid

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING

from skyagent.base.api_adapters.api_adapter import ApiAdapter
from skyagent.base.api_adapters.api_adapter import ApiUsage
from skyagent.base.api_adapters.api_adapter import CompletionResponse
from skyagent.base.api_adapters.api_adapter_registry import ApiRegistry
from skyagent.base.chat_message import AssistantMessage
from skyagent.base.exceptions import SkyAgentDetrimentalError
from skyagent.base.exceptions import SkyAgentTypeError
from skyagent.base.loggers.agent_logger import AgentLogger
from skyagent.base.tools import ToolCallResult


if TYPE_CHECKING:
    from pydantic import BaseModel

    from skyagent.base.chat_message import _BaseMessage
    from skyagent.base.chat_message import SystemMessage
    from skyagent.base.loggers.base_agent_logger import BaseAgentLogger
    from skyagent.base.tools import TOOL_ARG_TYPES
    from skyagent.base.tools import Tool
    from skyagent.base.tools import ToolCall


@dataclass
class _PreparedToolCall:
    tool_call: ToolCall
    tool_wrapper: Tool
    parsed_arguments: dict[str, TOOL_ARG_TYPES]


class Agent:

    def __init__(
        self,
        agent_name: str,
        model: str,
        api_adapter: str | type[ApiAdapter],
        system_prompt: SystemMessage | None = None,
        max_turns: int = 10,
        tools: list[Tool] | None = None,
        logger: type[BaseAgentLogger] | None = None,
        logger_extra_args: dict | None = None,
        token: str | None = None,
        num_processes: int = 4,
        timeout: int = 10,
        model_extra_args: dict | None = None,
        client_extra_args: dict | None = None,
    ):
        """
        Initializes a new instance of an Agent, that can handle tool call loops from an LLM.

        Args:
            agent_name (str): Human-readable name for the agent.
            model (str): The LLM model name/identifier.
            api_adapter (str | type[ApiAdapter]): The API adapter to use for the agent. Use one of the predefined adapters('openai', 'anthropic'...) or a custom one.
            system_prompt (SystemChatMessage | None, optional):  The system prompt instructing the agent.
            tools (list[BaseTool] | None): Optional list of tools the agent can use.
            max_turns (int, optional): Max conversation turns before completion. Defaults to 10.
            logger (type[BaseAgentLogger] | None, optional): Optional logger class for the agent. Defaults to None, in which case the base terminal logger will be used.
            logger_extra_args (dict | None, optional): Optional extra arguments for the logger. Defaults to None.
            token (str | None, optional): API token or authentication credentials for the LLM. Defaults to None.
            num_processes (int, optional): Number of worker processes for heavy or concurrent tasks. Defaults to 4.
            timeout (int, optional): Timeout in seconds for LLM queries. Defaults to 10.
            model_extra_args (dict | None, optional): Optional extra arguments for the LLM model. Defaults to None.
            client_extra_args (dict | None, optional): Optional extra arguments for the LLM Api client. Defaults to None.
        """

        self._agent_id = str(uuid.uuid4())

        self._agent_name = agent_name
        self._model = model
        self._system_prompt = system_prompt
        self._token = token
        self._timeout = timeout
        self._model_extra_args = model_extra_args
        self._client_extra_args = client_extra_args
        self._max_turns = max_turns
        self._num_processes = num_processes

        if isinstance(api_adapter, str):
            adapter_class = ApiRegistry.get_adapter_class(api_adapter)
        elif isinstance(api_adapter, type) and issubclass(api_adapter, ApiAdapter):
            adapter_class = api_adapter
        else:
            raise SkyAgentTypeError(
                "'api_adapter' must be either a string identifier or an ApiAdapter class(not instance)"
            )

        self._client = adapter_class(
            model=self._model,
            token=self._token,
            timeout=self._timeout,
            model_extra_args=self._model_extra_args,
            client_extra_args=self._client_extra_args,
        )

        self._tools_array = tools
        self._tools_dict = {tool.name: tool for tool in tools} if tools else {}

        self.chat_history: list[_BaseMessage] = [system_prompt] if system_prompt else []

        logger = logger if logger else AgentLogger
        self._logger = logger(
            agent_id=self._agent_id,
            agent_name=self._agent_name,
            agent_model=self._model,
            agent_chat_history=self.chat_history,
            agent_tools=self._tools_array,
            **(logger_extra_args if logger_extra_args else {}),
        )

        self._logger.log_agent_initialized()

    async def call_agent(
        self,
        input_chat_history: list[_BaseMessage],
        response_format: BaseModel | None = None,
    ) -> CompletionResponse:
        """
        Entry point to invoke the agent.

        Args:
            input_chat_history (list[BaseChatMessage]): The input chat history.
            response_format (BaseModel | None, optional): Optional for validated structured output. Defaults to None.

        Returns:
            CompletionResponse: The final complete LLM response.
        """

        return await self._execute_agent_call(
            input_chat_history=input_chat_history, response_format=response_format
        )

    def call_agent_sync(
        self,
        input_chat_history: list[_BaseMessage],
        response_format: BaseModel | None = None,
    ) -> CompletionResponse:
        """
        Entry point to invoke the agent synchronously.

        Args:
            input_chat_history (list[BaseChatMessage]): The user-provided input chat history.
            response_format (BaseModel | None, optional): Optional for validated structured output. Defaults to None.

        Returns:
            CompletionResponse: The final complete LLM response.
        """

        async def _run():
            return await self._execute_agent_call(
                input_chat_history=input_chat_history, response_format=response_format
            )

        return asyncio.get_event_loop().run_until_complete(_run())

    async def _execute_agent_call(
        self,
        input_chat_history: list[_BaseMessage],
        response_format: BaseModel | None = None,
    ) -> CompletionResponse:

        total_input_usage = 0
        total_output_usage = 0

        start_time = time.time()

        self._logger.log_input_chat_history_received(
            input_chat_history=input_chat_history
        )

        # Chat history is used by logger, so append instead of reassigning
        for message in input_chat_history:
            self.chat_history.append(message)

        try:

            for current_turn in range(1, self._max_turns + 1):

                self._logger.log_chat_loop_started(turn=current_turn)

                completion = self._client.get_completion(
                    chat_history=self.chat_history,
                    tools=self._tools_array,
                    response_format=response_format,
                )

                total_input_usage += completion.usage.input_tokens
                total_output_usage += completion.usage.output_tokens

                if completion.tool_calls:

                    self._logger.log_tool_calls_received(completion.tool_calls)

                    tool_call_results = await self._execute_all_tool_calls(
                        completion.tool_calls
                    )

                    for tool_call_result in tool_call_results:
                        tool_result_answer = self._client.convert_tool_result_answer(
                            tool_call_result=tool_call_result
                        )
                        self.chat_history.append(tool_result_answer)

                else:
                    self.chat_history.append(
                        AssistantMessage(content=completion.content)
                    )

                    execution_time = time.time() - start_time

                    self._logger.log_final_completion(
                        completion=completion, execution_time=execution_time
                    )

                    return CompletionResponse(
                        content=completion.content,
                        tool_calls=None,
                        usage=completion.usage,
                    )

            self._logger.log_error(
                SkyAgentDetrimentalError("Max turns reached with no final completion.")
            )

            return CompletionResponse(
                content=None,
                tool_calls=None,
                usage=ApiUsage(
                    output_tokens=total_output_usage, input_tokens=total_input_usage
                ),
            )

        except SkyAgentDetrimentalError as e:
            self._logger.log_error(e)
            raise e

    def _prepare_tool_call(self, tool_call: ToolCall) -> _PreparedToolCall:
        """Prepares a tool call by ensuring the tool exists and converting its arguments."""

        tool_wrapper = self._tools_dict.get(tool_call.function_name)

        if not tool_wrapper:
            raise SkyAgentDetrimentalError(
                f"Tool '{tool_call.function_name}' not found in agent '{self._agent_name}' definitions."
            )

        parsed_arguments = {}
        for param_name, param_value in tool_call.arguments.items():
            parsed_arguments[param_name] = (
                tool_wrapper.validate_and_convert_input_param(
                    input_param_name=param_name, input_param_value=param_value
                )
            )

        return _PreparedToolCall(
            tool_call=tool_call,
            tool_wrapper=tool_wrapper,
            parsed_arguments=parsed_arguments,
        )

    async def _execute_all_tool_calls(
        self, tool_calls: list[ToolCall]
    ) -> list[ToolCallResult]:
        """
        Prepares and routes tool calls into inline, async, or compute-heavy methods.
        All heavy operations (async and compute-heavy) run concurrently.
        """
        prepared_calls = [self._prepare_tool_call(tc) for tc in tool_calls]

        inline_calls = []
        async_calls = []
        compute_heavy_calls = []

        for prepared in prepared_calls:
            if prepared.tool_wrapper.is_async:
                async_calls.append(prepared)
            elif prepared.tool_wrapper.is_compute_heavy:
                compute_heavy_calls.append(prepared)
            else:
                inline_calls.append(prepared)

        inline_results = self._execute_inline_tool_calls(inline_calls)

        try:
            async with asyncio.timeout(self._timeout):
                async_task = asyncio.create_task(
                    self._execute_async_tool_calls(async_calls)
                )
                compute_heavy_task = asyncio.create_task(
                    asyncio.to_thread(
                        self._execute_compute_heavy_tool_calls, compute_heavy_calls
                    )
                )

                async_results, compute_heavy_results = await asyncio.gather(
                    async_task, compute_heavy_task, return_exceptions=True
                )

                for result in (async_results, compute_heavy_results):
                    if isinstance(result, Exception):
                        raise result

        except asyncio.TimeoutError as e:
            if not async_task.done():
                async_task.cancel()
            if not compute_heavy_task.done():
                compute_heavy_task.cancel()

            raise SkyAgentDetrimentalError(
                "Timeout while executing async and compute-heavy tools"
            ) from e

        except Exception as e:
            raise SkyAgentDetrimentalError(
                "Error executing async or compute-heavy tools"
            ) from e

        return [*inline_results, *compute_heavy_results, *async_results]

    def _execute_inline_tool_calls(
        self, prepared_calls: list[_PreparedToolCall]
    ) -> list[ToolCallResult]:
        """Execute a list of inline (synchronous) tool calls sequentially."""

        results = []
        for prepared in prepared_calls:
            try:
                self._logger.log_tool_call_started(
                    prepared.tool_call, is_async=False, is_compute_heavy=False
                )

                result_value = prepared.tool_wrapper.tool_function(
                    **prepared.parsed_arguments
                )

                tool_call_result = ToolCallResult(
                    id=prepared.tool_call.id,
                    function_name=prepared.tool_call.function_name,
                    arguments=prepared.tool_call.arguments,
                    result=result_value,
                )

                self._logger.log_tool_call_finished(tool_call_result)
                results.append(tool_call_result)

            except Exception as e:
                raise SkyAgentDetrimentalError(
                    f"Tool '{prepared.tool_call.function_name}' execution failed"
                ) from e

        return results

    async def _execute_async_tool_calls(
        self, prepared_calls: list[_PreparedToolCall]
    ) -> list[ToolCallResult]:
        """Execute all async tools concurrently using TaskGroup."""
        if not prepared_calls:
            return []

        try:
            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(self._execute_single_async_tool(prepared))
                    for prepared in prepared_calls
                ]

            return [task.result() for task in tasks]

        except* asyncio.TimeoutError as eg:

            failed_tools = [
                prepared_calls[i].tool_call.function_name
                for i, exc in enumerate(eg.exceptions)
                if isinstance(exc, asyncio.TimeoutError)
            ]

            raise SkyAgentDetrimentalError(
                f"Tools timed out: {', '.join(failed_tools)}"
            ) from eg

        except* Exception as eg:
            for exc in eg.exceptions:
                self._logger.error("Async tool execution failed", exc_info=exc)
            raise SkyAgentDetrimentalError(
                "Multiple async tool executions failed"
            ) from eg

    async def _execute_single_async_tool(
        self,
        prepared: _PreparedToolCall,
    ) -> ToolCallResult:
        """Execute a single async tool with proper error handling."""
        try:

            self._logger.log_tool_call_started(
                prepared.tool_call, is_async=True, is_compute_heavy=False
            )

            async with asyncio.timeout(self._timeout):
                result_value = await prepared.tool_wrapper.tool_function(
                    **prepared.parsed_arguments
                )

            tool_call_result = ToolCallResult(
                id=prepared.tool_call.id,
                function_name=prepared.tool_call.function_name,
                arguments=prepared.tool_call.arguments,
                result=result_value,
            )

            self._logger.log_tool_call_finished(tool_call_result)
            return tool_call_result

        except asyncio.TimeoutError as e:
            raise SkyAgentDetrimentalError(
                f"Tool '{prepared.tool_call.function_name}' timed out after {self._timeout}s"
            ) from e

        except Exception as e:
            raise SkyAgentDetrimentalError(
                f"Tool '{prepared.tool_call.function_name}' execution failed"
            ) from e

    def _execute_compute_heavy_tool_calls(
        self, prepared_calls: list[_PreparedToolCall]
    ) -> list[ToolCallResult]:
        """Execute all compute-heavy calls in process pool."""

        if not prepared_calls:
            return []

        results = []

        with ProcessPoolExecutor(max_workers=self._num_processes) as executor:
            try:
                futures = {}
                for prepared in prepared_calls:
                    self._logger.log_tool_call_started(
                        prepared.tool_call,
                        is_async=False,
                        is_compute_heavy=True,
                    )
                    future = executor.submit(
                        prepared.tool_wrapper.tool_function, **prepared.parsed_arguments
                    )
                    futures[future] = prepared

                for future in as_completed(futures):
                    prepared = futures[future]
                    try:
                        result_value = future.result(timeout=self._timeout)
                        tool_call_result = ToolCallResult(
                            id=prepared.tool_call.id,
                            function_name=prepared.tool_call.function_name,
                            arguments=prepared.tool_call.arguments,
                            result=result_value,
                        )
                        self._logger.log_tool_call_finished(tool_call_result)
                        results.append(tool_call_result)
                    except TimeoutError as e:
                        # Cancel remaining futures on timeout
                        for f in futures:
                            f.cancel()
                        raise SkyAgentDetrimentalError(
                            f"Tool '{prepared.tool_call.function_name}' timed out after {self._timeout}s"
                        ) from e
                    except Exception as e:
                        raise SkyAgentDetrimentalError(
                            f"Tool '{prepared.tool_call.function_name}' execution failed"
                        ) from e
            finally:
                # Ensure we clean up the executor
                executor.shutdown(wait=False, cancel_futures=True)

        return results
