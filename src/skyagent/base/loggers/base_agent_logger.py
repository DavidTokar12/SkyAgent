from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from enum import Enum
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from skyagent.base.api_adapters.api_adapter import CompletionResponse
    from skyagent.base.chat_message import _BaseMessage
    from skyagent.base.tools import Tool


class AgentStatus(Enum):
    INITIALIZING = "Initializing..."
    READY = "Ready..."
    CALLING_LLM = "Calling LLM..."
    EXECUTING_TOOL_CALLS = "Executing tool calls..."
    FINISHED = "Finished : )"
    FAILED = "Failed : ("


class BaseAgentLogger(ABC):
    """Abstract base class defining the interface for agent loggers."""

    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        agent_model: str,
        agent_chat_history: list[_BaseMessage],
        agent_tools: list[Tool],
    ):
        self._agent_id = agent_id
        self._agent_name = agent_name
        self._agent_model = agent_model
        self._agent_chat_history = agent_chat_history
        self._agent_tools = agent_tools

        self._status = AgentStatus.INITIALIZING
        self._agent_chat_loop_index = 0
        self._completion_tokens_used = 0
        self._prompt_tokens_used = 0
        self._pending_tool_calls = 0
        self._executed_tool_calls = 0

        self._initialize_logger()

    @abstractmethod
    def _initialize_logger(self) -> None:
        """Implementation-specific logger initialization."""

    def log_agent_initialized(self) -> None:
        """Log that the agent has finished initializing."""
        self._status = AgentStatus.READY
        self._log_agent_initialized_impl()

    @abstractmethod
    def _log_agent_initialized_impl(self) -> None:
        """Log that the agent has finished initializing."""

    def log_input_chat_history_received(
        self, input_chat_history: list[_BaseMessage]
    ) -> None:
        """Log that the agent received a chat history input."""
        self._status = AgentStatus.CALLING_LLM
        self._log_input_chat_history_received_impl(input_chat_history)

    @abstractmethod
    def _log_input_chat_history_received_impl(
        self, input_chat_history: list[_BaseMessage]
    ) -> None:
        """Log that the agent received a chat history input."""

    def log_chat_loop_started(self, turn: int) -> None:
        """Log that a chat loop has begun for a given turn."""
        self._agent_chat_loop_index = turn
        self._log_chat_loop_started_impl(turn)

    @abstractmethod
    def _log_chat_loop_started_impl(self, turn: int) -> None:
        """Log that a chat loop has begun for a given turn."""

    def log_final_completion(
        self, completion: CompletionResponse, execution_time: float
    ) -> None:
        """Log the agent's final completion response."""

        self._status = AgentStatus.FINISHED
        self._completion_tokens_used += completion.usage.output_tokens
        self._prompt_tokens_used += completion.usage.input_tokens

        self._log_final_completion_impl(completion, execution_time)

    @abstractmethod
    def _log_final_completion_impl(
        self, completion: CompletionResponse, execution_time: float
    ) -> None:
        """Log the agent's final completion response."""

    def log_tool_calls_received(self, tool_calls) -> None:
        """Log that tool calls were received from the model."""
        self._status = AgentStatus.EXECUTING_TOOL_CALLS
        self._pending_tool_calls += len(tool_calls)
        self._log_tool_calls_received_impl(tool_calls)

    @abstractmethod
    def _log_tool_calls_received_impl(self, tool_calls) -> None:
        """Log that tool calls were received from the model."""

    def log_tool_call_finished(self, tool_call_result) -> None:
        """Log the completion of a tool call."""
        self._executed_tool_calls += 1
        self._pending_tool_calls -= 1

        if self._pending_tool_calls == 0:
            self._status = AgentStatus.CALLING_LLM

        self._log_tool_call_finished_impl(tool_call_result)

    @abstractmethod
    def _log_tool_call_finished_impl(self, tool_call_result) -> None:
        """Log the completion of a tool call."""

    def log_error(self, error: Exception) -> None:
        """Log an error that occurred during operation."""
        self._status = AgentStatus.FAILED
        self._log_error_impl(error)

    @abstractmethod
    def _log_error_impl(self, error: Exception) -> None:
        """Log an error that occurred during operation."""

    def log_tool_call_started(
        self, tool_call, is_async: bool = False, is_compute_heavy: bool = False
    ) -> None:
        """Log when a tool call starts execution."""
        self._log_tool_call_started_impl(tool_call, is_async, is_compute_heavy)

    @abstractmethod
    def _log_tool_call_started_impl(
        self, tool_call, is_async: bool, is_compute_heavy: bool
    ) -> None:
        pass
