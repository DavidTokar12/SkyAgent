from __future__ import annotations

import logging
import traceback

from logging import FileHandler
from logging import StreamHandler
from typing import TYPE_CHECKING

from skyagent.base.loggers.base_agent_logger import BaseAgentLogger


if TYPE_CHECKING:
    from pathlib import Path

    from skyagent.base.api_adapters.api_adapter import CompletionResponse
    from skyagent.base.chat_message import _BaseMessage
    from skyagent.base.tools import Tool


class AgentLogger(BaseAgentLogger):
    """Standard logger implementation with file and HTTP logging capabilities."""

    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        agent_model: str,
        agent_chat_history: list[_BaseMessage],
        agent_tools: list[Tool],
        log_file_path: Path | None = None,
    ):
        self._log_file_path = log_file_path

        super().__init__(
            agent_id=agent_id,
            agent_name=agent_name,
            agent_model=agent_model,
            agent_chat_history=agent_chat_history,
            agent_tools=agent_tools,
        )

    def _initialize_logger(self) -> None:
        """Configure the internal Python logger with handlers."""
        self._logger = logging.getLogger(f"agent.{self._agent_id}")
        self._logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s - Agent: %(agent_name)s - ID: %(agent_id)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Always add console handler
        console_handler = StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.DEBUG)
        self._logger.addHandler(console_handler)

        # Add file handler if path provided
        if self._log_file_path:
            file_handler = FileHandler(self._log_file_path)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.DEBUG)
            self._logger.addHandler(file_handler)

        extra = {
            "agent_id": self._agent_id,
            "agent_name": self._agent_name,
            "agent_model": self._agent_model,
        }
        self._logger = logging.LoggerAdapter(self._logger, extra)

    def _log_agent_initialized_impl(self) -> None:
        self._logger.debug(
            "Initialized Agent with model '%s', tool definitions: '%s'",
            self._agent_model,
            [tool.name for tool in self._agent_tools] if self._agent_tools else [],
        )

    def _log_input_chat_history_received_impl(
        self, input_chat_history: list[_BaseMessage]
    ) -> None:
        self._logger.debug(
            "Agent received input chat history: '%s'", input_chat_history
        )

    def _log_chat_loop_started_impl(self, turn: int) -> None:
        self._logger.debug("Response loop '%s' starting.", turn)

    def _log_final_completion_impl(
        self, completion: CompletionResponse, execution_time: float
    ) -> None:
        self._logger.debug("Final chat history: '%s'", self._agent_chat_history)
        self._logger.debug("Received final completion from model: '%s'", completion)
        self._logger.debug(
            "Agent call successfully completed in: '%s' seconds.", execution_time
        )

    def _log_tool_calls_received_impl(self, tool_calls) -> None:
        self._logger.debug("Received tool calls from model: '%s'", tool_calls)

    def _log_tool_call_started_impl(
        self, tool_call, is_async: bool = False, is_compute_heavy: bool = False
    ) -> None:
        execution_type = (
            "Submitted tool call for execution"
            if is_compute_heavy or is_async
            else "Started executing tool call"
        )
        self._logger.debug(
            "%s%s%s: '%s'",
            execution_type,
            " (compute-heavy)" if is_compute_heavy else "",
            " (async)" if is_async else "",
            tool_call,
        )

    def _log_tool_call_finished_impl(self, tool_call_result) -> None:
        self._logger.debug(
            "Finished executing tool call with result: '%s'", tool_call_result
        )

    def _log_error_impl(self, error: Exception) -> None:
        self._logger.error(
            "Error occurred: %s\nTraceback: %s", error, traceback.format_exc()
        )
