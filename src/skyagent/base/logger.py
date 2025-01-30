from __future__ import annotations

import logging
import traceback

from contextlib import contextmanager
from enum import Enum
from logging import FileHandler
from logging import StreamHandler
from logging.handlers import HTTPHandler
from typing import TYPE_CHECKING

from rich.console import Console
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


if TYPE_CHECKING:
    from pathlib import Path

    from skyagent.base.chat_message import BaseChatMessage
    from skyagent.base.llm_api_adapter import CompletionResponse
    from skyagent.base.tools import BaseTool


class AgentStatus(Enum):
    INITIALIZING = "Initializing..."
    READY = "Ready..."

    CALLING_LLM = "Calling LLM..."
    EXECUTING_TOOL_CALLS = "Executing tool calls..."

    FINISHED = "Finished : )"
    FAILED = "Failed : ("


class AgentLogger:
    """Logger class to track agent status, activity, and provide optional live dashboards."""

    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        agent_model: str,
        agent_parallelize: bool,
        agent_chat_history: list[BaseChatMessage],
        agent_tools: list[BaseTool],
        log_file_path: Path | None = None,
        log_server: str | None = None,
        enable_live_display: bool = False,
    ):
        """
        Initializes an AgentLogger instance.

        Args:
            agent_id (str): Unique ID for the agent.
            agent_name (str): Name of the agent.
            agent_model (str): Model name the the agent uses.
            agent_parallelize (bool): Indicates if the agent runs tasks in parallel.
            agent_chat_history (list[BaseChatMessage]): List of the agent's message history.
            agent_tools (list[BaseTool]): List of tools available to the agent.

            log_file_path (Path | None, optional): File path for saving logs. Defaults to None.
            log_server (str | None, optional): Server address for HTTP logging. Defaults to None.
            enable_live_display (bool, optional): If True, shows a live console dashboard. Defaults to False.
        """
        # Basic fields
        self._agent_id = agent_id
        self._agent_name = agent_name
        self._agent_model = agent_model
        self._agent_parallelize = agent_parallelize
        self._agent_tools = agent_tools
        self._agent_chat_history = agent_chat_history
        self._agent_chat_loop_index = 0
        self._agent_final_result = None
        self._pending_tool_calls = 0
        self._executed_tool_calls = 0
        self._pending_tool_calls_dict = {}

        self._completion_tokens_used = 0
        self._prompt_tokens_used = 0

        self._status: AgentStatus = AgentStatus.INITIALIZING

        self._log_file_path = log_file_path
        self._log_server = log_server
        self._enable_live_display = enable_live_display

        self._console = Console() if enable_live_display else None
        self._live = None

        self._configure_logger(
            logs_file_path=self._log_file_path, log_server=self._log_server
        )

    def _configure_logger(
        self, logs_file_path: Path | None, log_server: str | None
    ) -> None:
        """
        Configures the internal Python logger with file, HTTP, or console handlers.

        Args:
            logs_file_path (Path | None): File path for writing log output.
            log_server (str | None): Server address for HTTP log submission.
        """

        self._logger = logging.getLogger(f"agent.{self._agent_id}")
        self._logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s - Agent: %(agent_name)s - ID: %(agent_id)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        if logs_file_path:
            file_handler = FileHandler(logs_file_path)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.DEBUG)
            self._logger.addHandler(file_handler)

        if log_server:
            http_handler = HTTPHandler(log_server, "/log", method="POST")
            http_handler.setFormatter(formatter)
            http_handler.setLevel(logging.INFO)
            self._logger.addHandler(http_handler)

        if not self._enable_live_display:
            console_handler = StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(logging.DEBUG)
            self._logger.addHandler(console_handler)

        extra = {
            "agent_id": self._agent_id,
            "agent_name": self._agent_name,
            "agent_model": self._agent_model,
        }
        self._logger = logging.LoggerAdapter(self._logger, extra)

    @contextmanager
    def live_dashboard_context(self):
        """
        Context manager that sets up the live Rich dashboard if enabled.
        Yields the live display context, allowing updates to the console.
        """
        if self._enable_live_display:
            if not self._console:
                self._console = Console()
            with Live(
                self.render_live_dashboard(),
                console=self._console,
            ) as live:
                self._live = live
                yield
                self._live = None
        else:
            yield

    def update_live_dashboard(self):
        """
        Updates the Live dashboard with current agent information.
        Should be called whenever the agent's state changes and needs re-rendering.
        """
        if self._live:
            self._live.update(self.render_live_dashboard())

    def log_agent_initialized(self) -> None:
        """
        Logs that the agent has finished initializing.
        Sets the agent's status to READY.
        """

        self._status = AgentStatus.READY
        self._logger.debug(
            "Initialized Agent with model '%s', tool definitions: '%s', "
            "parallelize=%s",
            self._agent_model,
            [tool.name for tool in self._agent_tools] if self._agent_tools else [],
            self._agent_parallelize,
        )

    def log_query_received(self, query: str) -> None:
        """
        Logs that the agent received a query from a user or external source.

        Args:
            query (str): The query or request the agent received.
        """

        self._status = AgentStatus.CALLING_LLM
        self._logger.debug("Agent received query: '%s'", query)
        self.update_live_dashboard()

    def log_chat_loop_started(self, turn: int) -> None:
        """
        Logs that a chat or conversation loop has begun for a given turn number.

        Args:
            turn (int): The turn number in the conversation loop.
        """

        self._agent_chat_loop_index = turn
        self._logger.debug("Response loop '%s' starting.", turn)
        self.update_live_dashboard()

    def log_final_completion(
        self, completion: CompletionResponse, execution_time: float
    ):
        """
        Logs the agent's final completion response from the LLM.

        Args:
            completion (CompletionResponse): The completion response data.
            execution_time (float): Time in seconds the call took.
        """

        self._logger.debug("Final chat history: '%s'", self._agent_chat_history)
        self._logger.debug("Received final completion from model: '%s'", completion)
        self._logger.debug(
            "Agent call successfully completed in: '%s' seconds.", execution_time
        )

        self._status = AgentStatus.FINISHED
        self._agent_final_result = completion.content
        self._completion_tokens_used += completion.usage.completion_tokens
        self._prompt_tokens_used += completion.usage.prompt_tokens
        self.update_live_dashboard()

    def log_tool_calls_received(self, tool_calls):
        """
        Logs that the agent has received one or more tool calls from the model.

        Args:
            tool_calls: A collection or list of tool call objects.
        """
        self._logger.debug("Received tool calls from model: '%s'", tool_calls)

        self._status = AgentStatus.EXECUTING_TOOL_CALLS
        self._pending_tool_calls += len(tool_calls)
        self.update_live_dashboard()

    def started_executing_tool_call(
        self, tool_call, is_async: bool = False, is_compute_heavy=False
    ) -> None:
        """
        Logs the start or submission of a tool call's execution.

        Args:
            tool_call: The tool call object being executed.
            is_async (bool, optional): If True, indicates asynchronous execution. Defaults to False.
            is_compute_heavy (bool, optional): If True, indicates the call is compute-heavy. Defaults to False.
        """

        self._pending_tool_calls_dict[tool_call.id] = {
            **tool_call.model_dump(),
            "compute_heavy": is_compute_heavy,
            "async": is_async,
        }

        self._logger.debug(
            "%s%s%s: '%s'",
            (
                "Submitted tool call for execution"
                if is_compute_heavy or is_async
                else "Started executing tool call"
            ),
            " (compute-heavy)" if is_compute_heavy else "",
            " (async)" if is_async else "",
            tool_call,
        )

        self.update_live_dashboard()

    def log_tool_call_finished(self, tool_call_result) -> None:
        """
        Logs the completion of a tool call and updates counters.

        Args:
            tool_call_result: The result object from the completed tool call.
        """

        self._pending_tool_calls_dict.pop(tool_call_result.id, None)

        self._logger.debug(
            "Finished executing tool call with result: '%s'", tool_call_result
        )

        self._executed_tool_calls += 1
        self._pending_tool_calls -= 1

        if self._pending_tool_calls == 0:
            self._status = AgentStatus.CALLING_LLM

        self.update_live_dashboard()

    def log_error(self, error: Exception):
        """
        Logs an error that occurred during the agent's operation.

        Args:
            error (Exception): The exception that was raised.
        """
        self._logger.error(
            "Error occurred: %s\nTraceback: %s", error, traceback.format_exc()
        )
        # Don't update live dashboard, so the last thing on the CLI is the error message.

    def render_live_dashboard(self):
        if not self._console:
            self._console = Console()

        table = Table(expand=True, title=f"Agent: {self._agent_name}")

        table.add_column("Status", justify="left")
        table.add_column("ID", justify="left")
        table.add_column("Pending Tool Calls", justify="center")
        table.add_column("Executed Tool Calls", justify="center")
        table.add_column("Prompt Tokens", justify="center")
        table.add_column("Completion Tokens", justify="center")

        table.add_row(
            f"[green]{self._status.value}[/green]",
            f"[magenta]{self._agent_id}[/magenta]",
            str(self._pending_tool_calls),
            str(self._executed_tool_calls),
            str(self._prompt_tokens_used),
            str(self._completion_tokens_used),
        )

        chat_history_str = "\n".join(str(msg) for msg in self._agent_chat_history[-4:])
        chat_history_panel = Panel(
            chat_history_str,
            title="Recent Chat History (Last 3)",
            expand=True,
            border_style="blue",
        )

        pending_calls_str = "\n".join(
            str(tool_data) for tool_data in self._pending_tool_calls_dict.values()
        )
        pending_calls_panel = Panel(
            pending_calls_str,
            title="Executing Tool Calls",
            expand=True,
            border_style="magenta",
        )

        renderables = [table, chat_history_panel, pending_calls_panel]

        if self._agent_final_result is not None:
            final_result_panel = Panel(
                f"[bold green]{self._agent_final_result}[/bold green]",
                title="[green]Final Result[/green]",
                expand=True,
                border_style="green",
            )
            renderables.append(final_result_panel)

        return Group(*renderables)
