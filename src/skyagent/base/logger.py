from __future__ import annotations

import logging

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
        # Basic fields
        self._agent_id = agent_id
        self._agent_name = agent_name
        self._agent_model = agent_model
        self._agent_parallelize = agent_parallelize
        self._agent_tools = agent_tools
        self._agent_chat_history = agent_chat_history
        self._agent_chat_loop_index = 0

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

        self._setup_logger(
            logs_file_path=self._log_file_path, log_server=self._log_server
        )

    def _setup_logger(
        self, logs_file_path: Path | None, log_server: str | None
    ) -> None:
        """
        Creates a logger, attaches file handlers, HTTP handlers (if any), and optionally
        a console (stream) handler if live logging is disabled.
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
    def live_dashboard(self):

        if self._enable_live_display:
            if not self._console:
                self._console = Console()
            with Live(
                self.render_dashboard(),
                console=self._console,
            ) as live:
                self._live = live
                yield
                self._live = None
        else:
            yield

    def refresh_live_display(self):
        """Call this method whenever you want to update the Live display."""
        if self._live:
            self._live.update(self.render_dashboard())

    def initialized_agent(self) -> None:
        self._status = AgentStatus.READY
        self._logger.debug(
            "Initialized Agent with model '%s', tool definitions: '%s', "
            "parallelize=%s",
            self._agent_model,
            [tool.name for tool in self._agent_tools] if self._agent_tools else [],
            self._agent_parallelize,
        )

    def query_received(self, query: str) -> None:
        self._status = AgentStatus.CALLING_LLM
        self._logger.debug("Agent received query: '%s'", query)
        self.refresh_live_display()

    def chat_loop_started(self, turn: int) -> None:
        self._agent_chat_loop_index = turn
        self._logger.debug("Response loop '%s' starting.", turn)
        self.refresh_live_display()

    def final_completion_received_from_server(
        self, completion: CompletionResponse, execution_time: float
    ):
        self._logger.debug("Received final completion from model: '%s'", completion)
        self._logger.debug(
            "Agent call successfully completed in: '%s' seconds.", execution_time
        )
        self._logger.debug("Chat history: '%s'", self._agent_chat_history)

        self._status = AgentStatus.FINISHED
        self._completion_tokens_used += completion.usage.completion_tokens
        self._prompt_tokens_used += completion.usage.prompt_tokens
        self.refresh_live_display()

    def tool_calls_received_from_server(self, tool_calls):
        self._logger.debug("Received tool calls from model: '%a'", tool_calls)

        self._status = AgentStatus.EXECUTING_TOOL_CALLS
        self._pending_tool_calls += len(tool_calls)
        self.refresh_live_display()

    def started_executing_tool_call(
        self, tool_call, is_async: bool = False, is_compute_heavy=False
    ) -> None:
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
        self.refresh_live_display()

    def finished_executing_tool_call(self, tool_call_result) -> None:
        self._pending_tool_calls_dict.pop(tool_call_result.id, None)
        self._logger.debug(
            "Finished executing tool call with result: '%s'", tool_call_result
        )

        self._executed_tool_calls += 1
        self._pending_tool_calls -= 1

        if self._pending_tool_calls == 0:
            self._status = AgentStatus.CALLING_LLM

        self.refresh_live_display()

    def error_happened(self, error: Exception):
        self._logger.error(error)
        self.refresh_live_display()

    def render_dashboard(self) -> Table:
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

        chat_history_str = "\n".join(str(msg) for msg in self._agent_chat_history[-3:])
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

        return Group(table, chat_history_panel, pending_calls_panel)
