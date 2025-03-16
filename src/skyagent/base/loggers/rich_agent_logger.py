from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from rich.console import Console
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from skyagent.base.loggers.base_agent_logger import BaseAgentLogger


if TYPE_CHECKING:
    from skyagent.base.api_adapters.api_adapter import CompletionResponse
    from skyagent.base.chat_message import _BaseMessage
    from skyagent.base.tools import Tool


class RichAgentLogger(BaseAgentLogger):
    """Rich console logger implementation with live dashboard capabilities."""

    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        agent_model: str,
        agent_chat_history: list[_BaseMessage],
        agent_tools: list[Tool],
    ):
        self._console = Console()
        self._live = None
        self._agent_final_result = None
        self._pending_tool_calls_dict = {}

        super().__init__(
            agent_id=agent_id,
            agent_name=agent_name,
            agent_model=agent_model,
            agent_chat_history=agent_chat_history,
            agent_tools=agent_tools,
        )

    def _initialize_logger(self) -> None:
        """No initialization needed for rich logger."""

    @contextmanager
    def live_dashboard_context(self):
        """Context manager that sets up the live Rich dashboard."""
        with Live(
            self.render_live_dashboard(),
            console=self._console,
        ) as live:
            self._live = live
            yield
            self._live = None

    def update_live_dashboard(self):
        """Updates the Live dashboard with current agent information."""
        if self._live:
            self._live.update(self.render_live_dashboard())

    def _log_agent_initialized_impl(self) -> None:
        self.update_live_dashboard()

    def _log_input_chat_history_received_impl(
        self, input_chat_history: list[_BaseMessage]
    ) -> None:
        self.update_live_dashboard()

    def _log_chat_loop_started_impl(self, turn: int) -> None:
        self.update_live_dashboard()

    def _log_final_completion_impl(
        self, completion: CompletionResponse, execution_time: float
    ) -> None:
        self._agent_final_result = completion.content
        self.update_live_dashboard()

    def _log_tool_calls_received_impl(self, tool_calls) -> None:
        self.update_live_dashboard()

    def _log_tool_call_started_impl(
        self, tool_call, is_async: bool = False, is_compute_heavy: bool = False
    ) -> None:
        self._pending_tool_calls_dict[tool_call.id] = {
            **tool_call.model_dump(),
            "compute_heavy": is_compute_heavy,
            "async": is_async,
        }
        self.update_live_dashboard()

    def _log_tool_call_finished_impl(self, tool_call_result) -> None:
        self._pending_tool_calls_dict.pop(tool_call_result.id, None)
        self.update_live_dashboard()

    def _log_error_impl(self, error: Exception) -> None:
        # Don't update dashboard on error so the error stays visible
        pass

    def render_live_dashboard(self):
        """Renders the current state of the dashboard."""
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
