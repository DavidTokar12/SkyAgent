from __future__ import annotations

from enum import Enum

from rich.console import Console
from rich.table import Table


class AgentStatus(Enum):
    INITIALIZING = "Initializing..."
    CALLING_LLM = "Calling LLM..."
    EXECUTING_TOOL_CALLS = "Executing tool calls..."

    FINISHED = "Finished : )"
    FAILED = "Failed : ("


class AgentLogger:
    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        model: str,
        parallelized_execution_enabled: bool,
    ):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.model = model
        self.parallelized_execution_enabled = parallelized_execution_enabled

        self.total_tool_calls = 0
        self.executed_tool_calls = 0
        self.pending_tool_calls = {}

        self.completion_tokens_used = 0
        self.prompt_tokens_used = 0

        self.console = Console()
        self.status = "Starting..."

    def started_executing_tool_call(
        self, tool_call_id: str, is_async: bool = False, is_compute_heavy_bool=False
    ) -> None:
        pass

    def finished_executing_tool_call(self, tool_call_id: str) -> None:
        pass

    def add_token_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens_used += prompt_tokens
        self.completion_tokens_used += completion_tokens

    def set_status(self, status: AgentStatus) -> None:
        self.status = status

    def render_dashboard(self) -> Table:
        """
        Create one table with columns for each piece of information so it lines up nicely.
        """
        table = Table(expand=True, title=f"Agent: {self.agent_name}")

        table.add_column("Status", justify="left")
        table.add_column("ID", justify="left")
        table.add_column("Total Calls", justify="center")
        table.add_column("Executed Calls", justify="center")
        table.add_column("Prompt Tokens", justify="center")
        table.add_column("Completion Tokens", justify="center")

        table.add_row(
            f"[green]{self.status.ljust(28)}[/green]",
            f"[magenta]{self.agent_id}[/magenta]",
            str(self.total_tool_calls),
            str(self.executed_tool_calls),
            str(self.prompt_tokens_used),
            str(self.completion_tokens_used),
        )

        return table


#     def run(self):
#         with Live(self.render_dashboard(), refresh_per_second=4) as live:
#             for _ in range(self.total_tool_calls):
#                 import time
#                 import random
#                 time.sleep(1.5)
#                 # Simulate tool execution
#                 self.update_tool_calls()
#                 self.update_tokens(
#                     prompt_tokens=random.randint(50, 100),
#                     completion_tokens=random.randint(150, 250)
#                 )
#                 # Random status updates
#                 self.set_status(
#                     random.choice(
#                         ["Executing...", "Waiting for LLM response...", "Idle"])
#                 )
#                 live.update(self.render_dashboard())

#             # Final update: Completed
#             self.set_status("Complete")
#             live.update(self.render_dashboard())
#             time.sleep(2)


# if __name__ == "__main__":

#     agent_logger = AgentLogger(
#         agent_id="12345",
#         agent_name="OpenAIAgent",
#         model="gpt-4",
#         parallelized_execution_enabled=True
#     )
#     agent_logger.run()
