from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from pydantic import BaseModel


if TYPE_CHECKING:
    from pathlib import Path


class InteractionHistory(BaseModel):
    input_to_environment: Any
    output_from_environment: Any


class EnvironmentAdapter:

    def __init__(self, log_file_path: str | Path | None = None):

        self.log_file_path = log_file_path
        self.interaction_history: list[InteractionHistory] = []

        self.log_file = (
            open(log_file_path, "w", encoding="utf-8")  # noqa: SIM115
            if log_file_path
            else None
        )

    def interact(self, environment_input):
        raise NotImplementedError("'interact' method must be implemented")

    def __del__(self):
        if self.log_file:
            self.log_file.close()


# adapter = UnixShellAdapter(
    # log_file_path="/workspaces/SkyAgent/src/skyagent/base/xd.log")

# print(adapter.interact("ls"))
