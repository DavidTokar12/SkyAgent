from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel


if TYPE_CHECKING:
    from pathlib import Path


class ShellInteractionInput(BaseModel):
    type: str
    content: str


class ShellInteractionOutput(BaseModel):
    type: str
    content: str
    state: str


class InteractionHistory(BaseModel):
    input_to_environment: ShellInteractionInput
    output_from_environment: ShellInteractionOutput


class EnvironmentAdapter:

    def __init__(self, log_file_path: str | Path | None = None):

        self.log_file_path = log_file_path
        self.interaction_history: list[InteractionHistory] = []

        self.log_file = (
            open(log_file_path, "w", encoding="utf-8")  # noqa: SIM115
            if log_file_path
            else None
        )

    def get_tool_functions(self) -> list[callable]:
        raise NotImplementedError("'get_tool_functions' method must be implemented")

    def __enter__(self) -> EnvironmentAdapter:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.log_file:
            self.log_file.close()
