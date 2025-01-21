from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import pexpect

from pydantic import BaseModel

from skyagent.base.environment_interactor import EnvironmentAdapter
from skyagent.base.exceptions import SkyAgentDetrimentalError


if TYPE_CHECKING:
    from pathlib import Path


class UnixShellInteractionState(Enum):
    FINISHED = "finished"  # Command finished
    TIMEOUT = "timeout"  # Execution stopped due to a long timeout
    # Execution possibly interrupted by interactive input or similar event
    INTERRUPTED = "interrupted"


class UnixShellInteractionResult(BaseModel):
    output: str
    state: str


class UnixShellAdapter(EnvironmentAdapter):

    def __init__(
        self,
        log_file_path: str | Path | None = None,
        shell_path: str = "/bin/bash",
        timeout: int = 10 * 60,
        prompt: str = "$ ",
        encoding: str = "utf-8",
    ):

        super().__init__(log_file_path=log_file_path)

        self.shell_path = shell_path
        self.timeout = timeout
        self.prompt = prompt
        self.encoding = encoding

        self.shell = pexpect.spawn(
            self.shell_path,
            encoding=self.encoding,
            timeout=self.timeout,
            echo=False,
            logfile=self.log_file,
        )

        try:
            self.shell.expect_exact(self.prompt, timeout=3)
        except pexpect.TIMEOUT as e:
            raise SkyAgentDetrimentalError(f"Failed spawning shell: {e}")

    def interact(self, environment_input: str) -> dict:
        """
        Use this tool to interact with a Unix shell environment. This tool spawns a single, persistent shell instance where you can execute commands.

        **Caution:** You must only try to run a single shell command with a single call to this tool. If you start to chain commands, the tool will not work as expected.

        It supports longer and interactive commands, returning partial results when necessary, allowing for user interaction or cancellation.

        The tool returns the command's output, and a state value.

        The execution state can be:
        *   **finished:** Command finished.
        *   **timeout:** Command execution exceeded the time limit (default: 10 minutes).
        *   **interrupted:** Command execution was interrupted, providing a partial result. You can continue (by not calling the tool again), cancel, or interact further.

        :param environment_input: The command to execute.

        :return: The result of the interaction, including command output and metadata for subsequent commands.
        """

        output = None

        max_loops = int(self.timeout // 3)

        self.shell.sendline(environment_input)

        for _ in range(max_loops):
            try:
                self.shell.expect_exact(self.prompt, timeout=3)

                output = self._format_command_output(self.shell.before)
                state = UnixShellInteractionState.FINISHED.value

                return UnixShellInteractionResult(
                    output=output,
                    state=state,
                ).model_dump()
            except pexpect.TIMEOUT:
                print("TIMEOUT")

    def _run_basic_command(self, command: str) -> str:
        self.shell.sendline(command)
        self.shell.expect_exact(self.prompt)

        pure_result = self.shell.before
        return self._format_command_output(pure_result)

    def _format_command_output(self, raw_output: str) -> str:
        """
        Format the raw output of a command output, by removing it's first and last lines.
        """
        lines = raw_output.splitlines()

        if len(lines) <= 2:
            return raw_output.strip()

        return "\n".join(lines[1:-1]).strip()

    def __del__(self):
        super().__del__()
        self.shell.close()
