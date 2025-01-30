from __future__ import annotations

import re

from enum import Enum
from pathlib import Path

import pexpect

from pydantic import BaseModel

from skyagent.base.environment_interactor import EnvironmentAdapter
from skyagent.base.exceptions import SkyAgentDetrimentalError


class UnixShellInteractionState(Enum):
    FINISHED = "finished"  # Command finished
    BUSY = "busy"  # Another command is running, cannot execute new command
    # Execution possibly interrupted by interactive input or similar event
    INTERRUPTED = "interrupted"


class UnixShellInteractionResult(BaseModel):
    output: str
    state: str


class UnixShellAdapter(EnvironmentAdapter):

    def __init__(
        self,
        base_dir: str | Path,
        log_file_path: str | Path | None = None,
        shell_path: str = "/bin/bash",
        timeout: int = 10 * 60,
        prompt: str = "$ ",
        encoding: str = "utf-8",
    ):

        super().__init__(log_file_path=log_file_path)

        self.base_dir = Path(base_dir)
        self.shell_path = shell_path
        self.timeout = timeout
        self.prompt = prompt
        self.encoding = encoding

        self.command_running = False

        if not self.base_dir.is_dir():
            raise SkyAgentDetrimentalError(f"Directory {self.base_dir} does not exist.")

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

        self.shell.sendline(f"cd {self.base_dir}")
        try:
            self.shell.expect_exact(self.prompt, timeout=3)
        except pexpect.TIMEOUT as e:
            raise SkyAgentDetrimentalError(
                f"Failed setting working directory to {self.base_dir}: {e}"
            )

    def get_tool_functions(self) -> list[callable]:
        return [
            self.run_command_in_shell,
            self.get_update_of_shell_output,
            self.send_control_signal,
        ]

    def run_command_in_shell(self, environment_input: str) -> dict:
        """
        Use this tool to run commands in a Unix shell. This tool spawns a single, persistent shell instance where you can execute commands.
        The tool will run your command with the pexpect sendline method, which appends os.linesep to your command automatically.

        **Caution:** You must only try to run a single shell command with a single call to this tool. If you start to chain commands, the tool might not work as expected.

        It supports longer and interactive commands, returning partial results when necessary, allowing for interaction or cancellation.

        The tool returns the command's output, and a state value.
        The execution state can be:
        *   **finished:** Command finished.
        *   **interrupted:** Command execution was interrupted, providing a partial result. You can use get_update_of_shell_output() to monitor the current command, or send a control signal to the shell to cancel it. You can't execute a new command before calling get_update_of_shell_output() or send_control_signal(), and ensuring the command is no longer running.
        *   **busy:** Another command is running, and you can't execute a new command until ensuring that the current command finishes. If you use the tools properly, you should never see this state.

        :param environment_input: The pure command to execute.
        """

        if self.command_running:
            return UnixShellInteractionResult(
                output="Cannot execute new command while previous command is still running. Use get_update_of_shell_output() to monitor the current command.",
                state=UnixShellInteractionState.BUSY.value,
            ).model_dump()

        self.command_running = True
        self.shell.sendline(environment_input)

        try:
            self.shell.expect_exact(self.prompt, timeout=3)
            output = self._format_command_output(self.shell.before, append_prompt=True)
            state = UnixShellInteractionState.FINISHED.value
            self.command_running = False
        except pexpect.TIMEOUT:
            output = self._format_command_output(self.shell.before)
            state = UnixShellInteractionState.INTERRUPTED.value

        return UnixShellInteractionResult(
            output=output,
            state=state,
        ).model_dump()

    def get_update_of_shell_output(self) -> dict:
        """
        Use this tool to get updates from a long-running shell command. This tool provides a way to monitor the progress and output of commands that take longer than the initial timeout period to complete.

        The tool repeatedly checks the shell's output buffer and returns both the current output and the command's state.

        The execution state in the response can be:
        *   **finished:** Command has completed execution. You can view the final output, and execute a new command.
        *   **interrupted:** Command is still running and producing output. Continue monitoring by calling this tool again.

        Example usage flow:
        1. Call run_command_in_shell() which returns 'interrupted' state.
        2. Call this tool repeatedly to monitor progress.
        3. When state returns 'finished', the command has completed.
        """

        if not self.command_running:
            return UnixShellInteractionResult(
                output=self._format_command_output(
                    self.shell.before, append_prompt=True
                ),
                state=UnixShellInteractionState.FINISHED.value,
            ).model_dump()

        try:
            index = self.shell.expect([self.prompt, pexpect.TIMEOUT], timeout=1)

            output = self._format_command_output(self.shell.before)

            if index == 0:
                state = UnixShellInteractionState.FINISHED.value
                self.command_running = False
            else:
                state = UnixShellInteractionState.INTERRUPTED.value

            return UnixShellInteractionResult(
                output=output,
                state=state,
            ).model_dump()

        except Exception as e:
            return UnixShellInteractionResult(
                output=self._format_command_output(self.shell.before),
                state=UnixShellInteractionState.INTERRUPTED.value,
            ).model_dump()

    def send_control_signal(self, signal: str) -> dict:
        """
        Use this tool to send a control signal to the shell. This tool is useful for interacting with running commands by sending control characters (e.g., c -> (SIGINT) - Interrupts the current command).

        After sending the signal, the tool will attempt to capture any immediate output or error messages from the shell. This can help confirm if the signal was processed as expected.

        :param signal: A single character representing the control signal to send (e.g., 'c' for Ctrl+C)
        """

        self.shell.sendcontrol(signal)

        output = self._get_output_until_prompt()

        self.command_running = False

        _ = self.run_command_in_shell(
            "pwd"
        )  # for some reason the first command output is not parsed correctly after a signal is sent

        return UnixShellInteractionResult(
            output=output,
            state=UnixShellInteractionState.FINISHED.value,
        ).model_dump()

    def _run_basic_command(self, command: str) -> str:
        self.shell.sendline(command)
        self.shell.expect_exact(self.prompt)

        pure_result = self.shell.before
        return self._format_command_output(pure_result)

    def _format_command_output(
        self, raw_output: str, append_prompt: bool = False
    ) -> str:
        """
        Format the raw output of a command output.

        Args:
            raw_output: The raw output string to format
            append_prompt: Whether to append the shell prompt to the output
        """
        cleaned = self._escape_ansi(raw_output.strip())
        if append_prompt:
            return (cleaned + self.prompt).strip()
        return cleaned.strip()

    def _escape_ansi(self, string: str) -> str:
        """
        Remove ANSI escape sequences from a string.
        """
        ansi_escape = re.compile(r"(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]")
        return ansi_escape.sub("", string)

    def __del__(self):
        super().__del__()
        self.shell.close()

    def _get_output_until_prompt(self) -> str:
        """
        Helper method to get output from the shell until the prompt is found or timeout.
        """
        try:
            self.shell.expect(self.prompt, timeout=3)
            output = self._format_command_output(self.shell.before, append_prompt=True)
        except pexpect.TIMEOUT:
            output = self._format_command_output(self.shell.before)
        return output
