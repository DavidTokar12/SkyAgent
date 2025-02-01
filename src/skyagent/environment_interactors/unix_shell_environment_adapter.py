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

        self.shell = None

    def get_tool_functions(self) -> list[callable]:
        return [
            self.run_command_in_shell,
            self.get_update_of_shell_output,
            self.send_control_signal,
            self.write_input_to_shell,
        ]

    def __enter__(self) -> UnixShellAdapter:
        super().__enter__()

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
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:

        if self.shell is not None:
            self.shell.close()

        super().__exit__(exc_type, exc_value, traceback)

    def run_command_in_shell(self, command_to_run: str) -> dict:
        """
        Use this tool to run a command in a Unix shell. You have access to a single, persistent shell instance.

        The tool will write your command as an input to the shell with the pexpect sendline method, which appends os.linesep to your input automatically.

        **Caution: You must only try to run a single shell command with a single call to this tool. If you start to chain commands, the tool might not work as expected.**

        It supports longer and interactive commands, returning partial results when necessary, allowing for interaction or cancellation.

        The tool returns the command's output, and a state value.
        The execution state can be:
        *   **finished:** Command finished.
        *   **interrupted:** Command execution was interrupted, providing a partial result. You can use the 'get_update_of_shell_output' tool to monitor the current command, the 'send_control_signal' tool to send a control signal to the shell(to cancel it for example), or the 'write_input_to_shell' if the command asks you for some input. You can't execute a new command before using any of the 'get_update_of_shell_output', 'send_control_signal', or 'write_input_to_shell' tools, and ensuring the command is no longer running(your requests to run commands before doing so will be automatically declined).
        *   **busy:** Another command is running, and you can't execute a new command until ensuring that the current command finishes. If you use the tools properly, you should never see this state.

        :param command_to_run: The command you wish to run in the shell.
        """

        if self.command_running:
            return UnixShellInteractionResult(
                output="Cannot execute new command while previous command is still running. You must proceed with any of the 'get_update_of_shell_output', 'send_control_signal', or 'write_input_to_shell' tools.",
                state=UnixShellInteractionState.BUSY.value,
            ).model_dump()

        self.command_running = True

        self.shell.sendline(command_to_run)

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

    def write_input_to_shell(self, input_to_write: str) -> dict:
        """
        Use this tool to write input to the shell. This tool is useful for interacting with running commands that require user input.

        The tool will write your input to the shell with the pexpect sendline method, which appends os.linesep to your input automatically.

        **CAUTION: This tool is not suitable for running commands. Instead, use the 'run_command_in_shell' tool with with your desired command.**

        The tool returns the output of the shell after input has been written to it, and a state value.
        The execution state can be:
        *   **finished:** Your input was processed, and no further input is required. The previous command is no longer running, and you can execute a new command.
        *   **interrupted:** Your input was processed, but the command is still running and producing output. You must continue monitoring the command by calling the 'get_update_of_shell_output' tool, or using the 'send_control_signal'(to cancel it for example), or writing additional input with this('write_input_to_shell') tool.

        :param input_to_write: The input string to write to the shell.
        """

        self.shell.sendline(input_to_write)

        try:
            self.shell.expect_exact(self.prompt, timeout=1)
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

        The tool checks the shell's output buffer and returns both the current output and the command's state.

        The execution state in the response can be:
        *   **finished:** Command has completed execution. You can view the final output, and execute a new command.
        *   **interrupted:** Command is still running and producing output. Continue monitoring by calling this tool('get_update_of_shell_output') again, or by sending a control signal with the 'send_control_signal' tool, or by writing some input to the shell(for interactive scripts) with the 'write_input_to_shell' tool.
        """

        if not self.command_running:
            return UnixShellInteractionResult(
                output=self._format_command_output(
                    self.shell.before, append_prompt=True
                ),
                state=UnixShellInteractionState.FINISHED.value,
            ).model_dump()

        try:
            # Wait for the prompt for 1 second.
            self.shell.expect_exact(self.prompt, timeout=1)
            output = self._format_command_output(self.shell.before)
            state = UnixShellInteractionState.FINISHED.value
            self.command_running = False
        except pexpect.TIMEOUT:
            # If timeout occurs, we treat the command as still running.
            output = self._format_command_output(self.shell.before)
            state = UnixShellInteractionState.INTERRUPTED.value

        return UnixShellInteractionResult(
            output=output,
            state=state,
        ).model_dump()

    def send_control_signal(self, signal: str) -> dict:
        """
        Use this tool to send a control signal to the shell. This tool is useful for interacting with running commands by sending control characters (e.g., c -> (SIGINT) - Interrupts the current command). This tool uses the pexpect sendcontrol method to send the signal.

        **CAUTION:** This tool is not suitable for interactive commands or scripts that require user input. If you come across such a command, don't use control signals. Instead, call the 'write_input_to_shell' tool with your desired input.

        After sending the signal, the tool will attempt to capture any immediate output or error messages from the shell. This can help confirm if the signal was processed as expected.

        :param signal: A single character representing the control signal to send (e.g., 'c' for Ctrl+C)
        """

        self.shell.sendcontrol(signal)

        output = self._get_output_until_prompt()

        # TODO this might not always be the case. Models will most commonly use this to terminate a command, but it could be used for other signals.
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
