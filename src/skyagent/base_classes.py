from __future__ import annotations

import logging

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

from pydantic import BaseModel


if TYPE_CHECKING:
    from skyagent.open_ai.open_ai_tool import OpenAITool

logger = logging.getLogger(__name__)


class ChatMessageRole(str, Enum):
    system = "system"
    assistant = "assistant"
    user = "user"
    tool = "tool"


class UserChatMessage(BaseModel):
    "Represents a user or system message in a chat history."
    role: ChatMessageRole = ChatMessageRole.user
    content: str


class AssistantChatMessage(BaseModel):
    "Represents a user or system message in a chat history."
    role: ChatMessageRole = ChatMessageRole.assistant
    content: str


class ToolCallOutgoingMessage(BaseModel):
    role: ChatMessageRole = ChatMessageRole.tool
    content: str
    tool_call_id: str


class ImageChatMessage(BaseModel):
    "Represents a user or system message in a chat history with an image value."
    role: ChatMessageRole
    image_url: str


BaseTypes = str | int | float | bool | list | dict | None


class ToolCall(BaseModel):
    """
    Represents a tool call returned by the model in the chat completion.
    """

    id: str
    function_name: str
    arguments: dict[str, BaseTypes]


class LLMUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int


class CompletionResponse(BaseModel):
    content: str | Any
    tool_calls: list[ToolCall] | None
    usage: LLMUsage


class LLMClient:

    def __init__(
        self,
        model: str,
        token: str | None,
        temperature: float = 0.0,
        timeout: int = 3,
    ):
        """Base class for a client handling direct communication with an LLM API.

        :param model (str): Model name.
        :param token (str | None): API token.
        :param temperature (float, optional): Temperature of the model. Defaults to 0.0.
        :param timeout (int, optional): Timeout in seconds. Defaults to 3.
        """
        self.model = model
        self.token = token
        self.temperature = temperature
        self.timeout = timeout

    def get_completion(
        self,
        message_history: list[UserChatMessage | ImageChatMessage],
        response_format: Any | None = None,
        tools: list[OpenAITool] | None = None,
    ) -> CompletionResponse:
        raise NotImplementedError(
            "The get_completion method must be implemented!")


class AgentDetrimentalError(Exception):
    """Exception meaning the error was not recoverable, and the Agent must terminate."""


class AgentConversationToLongError(Exception):
    """Exception meaning the conversation is larger then the context window."""


class AgentCopyrightError(Exception):
    """Exception meaning that the conversation included copyright material, thus could not be answered."""


class AgentBase:
    "Base class for a callable LLM agent."

    def __init__(
        self,
        name: str,
        model: str,
        system_prompt: str | Path,
        tools: list[OpenAITool] | None,
        max_turns: int = 10,
    ):
        """
        Initialize the agent.

        :param name: The agent's name.
        :param model: The model name.
        :param system_prompt: Instructional system message for the model, or the path to a markdown or text file.
        :param tools: A list of AgentTool instances.
        :param max_turns: The maximum number of turns in the conversation.
        """

        self.name = name
        self.model = model

        if isinstance(system_prompt, Path):
            if not system_prompt.exists():
                raise FileNotFoundError(
                    "System prompt file not found: '%s'", system_prompt
                )
            with open(system_prompt) as f:
                self.system_prompt = f.read()
        else:
            self.system_prompt = system_prompt

        self.tools_array = tools
        self.tools_dict = {
            tool.name: tool for tool in tools} if tools else None

        self.max_turns = max_turns

        logger.debug(
            "Initialized Agent '%s' with model '%s', system prompt '%s' and tool definitions: '%s'.",
            self.name,
            self.model,
            self.system_prompt,
        )

        self.chat_history = [
            UserChatMessage(role=ChatMessageRole.system,
                            content=self.system_prompt)
        ]

    def call(self, query: str) -> Any:
        """
        Sends the user's query to the model and handles a potentially infinite loop
        of tool calls. Returns an object containing:
            - final_answer: the final text response from the model
            - history: a list of message dicts (including all tool calls).

        :param query: The user's query as a string.
        """
        raise NotImplementedError("The call method must be implemented!")

    def execute_tool_call(self, tool_call: ToolCall) -> Any:
        """
        Executes the tool call (function) based on the provided ToolCallModel.
        Returns the result from the function execution.
        """

        tool_wrapper = self.tools_dict.get(tool_call.function_name)

        logger.debug(
            "Executing tool '%s' with arguments: %s",
            tool_call.function_name,
            tool_call.arguments,
        )

        if not tool_wrapper:
            raise AgentDetrimentalError(
                "Tool '%s' not found in the agent named '%s' tool definitions.",
                tool_call.function_name,
                self.name,
            )

        converted_args = {}
        for param_name, param_value in tool_call.arguments.items():
            converted_args[param_name] = self.tools_dict[
                tool_call.function_name
            ].validate_and_convert_input_param(
                input_param_name=param_name, input_param_value=param_value
            )

        # Actually execute the tool's Python function
        try:
            result = tool_wrapper.tool_function(**converted_args)
            logger.debug(
                "Tool '%s' execution result: '%s' from inputs '%s'",
                tool_call.function_name,
                result,
                converted_args,
            )
            return result
        except Exception as e:
            raise AgentDetrimentalError(
                f"Error executing tool '{tool_call.function_name}': '{e}'"
            )