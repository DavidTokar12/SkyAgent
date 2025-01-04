from __future__ import annotations

import json
import logging

from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

from openai import OpenAI
from pydantic import BaseModel


if TYPE_CHECKING:
    from collections.abc import OpenAI

    from skyagent.agent_tool import AgentTool

logger = logging.getLogger(__name__)


class AgentError(Exception):
    """Base exception for Agent-related errors."""


class FunctionModel(BaseModel):
    """
    Represents the core fields for a function call.
    """

    arguments: str
    name: str


class ToolCallModel(BaseModel):
    """
    Represents a tool call returned by the model in the chat completion.
    """

    id: str
    function: FunctionModel
    type: str


class Agent:
    """
    Orchestrates conversation between the user, the model, and available tools.
    """

    def __init__(
        self,
        name: str,
        client: OpenAI,
        model: str,
        system_prompt: str | Path,
        tools: list[AgentTool],
        max_turns: int = 10,
    ) -> None:
        """
        Initialize the agent.

        :param name: The agent's name.
        :param client: The OpenAI client.
        :param model: The model name (e.g., gpt-4).
        :param system_prompt: Instructional system message for the model, or the path to a markdown or text file.
        :param tools: A list of AgentTool instances.
        :param max_turns: The maximum number of turns in the conversation.
        """
        self.name = name
        self.client = client
        self.model = model
        self.max_turns = max_turns

        if isinstance(system_prompt, Path):
            if not system_prompt.exists():
                raise FileNotFoundError(
                    f"System prompt file not found: {system_prompt}"
                )
            with open(system_prompt) as f:
                system_prompt = f.read()

        self.system_prompt = system_prompt
        self.tools = tools

        self.chat_history = [{"role": "system", "content": self.system_prompt}]

        # Store each tool definition and function wrapper for easy access
        self.tool_definitions = [tool.to_dict() for tool in self.tools]
        self.tool_function_wrappers = {tool.name: tool for tool in self.tools}

        logger.debug(
            "Initialized Agent with model '%s', system prompt '%s' and tool definitions: '%s'.",
            self.model,
            self.system_prompt,
            self.tool_definitions,
        )

    def call(self, query: str) -> dict[str, Any]:
        """
        Sends the user's query to the model and handles a potentially infinite loop
        of tool calls. Returns an object containing:
            - final_answer: the final text response from the model
            - history: a list of message dicts (including all tool calls).
        """

        logger.debug("Agent.call received query: '%s'", query)

        user_message = {"role": "user", "content": query}
        self.chat_history.append(user_message)

        final_answer: str | None = None

        current_turn = 0

        while current_turn < self.max_turns:

            current_turn += 1

            completion = self._create_completion(
                self.chat_history, self.tool_definitions
            )

            if not completion or not completion.choices or len(completion.choices) == 0:
                raise AgentError("No completion returned by the model.")

            message_from_model = completion.choices[0].message

            tool_calls = message_from_model.tool_calls

            if tool_calls:

                logger.info("Model returned tool calls: %s", tool_calls)
                self.chat_history.append(message_from_model.to_dict())

                for tool_call in tool_calls:
                    tool_call_schema = tool_call.to_dict()
                    try:
                        tool_call_model = ToolCallModel(**tool_call_schema)
                    except Exception as e:
                        logger.exception("Error parsing tool call schema: %s", e)
                        continue

                    if tool_call_model.type != "function":
                        raise AgentError(
                            f"Unexpected tool call type from model: {
                                tool_call_model}"
                        )

                    result = self._execute_tool(tool_call_model)

                    tool_result_answer = self._generate_tool_result_answer(
                        tool_call_model, result
                    )
                    self.chat_history.append(tool_result_answer)

                # Loop back up and let the model see the new "tool" messages
                # so it can decide if it wants to call more tools or give a direct answer.
                continue

            else:
                # No tool calls => a direct answer from the model
                direct_answer = message_from_model.content
                logger.debug(
                    "Model returned direct answer without tool calls: %s", direct_answer
                )

                final_answer = direct_answer
                self.chat_history.append(message_from_model.to_dict())

                # Break the loop since we have our final direct answer
                break

        return {
            "final_answer": final_answer,
            "history": self.chat_history,
        }

    def _create_completion(self, messages: list[dict], tools: list[dict]) -> Any:
        """
        Creates a chat completion using the given messages and tool definitions.
        Returns the raw API response (OpenAI object).
        """
        try:
            logger.debug(
                "Creating chat completion with %d messages and %d tools.",
                len(messages),
                len(tools),
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
            )
            logger.debug("Chat completion response received.")
            return response
        except Exception as e:
            raise AgentError(f"Error creating chat completion: {e}")

    def _execute_tool(self, tool_call_model: ToolCallModel) -> Any:
        """
        Executes the tool call (function) based on the provided ToolCallModel.
        Returns the result from the function execution.
        """
        tool_wrapper = self.tool_function_wrappers.get(tool_call_model.function.name)

        logger.debug(
            "Executing tool '%s' with arguments: %s",
            tool_call_model.function.name,
            tool_call_model.function.arguments,
        )

        if not tool_wrapper:
            raise AgentError(
                f"Tool '{
                    tool_call_model.function.name}' not found in the agent's tool definitions."
            )

        # Parse arguments into Python
        try:
            raw_args = json.loads(tool_call_model.function.arguments)
        except json.JSONDecodeError as e:
            raise AgentError(f"Failed to parse JSON arguments: {e}")

        # Convert and validate arguments
        converted_args = {}
        for param_name, param_value in raw_args.items():
            if param_name not in tool_wrapper.param_types:
                raise AgentError(
                    f"Unexpected parameter '{
                                 param_name}' in tool call."
                )

            expected_type = tool_wrapper.param_types[param_name]
            conversion_func = tool_wrapper.PYTHON_TYPE_MAP[expected_type]
            try:
                converted_args[param_name] = conversion_func(param_value)
            except ValueError as e:
                raise AgentError(
                    f"Failed to convert parameter '{
                        param_name}' to {expected_type}: {e}"
                )

        logger.debug("Converted arguments: %s", converted_args)

        # Actually execute the tool's Python function
        try:
            result = tool_wrapper.func(**converted_args)
            logger.debug(
                "Tool '%s' execution result: %s",
                tool_call_model.function.name,
                result,
            )
            return result
        except Exception as e:
            raise AgentError(
                f"Error executing tool '{tool_call_model.function.name}': {e}"
            )

    def _generate_tool_result_answer(
        self, tool_call_model: ToolCallModel, result: Any
    ) -> dict:
        """
        Generates a message dict containing the result of a tool call.
        This message is appended to the conversation history so that
        the model can consume the tool's result.
        """
        content_dict = {
            "input": tool_call_model.function.arguments,
            "output": result,
        }

        logger.debug("Generated tool result answer: %s", content_dict)

        return {
            "role": "tool",
            "content": json.dumps(content_dict),
            "tool_call_id": tool_call_model.id,
        }
