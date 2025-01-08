from __future__ import annotations

import logging

from typing import TYPE_CHECKING
from typing import Any

from pydantic import BaseModel

from skyagent.base_classes import AgentBase
from skyagent.base_classes import AssistantChatMessage
from skyagent.base_classes import ChatMessageRole
from skyagent.base_classes import UserChatMessage
from skyagent.open_ai.open_ai_llm_client import OpenAILLMClient


if TYPE_CHECKING:
    from pathlib import Path

    from skyagent.open_ai.open_ai_tool import OpenAITool


logger = logging.getLogger(__name__)


class FunctionModel(BaseModel):
    """
    Represents the core fields for a function call.
    """

    arguments: str
    name: str


class OpenAIAgent(AgentBase):
    """
    Orchestrates conversation between the user, the openAI model, and available tools.
    """

    def __init__(
        self,
        name: str,
        model: str,
        system_prompt: str | Path,
        tools: list[OpenAITool] | None = None,
        max_turns: int = 10,
        token: str | None = None,
        temperature: float = 0.0,
        timeout: int = 3,
    ) -> None:
        """
        Initialize the agent.

        :param name: The agent's name.
        :param model: The model name.
        :param system_prompt: Instructional system message for the model, or the path to a markdown or text file.
        :param tools: A list of AgentTool instances.
        :param max_turns: The maximum number of turns in the conversation.
        :param token: Your OpenAI API key.
        :param temperature: The temperature of the LLM.
        """

        super().__init__(
            name=name,
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            max_turns=max_turns,
        )

        self.temperature = temperature
        self.timeout = timeout
        self.token = token

        self.client = OpenAILLMClient(
            model=self.model,
            token=self.token,
            temperature=self.temperature,
            timeout=self.timeout,
        )

    def call(self, query: str) -> dict[str, Any]:

        self.chat_history.append(
            UserChatMessage(role=ChatMessageRole.user, content=query)
        )

        for current_turn in range(1, self.max_turns + 1, 1):

            logger.debug(
                "Starting tool call loop: '%s', for agent '%s'", current_turn, self.name
            )

            completion = self.client.get_completion(
                chat_history=self.chat_history, tools=self.tools_array
            )

            if completion.tool_calls:

                for tool_call in completion.tool_calls:
                    tool_call_result = self.execute_tool_call(tool_call=tool_call)
                    tool_result_answer = self.client.generate_tool_result_answer(
                        tool_call=tool_call, result=tool_call_result
                    )
                    self.chat_history.append(tool_result_answer)
            else:
                self.chat_history.append(
                    AssistantChatMessage(content=completion.content)
                )
                return completion