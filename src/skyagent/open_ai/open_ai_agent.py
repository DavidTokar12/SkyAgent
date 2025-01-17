from __future__ import annotations

import time

from typing import TYPE_CHECKING

from skyagent.base.agent import BaseAgent
from skyagent.base.chat_message import AssistantChatMessage
from skyagent.base.chat_message import ChatMessageRole
from skyagent.base.chat_message import UserChatMessage
from skyagent.base.exceptions import SkyAgentDetrimentalError
from skyagent.open_ai.open_ai_api_adapter import OpenAiApiAdapter


if TYPE_CHECKING:
    from pathlib import Path

    from skyagent.base.llm_api_adapter import CompletionResponse
    from skyagent.open_ai.open_ai_tool import OpenAITool


class OpenAIAgent(BaseAgent):
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
        parallelize: bool = True,
        num_processes: int = 4,
        temperature: float = 0.0,
        timeout: int = 3,
        log_file_path: Path | None = None,
        log_server: str | None = None,
        enable_live_display: bool = False,
    ) -> None:

        super().__init__(
            name=name,
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            max_turns=max_turns,
            token=token,
            parallelize=parallelize,
            num_processes=num_processes,
            temperature=temperature,
            timeout=timeout,
            log_file_path=log_file_path,
            log_server=log_server,
            enable_live_display=enable_live_display,
        )

        self.client = OpenAiApiAdapter(
            model=self.model,
            token=self.token,
            temperature=self.temperature,
            timeout=self.timeout,
        )

    def _call_implementation(self, query: str) -> CompletionResponse:

        try:
            self.logger.query_received(query=query)

            start_time = time.time()

            self.chat_history.append(
                UserChatMessage(role=ChatMessageRole.user, content=query)
            )

            for current_turn in range(1, self.max_turns + 1, 1):

                self.logger.chat_loop_started(turn=current_turn)

                completion = self.client.get_completion(
                    chat_history=self.chat_history, tools=self.tools_array
                )

                if completion.tool_calls:

                    self.logger.tool_calls_received_from_server(completion.tool_calls)

                    tool_call_results = self.execute_tool_calls(completion.tool_calls)

                    for tool_call_result in tool_call_results:
                        tool_result_answer = self.client.convert_tool_result_answer(
                            tool_call_result=tool_call_result
                        )
                        self.chat_history.append(tool_result_answer)

                else:
                    self.chat_history.append(
                        AssistantChatMessage(content=completion.content)
                    )

                    execution_time = time.time() - start_time

                    self.logger.final_completion_received_from_server(
                        completion=completion, execution_time=execution_time
                    )

                    return completion
        except SkyAgentDetrimentalError as e:
            self.logger.error_happened(e)
