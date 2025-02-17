from __future__ import annotations

from skyagent.base.agent import BaseAgent
from skyagent.open_ai.open_ai_api_adapter import OpenAiApiAdapter


class OpenAIAgent(BaseAgent):

    def _initialize_client(self):
        self.client = OpenAiApiAdapter(
            model=self.model,
            token=self.token,
            temperature=self.temperature,
            timeout=self.timeout,
        )
