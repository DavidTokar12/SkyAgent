from __future__ import annotations

from skyagent.anthropic.anthropic_api_adapter import AnthropicApiAdapter
from skyagent.base.agent import BaseAgent


class AnthropicAgent(BaseAgent):

    def _initialize_client(self):
        self.client = AnthropicApiAdapter(
            model=self.model,
            token=self.token,
            temperature=self.temperature,
            timeout=self.timeout,
        )
