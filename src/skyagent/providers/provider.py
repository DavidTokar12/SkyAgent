from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic import BaseModel

    from skyagent.messages import BaseMessagePart
    from skyagent.response import IterationResponse
    from skyagent.tool import Tool


class Provider(ABC):

    def __init__(
        self,
        model: str,
        model_settings: dict | None = None,
        client_settings: dict | None = None,
    ):

        self.model = model
        self.model_settings = model_settings
        self.client_settings = client_settings

    @abstractmethod
    async def run_iteration(
        self,
        chat_history: list[BaseMessagePart],
        result_format: BaseModel | None = None,
        tools: list[Tool] | None = None,
    ) -> IterationResponse:
        raise NotImplementedError("The run_iteration method must be implemented!")

    @abstractmethod
    async def run_iteration_stream(
        self,
        chat_history: list[BaseMessagePart],
        result_format: BaseModel | None = None,
        tools: list[Tool] | None = None,
    ) -> AsyncIterator:
        raise NotImplementedError(
            "The run_iteration_stream method must be implemented!"
        )
