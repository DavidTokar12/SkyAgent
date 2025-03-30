from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from datetime import datetime

    from pydantic import BaseModel

    from skyagent.tool import ToolCall
    from skyagent.usage import Usage


@dataclass
class IterationResponse:
    usage: Usage
    timestamp: datetime
    content: str | BaseModel | None = None
    tool_calls: list[ToolCall] | None = None
