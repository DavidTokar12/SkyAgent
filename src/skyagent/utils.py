from __future__ import annotations

from typing import TYPE_CHECKING

from skyagent.base.exceptions import SkyAgentDetrimentalError


if TYPE_CHECKING:
    from pydantic import BaseModel


def model_to_string(model: type[BaseModel]) -> str:
    try:
        json_schema = model.model_json_schema()["properties"]
        return str(json_schema)
    except Exception as e:
        raise SkyAgentDetrimentalError(f"Failed to convert model to string: {e}")
