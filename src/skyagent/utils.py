from __future__ import annotations

import asyncio

from typing import TYPE_CHECKING

from skyagent.base.exceptions import SkyAgentDetrimentalError


if TYPE_CHECKING:
    from pydantic import BaseModel


def _model_to_string(model: type[BaseModel]) -> str:
    try:
        json_schema = model.model_json_schema()["properties"]
        return str(json_schema)
    except Exception as e:
        raise SkyAgentDetrimentalError(f"Failed to convert model to string: {e}")


TEXT_CHARS = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7F})


def _is_binary_string(bytes: bytes) -> bool:
    # based on https://github.com/file/file/blob/f2a6e7cb7db9b5fd86100403df6b2f830c7f22ba/src/encoding.c#L151-L228
    return bool(bytes.translate(None, TEXT_CHARS))


def _get_or_create_event_loop():

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop


def _run_async_in_task(async_func, *args, **kwargs):
    return asyncio.run(async_func(*args, **kwargs))
