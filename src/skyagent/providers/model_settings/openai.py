from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from openai import DEFAULT_MAX_RETRIES
from openai import NOT_GIVEN
from openai import NotGiven
from openai import Timeout


if TYPE_CHECKING:

    from collections.abc import Mapping

    import httpx


@dataclass
class OpenAIModelSettings:
    """Settings for OpenAI model configuration."""

    parallel_tool_calls: bool | NotGiven = NOT_GIVEN
    max_completion_tokens: int | NotGiven = NOT_GIVEN
    temperature: float | NotGiven = NOT_GIVEN
    top_p: float | NotGiven = NOT_GIVEN
    timeout: float | NotGiven = NOT_GIVEN
    seed: int | NotGiven = NOT_GIVEN
    presence_penalty: float | NotGiven = NOT_GIVEN
    frequency_penalty: float | NotGiven = NOT_GIVEN
    logit_bias: dict[str, float] | NotGiven = NOT_GIVEN
    reasoning_effort: str | NotGiven = NOT_GIVEN


@dataclass
class OpenAIClientSettings:
    """
    Settings for the OpenAI API client configuration.
    """

    api_key: str | None = None
    organization: str | None = None
    project: str | None = None

    base_url: str | httpx.URL | None = None
    websocket_base_url: str | httpx.URL | None = None

    timeout: float | Timeout | None | NotGiven = NOT_GIVEN
    max_retries: int = DEFAULT_MAX_RETRIES
    default_headers: Mapping[str, str] | None = None
    default_query: Mapping[str, object] | None = None

    http_client: httpx.Client | None = None
