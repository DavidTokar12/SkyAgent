from __future__ import annotations

import importlib

from dataclasses import dataclass
from functools import lru_cache
from typing import ClassVar

from skyagent.base.exceptions import SkyAgentNotSupportedError


@dataclass
class ApiProviderConfig:
    """Configuration for an API provider."""

    module_path: str
    adapter_class_name: str


class ApiRegistry:
    _providers: ClassVar[dict[str, ApiProviderConfig]] = {}

    @classmethod
    def register(
        cls, name: str, module_path: str, adapter_class_name: str, **default_kwargs
    ):
        """Register a new provider configuration."""
        cls._providers[name] = ApiProviderConfig(
            module_path=module_path,
            adapter_class_name=adapter_class_name,
        )

    @classmethod
    @lru_cache
    def get_adapter_class(cls, provider: str) -> type:
        """Get the adapter class for a provider."""

        if provider not in cls._providers:
            raise SkyAgentNotSupportedError(
                f"Unknown provider: {provider}. Registered providers: {list(cls._providers.keys())}"
            )

        config = cls._providers[provider]
        module = importlib.import_module(config.module_path)
        adapter_class = getattr(module, config.adapter_class_name)
        return adapter_class


ApiRegistry.register(
    "openai",
    "skyagent.base.api_adapters.predefined_adapters.openai",
    "OpenAiApiAdapter",
)

ApiRegistry.register(
    "anthropic",
    "skyagent.base.api_adapters.predefined_adapters.anthropic",
    "AnthropicApiAdapter",
)
