from __future__ import annotations

import importlib

from dataclasses import dataclass
from functools import lru_cache
from typing import ClassVar

from skyagent.exceptions import SkyAgentNotSupportedError


@dataclass
class ProviderConfig:
    """Configuration for an API provider."""

    module_path: str
    adapter_class_name: str


class ProviderRegistry:
    _providers: ClassVar[dict[str, ProviderConfig]] = {}

    @classmethod
    def register(
        cls, name: str, module_path: str, adapter_class_name: str, **default_kwargs
    ):
        """Register a new provider configuration."""
        cls._providers[name] = ProviderConfig(
            module_path=module_path,
            adapter_class_name=adapter_class_name,
        )

    @classmethod
    @lru_cache
    def get_provider_class(cls, provider: str) -> type:
        """Get the adapter class for a provider."""

        if provider not in cls._providers:
            raise SkyAgentNotSupportedError(
                f"Unknown provider: {provider}. Registered providers: {list(cls._providers.keys())}"
            )

        config = cls._providers[provider]
        module = importlib.import_module(config.module_path)
        adapter_class = getattr(module, config.adapter_class_name)
        return adapter_class


ProviderRegistry.register(
    "openai",
    "skyagent.base.api_adapters.predefined_adapters.openai",
    "OpenAiApiAdapter",
)

ProviderRegistry.register(
    "anthropic",
    "skyagent.base.api_adapters.predefined_adapters.anthropic",
    "AnthropicApiAdapter",
)
