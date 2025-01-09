from __future__ import annotations


class SkyAgentError(Exception):
    """Parent exception of all sky agent exceptions."""


class SkyAgentToolParsingError(SkyAgentError):
    """Base exception for AgentTool-related errors."""


class SkyAgentDetrimentalError(SkyAgentError):
    """Exception meaning the error was not recoverable, and the Agent must terminate."""


class SkyAgentContextWindowSaturatedError(SkyAgentError):
    """Exception meaning the conversation is larger then the context window."""


class SkyAgentCopyrightError(SkyAgentError):
    """Exception meaning that the conversation included copyright material, thus could not be answered."""
