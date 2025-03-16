from __future__ import annotations


class _SkyAgentError(Exception):
    """Parent exception of all sky agent exceptions."""


class SkyAgentToolParsingError(_SkyAgentError):
    """Base exception for AgentTool-related errors."""


class SkyAgentDetrimentalError(_SkyAgentError):
    """Exception meaning the error was not recoverable, and the Agent must terminate."""


class SkyAgentContextWindowSaturatedError(_SkyAgentError):
    """Exception meaning the conversation is larger then the context window."""


class SkyAgentCopyrightError(_SkyAgentError):
    """Exception meaning that the conversation included copyright material, thus could not be answered."""


class SkyAgentUnsupportedFileTypeError(Exception):
    """Exception raised when file type is not supported."""


class SkyAgentValidationError(Exception):
    """Base error for input validation issues"""


class SkyAgentFileError(Exception):
    """Base error for file operation issues"""


class SkyAgentNotSupportedError(Exception):
    """Base error for unsupported operations"""


class SkyAgentTypeError(Exception):
    """Base error for unsupported operations or unexpected types"""
