from __future__ import annotations

import pytest

from miniagent.agent_tool import AgentTool
from miniagent.agent_tool import AgentToolParsingError


def test_agent_tool_simple_parsing():

    def func(a: int, b: str, c: float) -> str:
        """A test function.
        This is a long description.

        Args:
            a: An integer.
            b: A string.
            c: A float.
        """
        return "Test string!"

    tool = AgentTool(func=func)

    result = tool.to_dict()
    assert result == {
        "type": "function",
        "function": {
            "name": "func",
            "description": "A test function. This is a long description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "An integer."},
                    "b": {"type": "string", "description": "A string."},
                    "c": {"type": "number", "description": "A float."},
                },
                "required": ["a", "b", "c"],
                "additionalProperties": False,
            },
        },
    }


def test_agent_tool_partial_docstring_1():

    def func(a: int, b: str, c: float) -> str:
        """
        Args:
            a: An integer.
            b: A string.
        """
        return "Test string!"

    with pytest.raises(AgentToolParsingError):
        AgentTool(func=func)


def test_invalid_return_type():

    def func(a: int, b: str, c: float) -> Exception:
        """A test function.
        This is a long description.

        Args:
            a: An integer.
            b: A string.
            c: A float.
        """
        return "Test string!"

    with pytest.raises(AgentToolParsingError):
        AgentTool(func=func)


def test_default_arguments():

    def func(a: int, b: str, c: float = 3.0) -> str:
        """A test function.
        This is a long description.

        Args:
            a: An integer.
            b: A string.
            c: A float.
        """
        return "Test string!"

    tool = AgentTool(func=func)

    result = tool.to_dict()
    assert result == {
        "type": "function",
        "function": {
            "name": "func",
            "description": "A test function. This is a long description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "An integer."},
                    "b": {"type": "string", "description": "A string."},
                    "c": {"type": "number", "description": "A float."},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    }
