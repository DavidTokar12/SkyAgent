from __future__ import annotations

import pytest

from skyagent.agent_tool import AgentTool
from skyagent.agent_tool import AgentToolParsingError


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


def test_missing_docstring():
    def func(a: int, b: str) -> str:
        return "Test string!"

    with pytest.raises(AgentToolParsingError):
        AgentTool(func=func)


def test_missing_arg_in_docstring():
    def func(a: int, b: str, c: float) -> str:
        """
        Args:
            a: An integer.
            b: A string.
        """
        return "Test string!"

    with pytest.raises(AgentToolParsingError):
        AgentTool(func=func)


def test_complex_argument_types():
    def func(a: list, b: dict) -> str:
        """A test function with complex types.

        Args:
            a: A list of integers.
            b: A dictionary with string keys and values.
        """
        return "Test string!"

    tool = AgentTool(func=func)

    result = tool.to_dict()

    assert result == {
        "type": "function",
        "function": {
            "name": "func",
            "description": "A test function with complex types. ",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "array", "description": "A list of integers."},
                    "b": {
                        "type": "object",
                        "description": "A dictionary with string keys and values.",
                    },
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    }
