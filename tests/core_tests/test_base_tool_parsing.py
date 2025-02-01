from __future__ import annotations

import pytest

from skyagent.base.exceptions import SkyAgentDetrimentalError
from skyagent.base.exceptions import SkyAgentToolParsingError
from skyagent.base.tools import BaseTool


def test_basic_tool_parsing():
    def sample_func(a: int, b: str, c: float) -> str:
        """Test function with basic types.

        This is a detailed description of the function.

        Args:
            a: An integer parameter.
            b: A string parameter.
            c: A floating point number.
        """
        return "test"

    tool = BaseTool(tool_function=sample_func)

    assert tool.name == "sample_func"
    assert (
        tool.description
        == "Test function with basic types. This is a detailed description of the function."
    )
    assert len(tool.parameters) == 3
    assert tool.required_properties == ["a", "b", "c"]
    assert not tool.is_async
    assert not tool.is_compute_heavy


def test_tool_with_optional_parameters():
    def func_with_defaults(a: int, b: str = "default", c: float = 1.0) -> dict:
        """Function with optional parameters.

        Args:
            a: Required integer.
            b: Optional string.
            c: Optional float.
        """
        return {}

    tool = BaseTool(tool_function=func_with_defaults)

    assert tool.required_properties == ["a"]
    assert len(tool.parameters) == 3
    assert {p.name: p.type for p in tool.parameters} == {
        "a": "integer",
        "b": "string",
        "c": "number",
    }


def test_tool_with_complex_types():
    def complex_func(data: dict, items: list, flag: bool = False) -> dict:
        """Function with complex types.

        Args:
            data: A dictionary of values.
            items: A list of items.
            flag: A boolean flag.
        """
        return {}

    tool = BaseTool(tool_function=complex_func)

    assert {p.name: p.type for p in tool.parameters} == {
        "data": "object",
        "items": "array",
        "flag": "boolean",
    }


def test_missing_docstring():
    def no_docs(a: int) -> str:
        return "test"

    with pytest.raises(SkyAgentToolParsingError) as exc_info:
        BaseTool(tool_function=no_docs)

    assert "must have a function description" in str(exc_info.value)


def test_missing_type_annotation():
    def missing_type(a, b: str) -> str:
        """Test function.

        Args:
            a: Missing type annotation.
            b: String parameter.
        """
        return "test"

    with pytest.raises(SkyAgentToolParsingError) as exc_info:
        BaseTool(tool_function=missing_type)

    assert "must have a type annotation" in str(exc_info.value)


def test_invalid_parameter_type():
    def invalid_type(a: set) -> str:
        """Test function.

        Args:
            a: Invalid type.
        """
        return "test"

    with pytest.raises(SkyAgentToolParsingError) as exc_info:
        BaseTool(tool_function=invalid_type)

    assert "must be annotated with one of" in str(exc_info.value)


def test_missing_return_type():
    def no_return(a: int):
        """Test function.

        Args:
            a: Integer parameter.
        """
        return 1

    with pytest.raises(SkyAgentToolParsingError) as exc_info:
        BaseTool(tool_function=no_return)

    assert "Return type annotation is required" in str(exc_info.value)


def test_input_parameter_validation():
    def sample_func(a: int, b: str) -> str:
        """Test function.

        Args:
            a: Integer parameter.
            b: String parameter.
        """
        return "test"

    tool = BaseTool(tool_function=sample_func)

    # Test valid conversions
    assert tool.validate_and_convert_input_param("a", "123") == 123
    assert tool.validate_and_convert_input_param("b", "test") == "test"

    # Test invalid parameter name
    with pytest.raises(SkyAgentDetrimentalError) as exc_info:
        tool.validate_and_convert_input_param("c", "invalid")
    assert "Not defined input parameter" in str(exc_info.value)

    # Test invalid type conversion
    with pytest.raises(SkyAgentDetrimentalError) as exc_info:
        tool.validate_and_convert_input_param("a", "not_a_number")
    assert "Failed to convert parameter" in str(exc_info.value)


def test_async_function_detection():
    async def async_func(a: int) -> str:
        """Async test function.

        Args:
            a: Integer parameter.
        """
        return "test"

    tool = BaseTool(tool_function=async_func)
    assert tool.is_async


def test_compute_heavy_flag():
    def sample_func(a: int) -> str:
        """Test function.

        Args:
            a: Integer parameter.
        """
        return "test"

    tool = BaseTool(tool_function=sample_func, is_compute_heavy=True)
    assert tool.is_compute_heavy


def test_additional_properties_flag():
    def sample_func(a: int) -> str:
        """Test function.

        Args:
            a: Integer parameter.
        """
        return "test"

    tool = BaseTool(tool_function=sample_func, additional_properties=True)
    assert tool.additional_properties
