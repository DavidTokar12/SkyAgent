from __future__ import annotations

from collections.abc import Callable  # noqa: TCH003
from dataclasses import dataclass
from enum import Enum
from enum import IntEnum

import pytest

from pydantic import BaseModel

from skyagent.exceptions import SkyAgentToolParsingError
from skyagent.tool import Tool


def test_simple_tool_parsing():

    async def tool_function(x: int) -> int:
        """SHORT
        LONG

        Args:
            x: PARAM
        """

    tool = Tool(tool_function=tool_function)

    assert not tool._is_compute_heavy
    assert tool._is_async
    assert tool._tool_function_schema.tool_name == "tool_function"

    model_fields = tool._tool_function_schema.params_pydantic_model.model_fields
    assert model_fields["x"].description == "PARAM"
    assert model_fields["x"].annotation is int
    assert model_fields["x"].is_required

    assert tool._tool_function_schema.tool_description == "SHORT LONG"


def test_simple_primitive_parameters():

    def tool_with_primitives(num: int, text: str, flag: bool, ratio: float) -> str:
        pass

    tool = Tool(tool_function=tool_with_primitives)

    model_fields = tool._tool_function_schema.params_pydantic_model.model_fields
    assert model_fields["num"].annotation is int
    assert model_fields["text"].annotation is str
    assert model_fields["flag"].annotation is bool
    assert model_fields["ratio"].annotation is float
    assert all(field.is_required for field in model_fields.values())

    assert "num" in tool._tool_function_schema.params_json_schema["required"]
    assert "text" in tool._tool_function_schema.params_json_schema["required"]
    assert "flag" in tool._tool_function_schema.params_json_schema["required"]
    assert "ratio" in tool._tool_function_schema.params_json_schema["required"]


class Color(str, Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class Number(IntEnum):
    ONE = 1
    TWO = 2
    THREE = 3


def test_simple_enum_parameters():

    async def tool_with_enums(color: Color, number: Number) -> str:
        pass

    tool = Tool(tool_function=tool_with_enums)

    model_fields = tool._tool_function_schema.params_pydantic_model.model_fields
    assert model_fields["color"].annotation is Color
    assert model_fields["number"].annotation is Number


class Status(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"


@dataclass
class Address:
    street: str
    city: str
    zip_code: str


class Contact(BaseModel):
    name: str
    email: str
    status: Status
    address: Address


def test_pydantic_and_dataclass_parameters():
    """Test tool with Pydantic models and dataclasses as parameters."""

    def tool_with_models(contact: Contact) -> str:
        pass

    tool = Tool(tool_function=tool_with_models)

    model_fields = tool._tool_function_schema.params_pydantic_model.model_fields
    assert model_fields["contact"].annotation is Contact

    schema = tool._tool_function_schema.params_json_schema
    contact_schema = schema["properties"]["contact"]

    if "$ref" in contact_schema:
        ref_path = contact_schema["$ref"]
        ref_name = ref_path.split("/")[-1]
        contact_schema = schema["$defs"][ref_name]

    assert contact_schema["type"] == "object"
    assert "name" in contact_schema["properties"]
    assert "email" in contact_schema["properties"]
    assert "status" in contact_schema["properties"]
    assert "address" in contact_schema["properties"]


def test_error_for_missing_return_type():
    def tool_without_return_type(param: str):
        pass

    with pytest.raises(SkyAgentToolParsingError) as exc_info:
        Tool(tool_function=tool_without_return_type)

    assert "missing a return type annotation" in str(exc_info.value)


def test_error_for_missing_parameter_type():

    def tool_with_untyped_param(typed_param: str, untyped_param) -> str:
        pass

    with pytest.raises(SkyAgentToolParsingError) as exc_info:
        Tool(tool_function=tool_with_untyped_param)

    assert "missing a type annotation" in str(exc_info.value)
    assert "untyped_param" in str(exc_info.value)


def test_error_for_var_positional_args():
    def tool_with_args(param: str, *args: int) -> str:
        pass

    with pytest.raises(SkyAgentToolParsingError) as exc_info:
        Tool(tool_function=tool_with_args)

    assert "Positional arguments are not supported" in str(exc_info.value)


def test_error_for_var_keyword_args():
    def tool_with_kwargs(param: str, **kwargs: str) -> str:
        pass

    with pytest.raises(SkyAgentToolParsingError) as exc_info:
        Tool(tool_function=tool_with_kwargs)

    assert "Keyword arguments are not supported" in str(exc_info.value)


def test_error_for_unsupported_callable_type():
    def tool_with_callable(callback: Callable[[str], None]) -> str:
        pass

    with pytest.raises(SkyAgentToolParsingError) as exc_info:
        Tool(tool_function=tool_with_callable)

    assert "Failed to generate JSON schema" in str(exc_info.value)
    assert "complex types" in str(exc_info.value)
