from __future__ import annotations

import ast
import inspect

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import get_type_hints

import docstring_parser

from pydantic import BaseModel

from skyagent.base.exceptions import SkyAgentDetrimentalError
from skyagent.base.exceptions import SkyAgentToolParsingError


if TYPE_CHECKING:
    from collections.abc import Callable

TOOL_ARG_TYPES = str | int | float | bool | list | dict | None


class ToolCall(BaseModel):
    """
    Represents a tool call returned by the model in the chat completion.
    """

    id: str
    function_name: str
    arguments: dict[str, TOOL_ARG_TYPES]

    def __str__(self) -> str:
        return (
            f"ToolCall(\n"
            f"  id={self.id},\n"
            f"  function_name={self.function_name},\n"
            f"  arguments={self.arguments}\n"
            f")"
        )


class ToolCallResult(BaseModel):
    """
    Represents the result of a tool call.
    """

    id: str
    function_name: str
    arguments: dict[str, TOOL_ARG_TYPES]
    result: TOOL_ARG_TYPES


@dataclass
class AgentToolParameter:
    """Represents a parameter of a tool."""

    name: str
    type: str
    description: str


class BaseTool:
    """
    A class representing a tool (function) the LLM can call.
    """

    ALLOWED_TYPES: ClassVar[dict[type, str]] = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    PYTHON_TYPE_MAP: ClassVar[dict[type, Callable[[Any], Any]]] = {
        str: str,
        int: int,
        float: float,
        bool: lambda x: x if isinstance(x, bool) else str(x).lower() in ("true", "1"),
        list: lambda x: x if isinstance(x, list) else ast.literal_eval(x),
        dict: lambda x: x if isinstance(x, dict) else ast.literal_eval(x),
        type(None): lambda x: None,
    }

    def __init__(
        self,
        tool_function,
        is_compute_heavy: bool = False,
        additional_properties: bool = False,
    ) -> None:

        self.tool_function = tool_function
        self.name = self.tool_function.__name__
        self.additional_properties = additional_properties
        self.is_compute_heavy = is_compute_heavy

        self._parse_function()

    def _parse_function(self):

        self.is_async = inspect.iscoroutinefunction(self.tool_function)

        try:
            self.docstring = docstring_parser.parse(self.tool_function.__doc__)
        except Exception as e:
            raise SkyAgentToolParsingError(
                f"Failed to parse docstring for function {self.name}: {e}"
            )

        self.long_description = self.docstring.long_description
        self.short_description = self.docstring.short_description

        if not self.short_description and not self.long_description:
            raise SkyAgentToolParsingError(
                f"Function {
                    self.name} must have a function description in it's docstring.",
            )

        self.description = (
            (self.short_description + " ") if self.short_description else ""
        ) + (self.long_description if self.long_description else "")

        self.param_descriptions = {
            param.arg_name: param.description for param in self.docstring.params
        }

        self.parameters: list[AgentToolParameter] = []
        self.required_properties = []

        self.param_name_to_type = {}

        signature = inspect.signature(self.tool_function)
        type_hints = get_type_hints(self.tool_function)

        for param_name, param in signature.parameters.items():

            if param_name not in type_hints:
                raise SkyAgentToolParsingError(
                    "Parameter '{param_name}' must have a type annotation."
                )

            annotation = type_hints[param_name]

            if annotation not in self.ALLOWED_TYPES:
                raise SkyAgentToolParsingError(
                    f"Parameter '{param_name}' must be annotated with one of '{
                        list(self.ALLOWED_TYPES.keys())}' but got '{annotation}'."
                )

            self.param_name_to_type[param_name] = annotation
            json_type = self.ALLOWED_TYPES[annotation]

            self.parameters.append(
                AgentToolParameter(
                    name=param_name,
                    type=json_type,
                    description=self.param_descriptions.get(param_name, ""),
                )
            )

            if param.default == inspect.Parameter.empty:
                self.required_properties.append(param_name)

        return_annotation = type_hints.get("return")
        if return_annotation is None:
            raise SkyAgentToolParsingError("Return type annotation is required.")
        if return_annotation not in self.ALLOWED_TYPES:
            raise SkyAgentToolParsingError(
                f"Return type must be one of '{list(self.ALLOWED_TYPES.keys())}' but got '{
                    return_annotation}'."
            )

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError("Tools must implement the to dict method.")

    def validate_and_convert_input_param(
        self, input_param_name: str, input_param_value: Any
    ) -> TOOL_ARG_TYPES:

        if input_param_name not in self.param_name_to_type:
            raise SkyAgentDetrimentalError(
                f"Not defined input parameter '{
                    input_param_name}' in tool call of function '{self.name}'."
            )

        expected_type = self.param_name_to_type[input_param_name]
        conversion_func = self.PYTHON_TYPE_MAP[expected_type]

        try:
            return conversion_func(input_param_value)
        except ValueError as e:
            raise SkyAgentDetrimentalError(
                f"Failed to convert parameter '{
                    input_param_name}' to '{expected_type}': '{e}'"
            )
