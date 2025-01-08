from __future__ import annotations

import ast
import inspect
import logging

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import get_type_hints

import docstring_parser

from skyagent.base_classes import AgentDetrimentalError


if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class AgentToolParsingError(Exception):
    """Base exception for AgentTool-related errors."""


@dataclass
class AgentToolParameter:
    """Represents a parameter of a tool."""

    name: str
    type: str
    description: str


class OpenAITool:
    """
    A class representing a tool (function) the LLM can call.
    Analyzes the function's signature and docstring to build a JSON schema.
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

    def __init__(self, tool_function, additional_properties: bool = False) -> None:
        """
        Initialize a Tool by analyzing the provided function.

        :param func: The function to expose as a tool.
        :param additional_properties: Whether to allow additional
                                      properties when calling this tool.
        """
        self.tool_function = tool_function
        self.name = self.tool_function.__name__
        self.additional_properties = additional_properties

        self.is_async = inspect.iscoroutinefunction(tool_function)

        try:
            self.docstring = docstring_parser.parse(self.tool_function.__doc__)
        except Exception as e:
            raise AgentToolParsingError(
                "Failed to parse docstring for function '%s': '%s'", self.name, e
            )

        self.long_description = self.docstring.long_description
        self.short_description = self.docstring.short_description

        if not self.short_description and not self.long_description:
            raise AgentToolParsingError(
                "Function '%s' must have a function description in it's docstring.",
                self.name,
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

        signature = inspect.signature(tool_function)
        type_hints = get_type_hints(tool_function)

        for param_name, param in signature.parameters.items():

            if param_name not in type_hints:
                raise AgentToolParsingError(
                    "Parameter '%s' must have a type annotation.", param_name
                )

            annotation = type_hints[param_name]

            if annotation not in self.ALLOWED_TYPES:
                raise AgentToolParsingError(
                    "Parameter '%s' must be annotated with one of '%s' but got '%s'.",
                    param_name,
                    list(self.ALLOWED_TYPES.keys()),
                    annotation,
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
            raise AgentToolParsingError("Return type annotation is required.")
        if return_annotation not in self.ALLOWED_TYPES:
            raise AgentToolParsingError(
                "Return type must be one of '%s' but got '%s'.",
                list(self.ALLOWED_TYPES.keys()),
                return_annotation,
            )

        logger.debug(
            "Initialized AgentTool for function '%s' with schema: %s",
            self.name,
            self.to_dict(),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Returns a dictionary that follows the ChatGPT-style function-calling
        schema. This can be passed directly to openai LLMs that support
        function calling.
        """
        properties_dict = {
            param.name: {
                "type": param.type,
                "description": param.description,
            }
            for param in self.parameters
        }

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties_dict,
                    "required": self.required_properties,
                    "additionalProperties": self.additional_properties,
                },
            },
        }

    def validate_and_convert_input_param(
        self, input_param_name: str, input_param_value: Any
    ) -> Any:

        if input_param_name not in self.param_name_to_type:
            raise AgentDetrimentalError(
                "Not defined input parameter '%s' in tool call of function '%s'.",
                input_param_name,
                self.name,
            )

        expected_type = self.param_name_to_type[input_param_name]
        conversion_func = self.PYTHON_TYPE_MAP[expected_type]

        try:
            return conversion_func(input_param_value)
        except ValueError as e:
            raise AgentDetrimentalError(
                "Failed to convert parameter '%s' to '%s': '%s'",
                input_param_name,
                expected_type,
                e,
            )