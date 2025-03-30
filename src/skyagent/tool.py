from __future__ import annotations

import inspect
import json

from dataclasses import dataclass
from typing import Any
from typing import get_type_hints

import docstring_parser

from pydantic import BaseModel
from pydantic import Field
from pydantic import create_model

from skyagent.exceptions import SkyAgentToolParsingError
from skyagent.utils import to_strict_json_schema


@dataclass
class ToolCall:
    """
    Represents a tool call returned by the model in the chat completion.
    """

    function_name: str
    call_id: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """
    Represents the result of a tool call.
    """

    id: str
    function_name: str
    arguments: dict[str, Any]
    result: Any


@dataclass
class ToolFunctionSchema:
    """
    A tool definition to be passed to an LLM.
    """

    tool_name: str
    tool_description: str | None
    params_pydantic_model: type[BaseModel]
    params_json_schema: dict[str, Any]


class Tool:
    """
    A class representing a tool (function) the LLM can call.
    """

    def __init__(
        self,
        tool_function,
        is_compute_heavy: bool = False,
    ) -> None:

        self._tool_function: callable = tool_function
        self._is_async: bool = inspect.iscoroutinefunction(self._tool_function)
        self._is_compute_heavy: bool = is_compute_heavy
        self._tool_function_schema: ToolFunctionSchema = (
            self._parse_tool_function_schema()
        )

    def _parse_tool_function_schema(self) -> ToolFunctionSchema:

        function_name = self._tool_function.__name__

        _signature = inspect.signature(self._tool_function)
        _type_hints = get_type_hints(self._tool_function)

        if "return" not in _type_hints:
            raise SkyAgentToolParsingError(
                f"Tool function '{function_name}' is missing a return type annotation."
            )

        for param_name in _signature.parameters:
            if param_name not in _type_hints:
                raise SkyAgentToolParsingError(
                    f"Parameter '{param_name}' in tool function '{function_name}' is missing a type annotation."
                )

        _params = list(_signature.parameters.items())

        function_description, _param_descriptions = self._get_function_documentation()

        fields: dict[str, Any] = {}

        for name, param in _params:
            _annotation = _type_hints.get(name, param.annotation)
            _default = param.default

            _param_description = _param_descriptions.get(name, None)

            if param.kind == param.VAR_POSITIONAL:
                raise SkyAgentToolParsingError(
                    f"Positional arguments are not supported in tool function '{function_name}'."
                )
            elif param.kind == param.VAR_KEYWORD:
                raise SkyAgentToolParsingError(
                    f"Keyword arguments are not supported in tool function '{function_name}'."
                )
            else:
                if _default == inspect._empty:
                    fields[name] = (
                        _annotation,
                        Field(..., description=_param_description),
                    )
                else:
                    fields[name] = (
                        _annotation,
                        Field(default=_default, description=_param_description),
                    )

        try:
            dynamic_model = create_model(
                f"{function_name}_args", __base__=BaseModel, **fields
            )
            json_schema = dynamic_model.model_json_schema()
        except Exception as e:
            raise SkyAgentToolParsingError(
                f"Failed to generate JSON schema for tool function '{function_name}'. "
                f"Your arguments likely contain complex types that are not supported."
            ) from e

        json_schema = to_strict_json_schema(json_schema, path=(), root=json_schema)

        return ToolFunctionSchema(
            tool_name=function_name,
            tool_description=function_description,
            params_pydantic_model=dynamic_model,
            params_json_schema=json_schema,
        )

    def _get_function_documentation(self) -> tuple[str, dict[str, str]]:
        """
        Extracts the function's docstring and parameter descriptions.
        """

        _docstring = docstring_parser.parse(self._tool_function.__doc__)

        long_description = _docstring.long_description
        short_description = _docstring.short_description

        description = ((short_description + " ") if short_description else "") + (
            long_description if long_description else ""
        )

        param_descriptions = {
            param.arg_name: param.description for param in _docstring.params
        }

        return description, param_descriptions

    def __str__(self) -> str:
        return f"Tool(name='{self._tool_function_schema.tool_name}')"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def name(self) -> str:
        return self._tool_function_schema.tool_name

    @property
    def description(self) -> str:
        return self._tool_function_schema.tool_description

    @property
    def param_schema(self) -> dict[str, Any]:
        return self._tool_function_schema.params_json_schema

    def validate_args(self, args: dict[str, Any]) -> None:
        """
        Validates the arguments against the tool's schema.
        """
        try:
            self._tool_function_schema.params_pydantic_model.model_validate(args)
        except Exception as e:
            raise SkyAgentToolParsingError(
                f"Failed to validate arguments for tool '{self.name}'."
            ) from e
