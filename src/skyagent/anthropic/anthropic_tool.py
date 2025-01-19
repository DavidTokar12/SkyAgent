from __future__ import annotations

from typing import Any

from skyagent.base.tools import BaseTool


class AnthropicTool(BaseTool):

    def to_dict(self) -> dict[str, Any]:

        properties_dict = {
            param.name: {
                "type": param.type,
                "description": param.description,
            }
            for param in self.parameters
        }

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties_dict,
                "required": self.required_properties,
            },
        }
