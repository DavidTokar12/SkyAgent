from __future__ import annotations

from dataclasses import dataclass
from typing import Self


@dataclass
class Usage:
    requests: int = 0

    input_tokens: int = 0
    output_tokens: int = 0

    details: dict[str, int] | None = None

    def add(self, other: Usage) -> Self:
        """
        Add another Usage object to this one.

        This method:
        - Adds the requests and token counts
        - Merges the details dictionaries by:
          - Adding values for keys that exist in both
          - Including keys that only exist in one

        Args:
            other: Another Usage object to add to this one

        Returns:
            Self: Returns self for method chaining
        """

        self.requests += other.requests
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens

        if other.details:
            if self.details is None:
                # If self has no details but other does, copy other's details
                self.details = other.details.copy()
            else:
                for key, value in other.details.items():
                    if key in self.details:
                        # Add values for keys that exist in both
                        self.details[key] += value
                    else:
                        # Add keys that only exist in other
                        self.details[key] = value
        return self
