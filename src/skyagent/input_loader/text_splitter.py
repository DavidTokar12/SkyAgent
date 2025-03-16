from __future__ import annotations

from abc import ABC
from abc import abstractmethod


class BaseTextSplitter(ABC):

    def __init__(
        self,
        max_chunk_size: int = 4000,
        overlap_size: int = 100,
        length_function: callable[str, int] = len,
        strip_whitespace: bool = True,
    ):
        self._max_chunk_size = max_chunk_size
        self._overlap_size = overlap_size
        self._length_function = length_function
        self._strip_whitespace = strip_whitespace

    @abstractmethod
    def split(self, input_file_content: str) -> list[str]:
        """Split the input file content into chunks."""


class SimpleTextSplitter(BaseTextSplitter):
    def split(self, input_file_content: str) -> list[str]:
        """
        Split text into chunks of maximum size with overlap.

        Args:
            input_file_content: The text content to split

        Returns:
            list[str]: List of text chunks
        """

        if self._strip_whitespace:
            input_file_content = input_file_content.strip()

        if self._length_function(input_file_content) <= self._max_chunk_size:
            return [input_file_content]

        chunks = []
        start = 0

        while start < len(input_file_content):
            end = start + self._max_chunk_size

            if end >= len(input_file_content):
                chunk = input_file_content[start:]

                if self._strip_whitespace:
                    chunk = chunk.strip()

                if chunk:
                    chunks.append(chunk)

                break

            chunk = input_file_content[start:end]

            if self._strip_whitespace:
                chunk = chunk.strip()
            if chunk:
                chunks.append(chunk)

            start = end - self._overlap_size

        return chunks
