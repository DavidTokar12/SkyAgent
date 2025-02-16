from __future__ import annotations

import logging
import uuid

from pathlib import Path
from typing import TYPE_CHECKING

from skyagent.base.exceptions import SkyAgentFileError
from skyagent.base.exceptions import SkyAgentValidationError
from skyagent.base.input_loader.input_file_loader import InputFileLoader


if TYPE_CHECKING:
    from skyagent.base.input_loader.file_types import BinaryFileType
    from skyagent.base.input_loader.text_splitter import TextSplitter


logger = logging.getLogger(__name__)


class InputDirectoryReader:
    def __init__(
        self,
        input_directory_path: str | Path,
        output_directory_path: None | str | Path = None,
        split_text: bool = True,
        text_splitter: TextSplitter | None = None,
        file_converter_functions: dict[BinaryFileType, callable] | None = None,
        ignore_patterns: list[str] | None = None,
    ):
        """
        Initialize directory reader.

        Args:
            input_directory_path: Path to directory to process
            output_directory_path: Path for output files
            split_text: Whether to split text content
            text_splitter: Custom text splitter
            file_converter_functions: Custom file converters
            ignore_patterns: List of glob patterns to ignore (e.g., ["*.git*", "*.pyc"])
        """
        self.id = str(uuid.uuid4())

        self.input_directory_path = self._validate_input_directory(input_directory_path)
        self.output_directory_path = self._setup_output_directory(output_directory_path)

        self.split_text = split_text
        self.text_splitter = text_splitter
        self.file_converter_functions = file_converter_functions
        self.ignore_patterns = ignore_patterns or []

        self.file_loaders: dict[Path, InputFileLoader] = {}

    def _validate_input_directory(self, directory_path: str | Path) -> Path:
        """Validate input directory exists."""
        try:
            path = Path(directory_path).resolve()
            if not path.exists():
                raise SkyAgentValidationError(f"Directory does not exist: {path}")
            if not path.is_dir():
                raise SkyAgentValidationError(f"Path is not a directory: {path}")
            return path
        except Exception as e:
            if isinstance(e, SkyAgentValidationError):
                raise
            raise SkyAgentValidationError(
                f"Invalid directory path: {directory_path}. Error: {e!s}"
            )

    def _setup_output_directory(self, output_path: str | Path | None) -> Path:
        """Setup output directory."""
        try:
            if output_path is None:
                # Create a new directory with reader's ID
                base_dir = (
                    self.input_directory_path.parent / f"skyagent_output_{self.id}"
                )
            else:
                base_dir = Path(output_path).resolve()

            if base_dir.exists() and not base_dir.is_dir():
                raise SkyAgentFileError(
                    f"Output path exists but is not a directory: {base_dir}"
                )

            base_dir.mkdir(parents=False, exist_ok=False)
            return base_dir

        except Exception as e:
            if isinstance(e, SkyAgentFileError):
                raise
            raise SkyAgentFileError(f"Failed to setup output directory: {e!s}")

    def _should_process_file(self, file_path: Path) -> bool:
        """Check if file should be processed based on ignore patterns."""
        return not any(file_path.match(pattern) for pattern in self.ignore_patterns)

    def _get_relative_output_path(self, input_file_path: Path) -> Path:
        """Get the relative output path maintaining directory structure."""
        return (
            self.output_directory_path
            / input_file_path.relative_to(self.input_directory_path).parent
        )

    def load(self) -> None:
        """
        Process all files in the directory recursively.
        Maintains original directory structure in output.
        """

        for input_file_path in self.input_directory_path.rglob("*"):

            if not input_file_path.is_file() or not self._should_process_file(
                input_file_path
            ):
                continue

            relative_output_dir = self._get_relative_output_path(input_file_path)
            relative_output_dir.mkdir(parents=True, exist_ok=True)

            try:
                file_loader = InputFileLoader(
                    input_path=input_file_path,
                    output_directory_path=relative_output_dir,
                    split_text=self.split_text,
                    text_splitter=self.text_splitter,
                    file_converter_functions=self.file_converter_functions,
                )
                file_loader.load()

                self.file_loaders[input_file_path] = file_loader

            except Exception as e:
                # Log error but continue processing other files
                logger.error(f"Failed to process file {input_file_path}: {e!s}")
                continue

    def get_file_loader(self, file_path: str | Path) -> InputFileLoader:
        """Get file loader for a specific file."""
        path = Path(file_path).resolve()

        if path not in self.file_loaders:
            raise KeyError(f"No loader found for file: {path}")

        return self.file_loaders[path]