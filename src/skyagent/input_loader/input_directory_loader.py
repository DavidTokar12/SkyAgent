from __future__ import annotations

import json
import logging
import uuid

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING

from skyagent.base.exceptions import SkyAgentFileError
from skyagent.base.exceptions import SkyAgentValidationError
from skyagent.input_loader._defaults import DEFAULT_FILE_CONVERTER_FUNCTIONS
from skyagent.input_loader._defaults import DEFAULT_TEXT_SPLITTER
from skyagent.input_loader.input_file_loader import InputFileLoader


if TYPE_CHECKING:
    from skyagent.input_loader.file_types import BinaryFileType
    from skyagent.input_loader.text_splitter import BaseTextSplitter


logger = logging.getLogger(__name__)


@dataclass
class InputDirectoryReaderMetadata:
    id: str
    original_input_directory_path: str
    split_text: bool
    ignore_patterns: list[str]
    relative_processed_directories: list[str]


class InputDirectoryLoader:
    def __init__(
        self,
        input_directory_path: str | Path,
        output_directory_path: None | str | Path = None,
        split_text: bool = True,
        text_splitter: BaseTextSplitter = DEFAULT_TEXT_SPLITTER,
        file_converter_functions: dict[
            BinaryFileType, callable
        ] = DEFAULT_FILE_CONVERTER_FUNCTIONS,
        ignore_patterns: list[str] | None = None,
    ):
        """
        Initialize directory reader.

        Args:
            input_directory_path: Path to directory to process.
            output_directory_path: Path for output files.
            split_text: Whether to split text content.
            text_splitters: Custom text splitter(s).
            file_converter_functions: Custom file converters.
            ignore_patterns: List of glob patterns to ignore (e.g., ["*.git*", "*.pyc"]).
        """
        self._id = str(uuid.uuid4())

        self._input_directory_path = self._validate_input_directory(
            input_directory_path
        )
        self._output_directory_path = self._setup_output_directory(
            output_directory_path
        )

        self._split_text = split_text
        self._text_splitter = text_splitter
        self._file_converter_functions = file_converter_functions
        self._ignore_patterns = ignore_patterns or []

        # Keys will now be relative file paths (string form).
        self.file_loaders: dict[str, InputFileLoader] = {}

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
        """Prepare the base output directory."""
        try:
            if output_path is None:
                base_dir = Path(mkdtemp(prefix="skyagent_"))
            else:
                base_dir = Path(output_path).resolve()

            if base_dir.exists() and not base_dir.is_dir():
                raise SkyAgentFileError(
                    f"Output path exists but is not a directory: {base_dir}"
                )
            return base_dir
        except Exception as e:
            if isinstance(e, SkyAgentFileError):
                raise
            raise SkyAgentFileError(f"Failed to setup output directory: {e!s}")

    def _collect_files(self) -> list[Path]:
        """
        Collects all files in self._input_directory_path, excluding anything
        that matches an ignore pattern.
        """

        # Gather all files
        all_files = [f for f in self._input_directory_path.rglob("*") if f.is_file()]

        # Filter out ignored files
        for pattern in self._ignore_patterns:
            all_files = [f for f in all_files if not f.match(pattern)]

        return all_files

    def load(self) -> None:
        """
        Process all files in the directory recursively with a file loader.
        Maintains original directory structure in output.
        """

        self._output_directory_path.mkdir(parents=True, exist_ok=True)

        # Collect all candidate files, skipping ignores
        files_to_process = self._collect_files()
        if not files_to_process:
            logger.warning("No files to process after applying ignore patterns.")
            return

        for input_file_path in files_to_process:

            # The relative path from input_directory to the file
            relative_file_path = input_file_path.relative_to(self._input_directory_path)

            # Build the corresponding output directory so that structure is maintained
            relative_output_dir = (
                self._output_directory_path / relative_file_path.parent
            )

            relative_output_dir.mkdir(parents=True, exist_ok=True)

            try:
                file_loader = InputFileLoader(
                    input_path=input_file_path,
                    output_directory_path=relative_output_dir,
                    split_text=self._split_text,
                    text_splitter=self._text_splitter,
                    file_converter_functions=self._file_converter_functions,
                ).load()

                self.file_loaders[str(relative_file_path)] = file_loader

            except Exception as e:
                # Log error but continue processing other files
                logger.error(f"Failed to process file {input_file_path}: {e!s}")
                continue

        self._save_metadata()

    def get_file_loader(self, file_path: str | Path) -> InputFileLoader:
        """
        Get the file loader for a specific file based on relative path.

        Args:
            file_path: If you pass a full path, we derive its relative path
                       to look up in _file_loaders. If you already have a
                       relative path, you can pass that too.

        Returns:
            The corresponding InputFileLoader object.
        """
        p = Path(file_path)

        if p.is_absolute() and self._input_directory_path in p.parents:
            # Convert the absolute path to a relative path
            rel = str(p.relative_to(self._input_directory_path))
        else:
            # Assume the user gave us a relative path
            rel = str(p)

        if rel not in self.file_loaders:
            raise KeyError(f"No loader found for file (relative='{rel}').")

        return self.file_loaders[rel]

    def _save_metadata(self) -> None:
        """
        Save metadata about the processed directory to enable later loading.

        If no files were processed, we can skip or just write an empty metadata file.
        """

        relative_processed_directories = []
        for loader in self.file_loaders.values():

            relative_dir = loader._output_directory_path.relative_to(
                self._output_directory_path
            )
            rel_str = str(relative_dir)

            if rel_str not in relative_processed_directories:
                relative_processed_directories.append(rel_str)

        metadata = InputDirectoryReaderMetadata(
            id=self._id,
            original_input_directory_path=str(self._input_directory_path),
            split_text=self._split_text,
            ignore_patterns=self._ignore_patterns,
            relative_processed_directories=relative_processed_directories,
        )

        metadata_path = self._output_directory_path / "directory_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(metadata), f, indent=2)

    @classmethod
    def from_directory(
        cls,
        processed_directory_path: str | Path,
    ) -> InputDirectoryLoader:
        """
        Create an InputDirectoryReader instance from a previously processed directory.

        Args:
            processed_directory_path: Path to the directory containing processed files

        Returns:
            InputDirectoryReader instance with restored state
        """

        directory_path = Path(processed_directory_path).resolve()
        metadata_path = directory_path / "directory_metadata.json"

        if not directory_path.exists() or not directory_path.is_dir():
            raise SkyAgentValidationError(f"Invalid directory path: {directory_path}")

        if not metadata_path.exists():
            raise SkyAgentValidationError(
                f"No directory_metadata.json found in directory: {directory_path}"
            )

        with metadata_path.open("r", encoding="utf-8") as f:
            metadata_dict = json.load(f)
            metadata = InputDirectoryReaderMetadata(**metadata_dict)

        instance = cls(
            input_directory_path=metadata.original_input_directory_path,
            output_directory_path=directory_path,
            split_text=metadata.split_text,
            ignore_patterns=metadata.ignore_patterns,
        )

        instance._id = metadata.id
        instance.file_loaders = {}

        for rel_dir in metadata.relative_processed_directories:

            process_dir = directory_path / rel_dir
            if not process_dir.exists() or not process_dir.is_dir():
                logger.warning(f"Processed directory not found: {process_dir}")
                continue

            try:
                file_loader = InputFileLoader.from_directory(process_dir)

                original_dir = Path(metadata.original_input_directory_path).resolve()
                full_original_path = Path(file_loader._original_input_path).resolve()

                rel_path = full_original_path.relative_to(original_dir)
                instance.file_loaders[str(rel_path)] = file_loader

            except Exception as e:
                logger.error(f"Failed to load directory {process_dir}: {e!s}")
                continue

        if not instance.file_loaders:
            logger.warning("No file loaders could be restored from the directory.")

        return instance
