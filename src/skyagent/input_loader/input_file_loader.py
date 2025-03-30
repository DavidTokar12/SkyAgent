from __future__ import annotations

import json
import logging
import uuid

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from skyagent.exceptions import SkyAgentFileError
from skyagent.exceptions import SkyAgentValidationError
from skyagent.input_loader._conversion_mappings import CONVERSION_MAPPINGS
from skyagent.input_loader._defaults import DEFAULT_FILE_CONVERTER_FUNCTIONS
from skyagent.input_loader._defaults import DEFAULT_TEXT_SPLITTER
from skyagent.input_loader._file_extensions import BINARY_FILE_EXTENSIONS
from skyagent.input_loader._file_extensions import TEXT_FILE_EXTENSIONS
from skyagent.input_loader.file_types import BinaryFileType
from skyagent.input_loader.file_types import TextFileType
from skyagent.messages import FileAttachment
from skyagent.messages import ImageAttachment
from skyagent.utils import is_binary_string


if TYPE_CHECKING:
    from skyagent.input_loader.text_splitter import BaseTextSplitter


@dataclass
class InputFileLoaderMetadata:
    file_id: str
    original_input_path: str
    file_type: str  # Store as string since enums aren't JSON serializable
    split_text: bool
    chunk_lengths: list[int]

    relative_text_file_paths: list[str]
    relative_image_paths: list[str]


logger = logging.getLogger(__name__)


class InputFileLoader:
    def __init__(
        self,
        input_path: str | Path,
        output_directory_path: str | Path,
        split_text: bool = True,
        text_splitter: BaseTextSplitter = DEFAULT_TEXT_SPLITTER,
        file_converter_functions: dict[
            BinaryFileType | TextFileType, callable
        ] = DEFAULT_FILE_CONVERTER_FUNCTIONS,
    ):

        self._id = str(uuid.uuid4())
        self._original_input_path = self._validate_input(input_path)
        self._original_input_path_name = self._original_input_path.name
        self._output_directory_path = self._setup_output_directory(
            output_path=output_directory_path,
            file_name=self._original_input_path.stem,
        )

        self._file_type: BinaryFileType | TextFileType | None = None
        self._extracted_text_file_paths: list[Path] = []
        self._extracted_image_paths: list[Path] = []
        self._chunk_lengths: list[int] = []

        self._split_text = split_text
        self._text_splitter = text_splitter

        self._file_converter_functions = DEFAULT_FILE_CONVERTER_FUNCTIONS.copy()
        self._file_converter_functions.update(file_converter_functions)

    def _validate_input(self, input_path: str | Path) -> Path:
        try:
            path = Path(input_path).resolve()

            if not path.exists():
                raise SkyAgentValidationError(f"Input path does not exist: {path}")

            if not path.is_file():
                raise SkyAgentValidationError(f"Input path is not a file: {path}")

            return path
        except Exception as e:
            if isinstance(e, SkyAgentValidationError):
                raise
            raise SkyAgentValidationError(f"Invalid input path: {input_path}") from e

    def _setup_output_directory(
        self, output_path: str | Path | None, file_name: str
    ) -> Path:
        try:
            base_dir = Path(output_path).resolve()
            if not base_dir.exists():
                base_dir.mkdir(parents=True)
            if not base_dir.is_dir():
                raise SkyAgentFileError(
                    f"Output path exists but is not a directory: {base_dir}"
                )
            return base_dir / f"{file_name}_{self._id}"
        except Exception as e:
            if isinstance(e, SkyAgentFileError):
                raise
            raise SkyAgentFileError("Failed to setup output directory.") from e

    def _determine_file_type(self) -> None:
        """
        Determine whether the file is binary or text, then set the _file_type
        accordingly. For text files, default to 'TEXT' if no known extension
        is found. For binary files, raise if no extension is matched.
        """

        with self._original_input_path.open("rb") as f:
            sample_bytes = f.read(1024)

        is_bin = is_binary_string(sample_bytes)

        if not is_bin:

            for file_type, exts in TEXT_FILE_EXTENSIONS.items():
                if self._original_input_path.suffix.lower() in [
                    ext.lower() for ext in exts
                ]:
                    self._file_type = TextFileType[file_type.upper()]
                    return

            self._file_type = TextFileType.TEXT

        else:
            for file_type, exts in BINARY_FILE_EXTENSIONS.items():
                if self._original_input_path.suffix.lower() in [
                    ext.lower() for ext in exts
                ]:
                    self._file_type = BinaryFileType[file_type.upper()]
                    return

            all_exts = {ext for exts in BINARY_FILE_EXTENSIONS.values() for ext in exts}
            raise SkyAgentValidationError(
                f"Unsupported binary file type: {self._original_input_path.suffix}. "
                f"Supported types: {', '.join(sorted(all_exts))}"
            )

    def _convert_files_to_text_and_images(self) -> tuple[list[str], list[str]]:
        """
        Convert the file to text sections and base64 images.
        Returns:
          (text_sections, base64_images)
            text_sections = list[str], each item is a 'section' (e.g. a sheet in Excel).
            base64_images = list[str], each item is a base64 string for an image.
        """

        try:
            converter_func = self._file_converter_functions[self._file_type]
            text_contents, base64_images = converter_func(self._original_input_path)
        except Exception as e:
            raise SkyAgentFileError(
                f"Failed to convert file: {self._original_input_path}. Error: {e}"
            ) from e
        return text_contents, base64_images

    def _split_text_sections(self, text_sections: list[str]) -> list[list[str]]:
        """
        For each 'section' in text_sections, optionally split into smaller chunks.
        Returns a list of lists:
          [ [chunk1_of_section1, chunk2_of_section1, ...],
            [chunk1_of_section2, chunk2_of_section2, ...],
            ...
          ]
        """

        # skip splitting for certain binary types
        should_split = self._split_text and self._file_type not in (
            TextFileType.CSV,
            BinaryFileType.XLS,
            BinaryFileType.IMG,
        )

        splitted_sections = []
        if should_split:
            for section_content in text_sections:
                chunks = self._text_splitter.split(section_content)
                splitted_sections.append(chunks)
        else:
            # Each entire section remains one chunk
            for section_content in text_sections:
                splitted_sections.append([section_content])

        return splitted_sections

    @staticmethod
    def get_output_path_for_chunk(
        file_type: BinaryFileType | TextFileType,
        output_directory_path: Path,
        section_idx: int,
        chunk_idx: int,
    ) -> Path:
        """
        Generate an output file path for either text chunk or image.
        """
        return (
            output_directory_path
            / f"file_{file_type.value}_section_{section_idx}_chunk_{chunk_idx}.{CONVERSION_MAPPINGS[file_type]}"
        )

    @staticmethod
    def get_output_path_for_extracted_image(
        output_directory_path: Path,
        image_idx: int,
    ) -> Path:
        """
        Generate an output file path for either text chunk or image.
        """
        return (
            output_directory_path
            / f"image_{image_idx}.{CONVERSION_MAPPINGS[BinaryFileType.IMG]}"
        )

    def _save_files(
        self, splitted_sections: list[list[str]], base64_images: list[str]
    ) -> None:
        """
        1. Saves the pre-split sections to disk.
        2. Saves images (in base64 form) to disk.
        3. Updates self._extracted_text_file_paths, self._extracted_image_paths, self._chunk_lengths
        """

        # Save text chunks
        for section_idx, chunks in enumerate(splitted_sections, start=1):
            for chunk_idx, chunk in enumerate(chunks, start=1):

                output_path = self.get_output_path_for_chunk(
                    file_type=self._file_type,
                    output_directory_path=self._output_directory_path,
                    section_idx=section_idx,
                    chunk_idx=chunk_idx,
                )

                output_path.write_text(chunk, encoding="utf-8")

                self._extracted_text_file_paths.append(output_path)
                self._chunk_lengths.append(self._text_splitter._length_function(chunk))

        for img_idx, image_data in enumerate(base64_images, start=1):

            image_path = self.get_output_path_for_extracted_image(
                output_directory_path=self._output_directory_path,
                image_idx=img_idx,
            )

            image_path.write_text(image_data, encoding="utf-8")

            self._extracted_image_paths.append(image_path)

    def _save_metadata(self) -> None:
        """
        Write a JSON file containing metadata about the load process.
        """
        if self._file_type is None:
            raise SkyAgentValidationError("No file has been processed yet.")

        metadata = InputFileLoaderMetadata(
            file_id=self._id,
            original_input_path=str(self._original_input_path),
            file_type=self._file_type.name if self._file_type else None,
            split_text=self._split_text,
            chunk_lengths=self._chunk_lengths,
            relative_text_file_paths=[
                str(p.relative_to(self._output_directory_path))
                for p in self._extracted_text_file_paths
            ],
            relative_image_paths=[
                str(p.relative_to(self._output_directory_path))
                for p in self._extracted_image_paths
            ],
        )

        metadata_path = self._output_directory_path / "file_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(metadata), f, indent=2)

    def load(self) -> InputFileLoader:
        """
        Load the input file and process it.
        The input file is converted to text and it's images are extracted using the defined file converter.
        The extracted text content is split into chunks using the defined text splitter.
        """

        # Create the output directory (fail if already exists)
        self._output_directory_path.mkdir(parents=True, exist_ok=False)

        # Step 1: Determine the file type
        self._determine_file_type()

        # Step 2: Convert file to text sections & images
        text_sections, base64_images = self._convert_files_to_text_and_images()

        # Step 3: Split each text section if needed
        splitted_sections = self._split_text_sections(text_sections)

        # Step 4: Save text (already split) and images
        self._save_files(splitted_sections, base64_images)

        # Step 5: Save metadata
        self._save_metadata()

        return self

    def to_attachment(self) -> FileAttachment:
        """
        Convert the processed files to a FileAttachment.
        """

        if self._split_text:
            raise SkyAgentValidationError(
                "Cannot convert to attachments when split_text is True. "
                "Re-load the file with split_text set to False, or use a file interactor to give the model access to the text chunks."
            )

        extracted_images = []

        for image_path in self._extracted_image_paths:
            extracted_images.append(
                ImageAttachment(
                    file_name=self._original_input_path_name,
                    base_64=image_path.read_text(),
                )
            )

        return FileAttachment(
            file_name=self._original_input_path_name,
            original_file_type=self._file_type.name,
            text_content=self._extracted_text_file_paths[0].read_text(),
            extracted_images=extracted_images,
        )

    @classmethod
    def from_directory(cls, directory_path: str | Path) -> InputFileLoader:
        """
        Create an InputFileLoader instance from a previously processed directory.

        Args:
            directory_path: Path to the directory containing processed files

        Returns:
            InputFileLoader instance with restored state
        """

        directory_path = Path(directory_path).resolve()
        metadata_path = directory_path / "file_metadata.json"

        if not directory_path.exists() or not directory_path.is_dir():
            raise SkyAgentValidationError(f"Invalid directory path: {directory_path}")

        if not metadata_path.exists() and not metadata_path.is_file():
            raise SkyAgentValidationError(
                f"No metadata.json found in directory: {directory_path}"
            )

        try:
            with metadata_path.open("r") as f:
                metadata_dict = json.load(f)
                metadata = InputFileLoaderMetadata(**metadata_dict)
        except Exception as e:
            raise SkyAgentValidationError(
                f"Failed to read metadata file: {metadata_path}"
            ) from e

        instance = cls(
            input_path=metadata.original_input_path,
            output_directory_path=directory_path.parent,
            split_text=metadata.split_text,
        )

        instance._id = metadata.file_id
        instance._output_directory_path = directory_path

        if metadata.file_type in BinaryFileType.__members__:
            instance._file_type = BinaryFileType[metadata.file_type]
        elif metadata.file_type in TextFileType.__members__:
            instance._file_type = TextFileType[metadata.file_type]

        instance._chunk_lengths = metadata.chunk_lengths

        instance._extracted_text_file_paths = [
            directory_path / p for p in metadata.relative_text_file_paths
        ]
        instance._extracted_image_paths = [
            directory_path / p for p in metadata.relative_image_paths
        ]

        missing_files = [
            p
            for p in instance._extracted_text_file_paths
            + instance._extracted_image_paths
            if not p.exists()
        ]

        if missing_files:
            raise SkyAgentValidationError(
                f"Some processed files are missing: {missing_files}"
            )

        return instance
