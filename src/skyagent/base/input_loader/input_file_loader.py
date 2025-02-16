from __future__ import annotations

import logging
import uuid

from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING

from skyagent.base.exceptions import SkyAgentFileError
from skyagent.base.exceptions import SkyAgentValidationError
from skyagent.base.input_loader.file_converter import default_csv_converter
from skyagent.base.input_loader.file_converter import default_doc_converter
from skyagent.base.input_loader.file_converter import default_img_converter
from skyagent.base.input_loader.file_converter import default_pdf_converter
from skyagent.base.input_loader.file_converter import default_ppt_converter
from skyagent.base.input_loader.file_extensions import BINARY_FILE_EXTENSIONS
from skyagent.base.input_loader.file_extensions import TEXT_FILE_EXTENSIONS
from skyagent.base.input_loader.file_types import BinaryFileType
from skyagent.base.input_loader.file_types import TextFileType
from skyagent.utils import is_binary_string


if TYPE_CHECKING:
    from skyagent.base.input_loader.text_splitter import TextSplitter

# TODO: make it possible to load from input folder
# TODO: Write tests for each file type

logger = logging.getLogger(__name__)


class InputFileLoader:
    def __init__(
        self,
        input_path: str | Path,
        output_directory_path: None | str | Path = None,
        split_text: bool = True,
        text_splitter: TextSplitter | None = None,
        file_converter_functions: dict[BinaryFileType, callable] | None = None,
    ):

        self.id = str(uuid.uuid4())
        self.input_path = self._validate_input(input_path)
        self.output_directory_path = self._setup_output_directory(
            output_path=output_directory_path, file_name=self.input_path.stem
        )

        self.file_type: BinaryFileType | TextFileType | None = None
        self.extracted_text_file_paths: list[Path] = []
        self.extracted_image_paths: list[Path] = []
        self.chunk_lengths = []

        self.split_text = split_text
        if self.split_text and text_splitter is None:
            from skyagent.base.input_loader.text_splitter import SimpleTextSplitter

            self.text_splitter = SimpleTextSplitter()

        self.file_converter_functions = {
            BinaryFileType.PDF: default_pdf_converter,
            BinaryFileType.DOC: default_doc_converter,
            BinaryFileType.CSV: default_csv_converter,
            BinaryFileType.PPT: default_ppt_converter,
            BinaryFileType.IMG: default_img_converter,
        }

        if file_converter_functions:
            self.file_converter_functions.update(file_converter_functions)

    def _validate_input(self, input_path: str | Path) -> str | Path:

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
            raise SkyAgentValidationError(
                f"Invalid input path: {input_path}. Error: {e!s}"
            )

    def _setup_output_directory(
        self, output_path: str | Path | None, file_name: str
    ) -> Path:
        """
        Setup the output directory structure.

        Args:
            output_path: Base directory for outputs
            file_name: Name of the input file (without extension)

        Returns:
            Path to the created file-specific directory

        Raises:
            SkyAgentFileError: If directory creation fails or if output_path doesn't exist
        """
        try:
            if output_path is None:
                base_dir = Path(mkdtemp(prefix="skyagent_"))
            else:
                base_dir = Path(output_path).resolve()
                if not base_dir.exists():
                    raise SkyAgentFileError(
                        f"Output directory does not exist: {base_dir}"
                    )
                if not base_dir.is_dir():
                    raise SkyAgentFileError(
                        f"Output path exists but is not a directory: {base_dir}"
                    )

            file_dir = base_dir / f"{file_name}_{self.id}"
            file_dir.mkdir(parents=True, exist_ok=False)

            return file_dir

        except Exception as e:
            if isinstance(e, SkyAgentFileError):
                raise
            raise SkyAgentFileError(f"Failed to setup output directory: {e!s}")

    def _get_output_path(
        self,
        file_type: BinaryFileType | TextFileType,
        section_idx: int | None = None,
        chunk_idx: int | None = None,
        is_image: bool = False,
    ) -> Path:
        """Generate standardized output file path."""

        if is_image:
            return self.output_directory_path / f"image_{section_idx}_b64.txt"

        base_name = f"file_{file_type.value}"

        if section_idx is not None:
            base_name += f"_section_{section_idx}"
        if chunk_idx is not None:
            base_name += f"_chunk_{chunk_idx}"

        extension = ".csv" if file_type == BinaryFileType.CSV else ".txt"

        return self.output_directory_path / f"{base_name}{extension}"

    def _process_content(
        self,
        content: str,
        file_type: BinaryFileType | TextFileType,
        section_idx: int | None = None,
    ) -> list[Path]:
        """Process and save content, with or without splitting."""
        output_paths = []

        should_split = self.split_text and file_type not in (
            BinaryFileType.CSV,
            BinaryFileType.IMG,
        )

        if should_split:

            chunks = self.text_splitter.split(content)

            for chunk_idx, chunk in enumerate(chunks, 1):
                output_path = self._get_output_path(file_type, section_idx, chunk_idx)
                self.chunk_lengths.append(self.text_splitter._length_function(chunk))
                output_path.write_text(chunk)
                output_paths.append(output_path)
        else:

            output_path = self._get_output_path(file_type, section_idx, chunk_idx=1)

            self.chunk_lengths.append(self.text_splitter._length_function(content))

            output_path.write_text(content)
            output_paths.append(output_path)

        return output_paths

    def _load_text_file(self):

        self.file_type = TextFileType.TEXT
        for file_type, extensions in TEXT_FILE_EXTENSIONS.items():
            if self.input_path.suffix in extensions:
                self.file_type = TextFileType[file_type.upper()]
                break

        content = self.input_path.read_text()
        self.extracted_text_file_paths = self._process_content(content, self.file_type)

    def _load_binary_file(self):

        self.file_type = None
        for file_type, extensions in BINARY_FILE_EXTENSIONS.items():
            if self.input_path.suffix in extensions:
                self.file_type = BinaryFileType[file_type.upper()]
                break

        if self.file_type is None:
            raise SkyAgentValidationError(
                f"Unsupported binary file type: {self.input_path.suffix}. "
                f"Supported types are: {', '.join(sorted({ext for exts in BINARY_FILE_EXTENSIONS.values() for ext in exts}))}"
            )

        # Convert file
        try:
            text_contents, base64_images = self.file_converter_functions[
                self.file_type
            ](self.input_path)
        except Exception as e:
            raise SkyAgentFileError(
                f"Failed to convert file: {self.input_path}. Error: {e!s}"
            )

        # Process text contents
        self.extracted_text_file_paths = []
        for section_idx, content in enumerate(text_contents, 1):
            self.extracted_text_file_paths.extend(
                self._process_content(content, self.file_type, section_idx)
            )

        # Save images
        for img_idx, image_data in enumerate(base64_images, 1):
            image_path = self._get_output_path(self.file_type, img_idx, is_image=True)
            image_path.write_text(image_data)
            self.extracted_image_paths.append(image_path)

    def load(self):
        with self.input_path.open("rb") as f:
            if is_binary_string(f.read(1024)):
                self._load_binary_file()
            else:
                self._load_text_file()
