from __future__ import annotations

from pathlib import Path

from skyagent.input_loader.input_file_loader import InputFileLoader
from tests.utils import compare_file_loaders


def test_file_loader_from_directory(temp_output_dir):
    """Test loading a processed directory reconstructs the same state"""

    input_file = (
        Path(__file__).parent / "input_file_parsing_inputs" / "pdf" / "lorem_ipsum.pdf"
    )

    assert input_file.exists(), f"Test file not found: {input_file}"

    original_loader = InputFileLoader(
        input_path=input_file, output_directory_path=temp_output_dir
    )
    original_loader.load()

    processed_dir = next(temp_output_dir.iterdir())

    assert processed_dir.is_dir(), "No directory was created"

    loaded_loader = InputFileLoader.from_directory(processed_dir)

    compare_file_loaders(original_loader, loaded_loader)
