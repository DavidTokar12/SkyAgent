from __future__ import annotations

from pathlib import Path

from skyagent.input_loader.input_directory_loader import InputDirectoryLoader
from tests.utils import compare_file_loaders


def test_directory_reader_from_directory(temp_output_dir):
    """Test loading a processed directory structure"""

    input_dir = Path(__file__).parent / "input_file_parsing_inputs"

    assert input_dir.exists(), f"Test directory not found: {input_dir}"

    original_reader = InputDirectoryLoader(
        input_directory_path=input_dir,
        output_directory_path=temp_output_dir,
        ignore_patterns=["*.git"],
    )

    original_reader.load()

    processed_dir = original_reader._output_directory_path

    metadata_path = processed_dir / "directory_metadata.json"

    assert metadata_path.exists(), "Directory metadata file wasn't created"

    loaded_reader = InputDirectoryLoader.from_directory(processed_dir)

    assert original_reader._id == loaded_reader._id, "IDs don't match"
    assert (
        original_reader._split_text == loaded_reader._split_text
    ), "Split text settings don't match"
    assert (
        original_reader._ignore_patterns == loaded_reader._ignore_patterns
    ), "Ignore patterns don't match"

    assert len(original_reader.file_loaders) == len(
        loaded_reader.file_loaders
    ), f"Number of file loaders don't match: {len(original_reader.file_loaders)} vs {len(loaded_reader.file_loaders)}"

    for original_path, original_loader in original_reader.file_loaders.items():
        assert (
            original_path in loaded_reader.file_loaders
        ), f"Missing file loader for {original_path}"

        loaded_loader = loaded_reader.file_loaders[original_path]

        compare_file_loaders(original_loader, loaded_loader)
