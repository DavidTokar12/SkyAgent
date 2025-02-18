from __future__ import annotations

from pathlib import Path

from skyagent.base.input_loader.input_directory_loader import InputDirectoryReader


def test_directory_reader_from_directory(temp_output_dir):
    """Test loading a processed directory structure"""

    input_dir = Path(__file__).parent / "input_file_parsing_inputs"

    assert input_dir.exists(), f"Test directory not found: {input_dir}"

    original_reader = InputDirectoryReader(
        input_directory_path=input_dir,
        output_directory_path=temp_output_dir,
        ignore_patterns=["*.git*", "*.pyc"],
    )
    original_reader.load()

    processed_dir = original_reader.output_directory_path

    metadata_path = processed_dir / "directory_metadata.json"
    assert metadata_path.exists(), "Directory metadata file wasn't created"

    loaded_reader = InputDirectoryReader.from_directory(processed_dir)

    assert original_reader.id == loaded_reader.id, "IDs don't match"
    assert (
        original_reader.split_text == loaded_reader.split_text
    ), "Split text settings don't match"
    assert (
        original_reader.ignore_patterns == loaded_reader.ignore_patterns
    ), "Ignore patterns don't match"

    assert len(original_reader.file_loaders) == len(
        loaded_reader.file_loaders
    ), f"Number of file loaders don't match: {len(original_reader.file_loaders)} vs {len(loaded_reader.file_loaders)}"

    for original_path, original_loader in original_reader.file_loaders.items():
        assert (
            original_path in loaded_reader.file_loaders
        ), f"Missing file loader for {original_path}"
        loaded_loader = loaded_reader.file_loaders[original_path]

        assert (
            original_loader.id == loaded_loader.id
        ), f"Loader IDs don't match for {original_path}"
        assert (
            original_loader.file_type == loaded_loader.file_type
        ), f"File types don't match for {original_path}"
        assert len(original_loader.extracted_text_file_paths) == len(
            loaded_loader.extracted_text_file_paths
        ), f"Number of extracted files don't match for {original_path}"
