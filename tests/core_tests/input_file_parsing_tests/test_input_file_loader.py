from __future__ import annotations

from pathlib import Path

from skyagent.base.input_loader.input_file_loader import InputFileLoader


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

    assert original_loader.id == loaded_loader.id, "IDs don't match"
    assert (
        original_loader.file_type == loaded_loader.file_type
    ), "File types don't match"
    assert (
        original_loader.split_text == loaded_loader.split_text
    ), "Split text settings don't match"
    assert (
        original_loader.chunk_lengths == loaded_loader.chunk_lengths
    ), "Chunk lengths don't match"

    original_text_paths = {p.name for p in original_loader.extracted_text_file_paths}
    loaded_text_paths = {p.name for p in loaded_loader.extracted_text_file_paths}
    assert original_text_paths == loaded_text_paths, "Text file paths don't match"

    original_image_paths = {p.name for p in original_loader.extracted_image_paths}
    loaded_image_paths = {p.name for p in loaded_loader.extracted_image_paths}
    assert original_image_paths == loaded_image_paths, "Image file paths don't match"
