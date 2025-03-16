from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from skyagent.input_loader.input_file_loader import InputFileLoader


def compare_file_loaders(loader1: InputFileLoader, loader2: InputFileLoader) -> None:
    """
    Compare two InputFileLoader instances to ensure they have the same
    core attributes: ID, file type, split settings, chunk lengths,
    and extracted text/image file paths.

    Raises:
        AssertionError: If any of the relevant fields differ.
    """

    assert (
        loader1._id == loader2._id
    ), f"Loader IDs do not match: {loader1._id} vs {loader2._id}"

    assert (
        loader1._file_type == loader2._file_type
    ), f"File types do not match: {loader1._file_type} vs {loader2._file_type}"

    assert (
        loader1._split_text == loader2._split_text
    ), f"Split text settings do not match: {loader1._split_text} vs {loader2._split_text}"

    assert (
        loader1._chunk_lengths == loader2._chunk_lengths
    ), f"Chunk lengths do not match: {loader1._chunk_lengths} vs {loader2._chunk_lengths}"

    loader1_text_paths = {p.name for p in loader1._extracted_text_file_paths}
    loader2_text_paths = {p.name for p in loader2._extracted_text_file_paths}
    assert (
        loader1_text_paths == loader2_text_paths
    ), f"Extracted text file paths differ: {loader1_text_paths} vs {loader2_text_paths}"

    loader1_image_paths = {p.name for p in loader1._extracted_image_paths}
    loader2_image_paths = {p.name for p in loader2._extracted_image_paths}
    assert (
        loader1_image_paths == loader2_image_paths
    ), f"Extracted image file paths differ: {loader1_image_paths} vs {loader2_image_paths}"
