from __future__ import annotations

from pathlib import Path

from skyagent.input_loader._default_file_converters import (
    _extract_text_and_images_from_html,
)


def test_extract_text_and_images_from_html():
    input_dir = Path(__file__).parent / "html_parsing_inputs"

    input_file = input_dir / "test_image_extraction.html"

    assert input_file.exists(), f"Test file not found: {input_file}"

    with open(input_file) as html_file:
        html_content = html_file.read()

    md_content, base64_images = _extract_text_and_images_from_html(
        html_content=html_content, original_file_path=input_file
    )

    assert md_content is not None
    assert len(base64_images) == 3
