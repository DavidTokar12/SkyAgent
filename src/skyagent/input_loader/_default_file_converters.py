from __future__ import annotations

import base64
import logging
import os
import re
import urllib.error
import urllib.request

from io import BytesIO
from pathlib import Path

from PIL import Image

from skyagent.exceptions import SkyAgentFileError


logger = logging.getLogger(__name__)


def _is_url_image(image_src: str) -> bool:
    """Check if the image source is a URL."""
    return image_src.startswith(("http://", "https://"))


def _is_base64_image(image_src: str) -> bool:
    """Check if the image source is a base64 encoded image."""
    return image_src.startswith("data:image")


def _read_image_to_base64(image_path: Path) -> str:
    """
    Read an image file from disk and return it as a base64-encoded JPEG (re-encoded to JPEG).
    """
    try:
        with Image.open(image_path) as im:
            rgb_im = im.convert("RGB")
            output = BytesIO()
            rgb_im.save(output, format="JPEG")
            jpeg_data = output.getvalue()

        return f'data:image/jpeg;base64,{base64.b64encode(jpeg_data).decode("utf-8")}'
    except Exception as e:
        raise SkyAgentFileError("Failed to read image file") from e


def _convert_to_jpeg_base64(base64_str: str) -> str:
    """
    Convert any base64-encoded image to JPEG format and return as base64.
    Always re-encodes the image to JPEG.
    """
    if "," in base64_str:
        _, data = base64_str.split(",", 1)
    else:
        raise SkyAgentFileError("Failed to convert image to JPEG")

    try:
        image_data = base64.b64decode(data)
        with Image.open(BytesIO(image_data)) as img:
            rgb_img = img.convert("RGB")

            output = BytesIO()
            rgb_img.save(output, format="JPEG")
            jpeg_data = output.getvalue()

        return f'data:image/jpeg;base64,{base64.b64encode(jpeg_data).decode("utf-8")}'
    except Exception as e:
        raise SkyAgentFileError("Failed to convert image to JPEG") from e


def _download_image_to_base64(url: str) -> str:
    """
    Download an image from a URL and return it as a base64-encoded JPEG (re-encoded to JPEG).
    """
    try:
        with urllib.request.urlopen(url) as response:
            image_data = response.read()

        with Image.open(BytesIO(image_data)) as im:
            rgb_im = im.convert("RGB")
            output = BytesIO()
            rgb_im.save(output, format="JPEG")
            jpeg_data = output.getvalue()

        return f'data:image/jpeg;base64,{base64.b64encode(jpeg_data).decode("utf-8")}'
    except (urllib.error.URLError, Exception) as e:
        raise SkyAgentFileError("Failed to download image from URL") from e


def _extract_images_from_markdown(
    markdown_content: str, original_file_path: str | Path
) -> tuple[str, list[str]]:
    """
    Extract images from markdown content and return cleaned markdown and base64 encoded JPEGs.

    Args:
        markdown_content (str): The original markdown content
        original_file_path (str | Path): The path to the original markdown file

    Returns:
        tuple[str, list[str]]: (cleaned markdown content, list of base64 encoded JPEG images)
    """
    original_path = Path(original_file_path)
    base_path = original_path.parent

    image_pattern = r"!\[([^\]]*)\]\(([^)]+)\)"
    images = []

    def handle_images(match):
        alt_text, image_src = match.groups()

        if _is_base64_image(image_src):

            try:
                jpeg_base64 = _convert_to_jpeg_base64(image_src)
                images.append(jpeg_base64)
            except SkyAgentFileError as e:
                logger.error(f"Error converting base64 image: {e}")

            return f'![{alt_text}]({image_src.split(",")[0]},)'

        elif _is_url_image(image_src):
            try:
                base64_data = _download_image_to_base64(image_src)
                images.append(base64_data)
            except SkyAgentFileError as e:
                logger.error(f"Error downloading image from URL {image_src}: {e}")

            return match.group(0)  # Keep original path in markdown

        else:
            if os.path.isabs(image_src):
                image_path = Path(image_src)
            else:
                image_path = base_path / image_src

            if image_path.exists():
                try:
                    base64_data = _read_image_to_base64(image_path)
                    images.append(base64_data)
                except SkyAgentFileError as e:
                    logger.error(f"Error reading image file {image_path}: {e}")
            else:
                logger.error(
                    f"Could not find image file: {image_path}. Image will not be included in output."
                )
            return match.group(0)  # Keep original path in markdown

    cleaned_markdown = re.sub(image_pattern, handle_images, markdown_content)
    return cleaned_markdown, images


def default_img_converter(file_path: Path) -> tuple[list[str], list[str]]:
    """
    Convert image file to base64 string.

    Args:
        file_path: Path to the image file (jpg, jpeg, or png)

    Returns:
        Tuple containing:
            - List with single base64 string representation of the image
            - Empty list (we don't extract images from images)
    """
    try:
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            return [encoded_string], []
    except FileNotFoundError:
        raise SkyAgentFileError(f"Image file not found: {file_path}")
    except Exception as e:
        raise SkyAgentFileError("Failed to process image file.") from e


def _no_special_conversion(file_path: Path) -> tuple[list[str], list[str]]:
    """
    Generic text reader: reads file content as plain text.

    Returns:
        (list_of_text, empty_list_for_images)
    """
    try:
        file_content = file_path.read_text()
        return [file_content], []
    except FileNotFoundError:
        raise SkyAgentFileError(f"File not found: {file_path}")
    except Exception as e:
        raise SkyAgentFileError("Failed to read text from file.") from e


def default_code_converter(file_path: Path) -> tuple[list[str], list[str]]:
    return _no_special_conversion(file_path=file_path)


def default_markdown_converter(file_path: Path) -> tuple[list[str], list[str]]:
    """
    Extract images from markdown content (base64, URLs, or local files),
    and return the cleaned markdown plus a list of extracted base64 images.
    """
    try:
        markdown_content = file_path.read_text()
        cleaned_markdown, base64_images = _extract_images_from_markdown(
            markdown_content=markdown_content, original_file_path=file_path
        )
        return [cleaned_markdown], base64_images
    except FileNotFoundError:
        raise SkyAgentFileError(f"Markdown file not found: {file_path}")
    except Exception as e:
        raise SkyAgentFileError(f"Failed to process markdown file: {e}") from e


def default_json_converter(file_path: Path) -> tuple[list[str], list[str]]:
    return _no_special_conversion(file_path=file_path)


def default_yaml_converter(file_path: Path) -> tuple[list[str], list[str]]:
    return _no_special_conversion(file_path=file_path)


def default_xml_converter(file_path: Path) -> tuple[list[str], list[str]]:
    return _no_special_conversion(file_path=file_path)


def default_text_converter(file_path: Path) -> tuple[list[str], list[str]]:
    return _no_special_conversion(file_path=file_path)


def default_csv_converter(file_path: Path) -> tuple[list[str], list[str]]:
    return _no_special_conversion(file_path=file_path)
