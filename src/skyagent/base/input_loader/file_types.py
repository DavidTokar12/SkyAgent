from __future__ import annotations

from enum import Enum


class TextFileType(Enum):
    CODE = "code"
    MARKDOWN = "markdown"
    JSON = "json"
    YAML = "yaml"
    XML = "xml"
    TEXT = "text"


class BinaryFileType(Enum):
    PDF = "pdf"
    DOC = "doc"
    CSV = "csv"
    PPT = "ppt"
    IMG = "img"
