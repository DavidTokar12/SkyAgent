from __future__ import annotations

from enum import Enum


class TextFileType(Enum):
    CODE = "code"
    MARKDOWN = "markdown"
    JSON = "json"
    YAML = "yaml"
    XML = "xml"
    TEXT = "text"
    CSV = "csv"


class BinaryFileType(Enum):
    PDF = "pdf"
    DOC = "doc"
    PPT = "ppt"
    XLS = "xls"
    IMG = "img"
