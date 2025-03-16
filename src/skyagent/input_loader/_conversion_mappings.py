from __future__ import annotations

from skyagent.input_loader.file_types import BinaryFileType
from skyagent.input_loader.file_types import TextFileType


CONVERSION_MAPPINGS = {
    TextFileType.CODE: "txt",
    TextFileType.MARKDOWN: "md",
    TextFileType.JSON: "txt",
    TextFileType.YAML: "txt",
    TextFileType.XML: "txt",
    TextFileType.TEXT: "txt",
    TextFileType.CSV: "csv",
    BinaryFileType.PDF: "md",
    BinaryFileType.DOC: "md",
    BinaryFileType.XLS: "csv",
    BinaryFileType.PPT: "md",
    BinaryFileType.IMG: "b64",
}
