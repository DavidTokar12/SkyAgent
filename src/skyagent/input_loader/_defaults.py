from __future__ import annotations

from skyagent.input_loader._default_file_converters import default_code_converter
from skyagent.input_loader._default_file_converters import default_csv_converter
from skyagent.input_loader._default_file_converters import default_doc_converter
from skyagent.input_loader._default_file_converters import default_img_converter
from skyagent.input_loader._default_file_converters import default_json_converter
from skyagent.input_loader._default_file_converters import default_markdown_converter
from skyagent.input_loader._default_file_converters import default_pdf_converter
from skyagent.input_loader._default_file_converters import default_ppt_converter
from skyagent.input_loader._default_file_converters import default_text_converter
from skyagent.input_loader._default_file_converters import default_xls_converter
from skyagent.input_loader._default_file_converters import default_xml_converter
from skyagent.input_loader._default_file_converters import default_yaml_converter
from skyagent.input_loader.file_types import BinaryFileType
from skyagent.input_loader.file_types import TextFileType
from skyagent.input_loader.text_splitter import SimpleTextSplitter


DEFAULT_TEXT_SPLITTER = SimpleTextSplitter()

DEFAULT_FILE_CONVERTER_FUNCTIONS = {
    BinaryFileType.PDF: default_pdf_converter,
    BinaryFileType.DOC: default_doc_converter,
    BinaryFileType.XLS: default_xls_converter,
    BinaryFileType.PPT: default_ppt_converter,
    BinaryFileType.IMG: default_img_converter,
    TextFileType.CODE: default_code_converter,
    TextFileType.MARKDOWN: default_markdown_converter,
    TextFileType.JSON: default_json_converter,
    TextFileType.YAML: default_yaml_converter,
    TextFileType.XML: default_xml_converter,
    TextFileType.TEXT: default_text_converter,
    TextFileType.CSV: default_csv_converter,
}
