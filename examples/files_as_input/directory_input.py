from __future__ import annotations

from dotenv import load_dotenv

from skyagent.input_loader.input_directory_loader import InputDirectoryLoader


load_dotenv("/workspaces/SkyAgent/.env")  # Load Anthropic or OpenAI tokens.


# input_file = InputFileLoader(
#     input_path="/workspaces/SkyAgent/examples/files_as_input/input_directory/sky_agent_readme.md",
#     output_directory_path="/workspaces/SkyAgent/examples/files_as_input/output_directory",
# ).load()

# file_attachments = input_file.to_attachments()

input_directory = InputDirectoryLoader(
    input_directory_path="/workspaces/SkyAgent/tests/core_tests/input_file_parsing_tests/input_file_parsing_inputs",
    output_directory_path="/workspaces/SkyAgent/examples/files_as_input/output_directory_test",
).load()
