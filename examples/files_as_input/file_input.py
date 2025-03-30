from __future__ import annotations

from dotenv import load_dotenv

from skyagent.input_loader.input_file_loader import InputFileLoader
from skyagent.messages import ModelInput


load_dotenv("/workspaces/SkyAgent/.env")


file_attachment = (
    InputFileLoader(
        input_path="/workspaces/SkyAgent/examples/files_as_input/input_directory/sky_agent_readme.md",
        output_directory_path="/workspaces/SkyAgent/examples/files_as_input/output_directory",
        split_text=False,
    )
    .load()
    .to_attachment()
)


# agent = Agent(
# agent_name="File analyzer",
# model="claude-3-5-sonnet-latest",
# api_adapter="anthropic",
# model="gpt-4o",
# api_adapter="openai",
# logger=RichAgentLogger,
# )

input_with_file = ModelInput(
    content="Explain what you see in this file...",
    attached_files=[file_attachment],
)

print(input_with_file.to_text_and_image_messages())

# result = agent.call_agent_sync(
#     input_chat_history=[input_with_file],
# )
