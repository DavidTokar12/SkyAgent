from __future__ import annotations

from dotenv import load_dotenv

from skyagent.agent import Agent
from skyagent.base.loggers.rich_agent_logger import RichAgentLogger
from skyagent.messages import ImageAttachment
from skyagent.messages import ModelInput


load_dotenv("/workspaces/SkyAgent/.env")  # Load Anthropic or OpenAI tokens.


agent = Agent(
    agent_name="Image analyzer",
    model="claude-3-5-sonnet-latest",
    api_provider="anthropic",
    # model="gpt-4o",
    # api_adapter="openai",
    logger=RichAgentLogger,
)

input_with_image = ModelInput(
    content="Explain what you see in this image.",
    attached_images=[
        ImageAttachment.from_file_path("/workspaces/SkyAgent/examples/image_input.jpg")
    ],
)

with agent._logger.live_dashboard_context():
    result = agent.call_agent_sync(
        input_chat_history=[input_with_image],
    )
