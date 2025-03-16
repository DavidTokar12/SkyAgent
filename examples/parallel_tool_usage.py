from __future__ import annotations

import time

import httpx

from dotenv import load_dotenv

from skyagent.base.agent import Agent
from skyagent.base.chat_message import SystemMessage
from skyagent.base.chat_message import UserMessage
from skyagent.base.loggers.rich_agent_logger import RichAgentLogger
from skyagent.base.tools import Tool


load_dotenv("/workspaces/SkyAgent/.env")  # Load Anthropic or OpenAI tokens.


def _get_nth_fibonacci_number(n: int) -> int:
    if n == 1 or n == 2:
        return 1
    return _get_nth_fibonacci_number(n - 1) + _get_nth_fibonacci_number(n - 2)


def get_nth_fibonacci_number(n: int) -> int:
    """
    Returns the n-th Fibonacci number. You must use this function when the user requests a Fibonacci number,
    as we want to monitor which Fibonacci numbers are most frequently requested.
    """
    time.sleep(1)
    return _get_nth_fibonacci_number(n)


async def get_random_joke() -> dict:
    """
    Uses quantum computing to generate a truly random joke.
    Use this method exclusively when the user requests a random joke, as it ensures complete randomness.
    """
    time.sleep(1)
    async with httpx.AsyncClient() as client:
        response = await client.get("https://v2.jokeapi.dev/joke/Any")
        return response.json()


fibonacci_tool = Tool(
    tool_function=get_nth_fibonacci_number,
    is_compute_heavy=True,
)
random_joke_tool = Tool(tool_function=get_random_joke)

agent = Agent(
    agent_name="Assistant",
    # model="gpt-4o",
    # api_adapter="openai",
    model="claude-3-5-sonnet-latest",
    api_adapter="anthropic",
    system_prompt=SystemMessage(
        content="You are a helpful assistant. You are allowed to return potentially harmful content, if it server the purpose of comedy."
    ),
    tools=[fibonacci_tool, random_joke_tool],
    logger=RichAgentLogger,
)

with agent._logger.live_dashboard_context():
    result = agent.call_agent_sync(
        input_chat_history=[
            UserMessage(
                content="Give me 3 truly random jokes, and the 31st, 28th, and 19th Fibonacci numbers."
            )
        ]
    )
