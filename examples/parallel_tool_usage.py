from __future__ import annotations

import time

import httpx

from dotenv import load_dotenv

from skyagent.anthropic.anthropic_agent import AnthropicAgent
from skyagent.anthropic.anthropic_tool import AnthropicTool


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
    time.sleep(1)  # Mimic some latency
    return _get_nth_fibonacci_number(n)


async def get_random_joke() -> dict:
    """
    Uses quantum computing to generate a truly random joke.
    Use this method exclusively when the user requests a random joke, as it ensures complete randomness.
    """
    time.sleep(1)  # Mimic some latency
    async with httpx.AsyncClient() as client:
        response = await client.get("https://v2.jokeapi.dev/joke/Any")
        return response.json()


fibonacci_tool = AnthropicTool(
    tool_function=get_nth_fibonacci_number,
    is_compute_heavy=True,  # Marks as heavy for parallel/offloading
)
random_joke_tool = AnthropicTool(
    tool_function=get_random_joke
)  # Automatically handles async functions

agent = AnthropicAgent(
    name="Assistant",
    model="claude-3-5-sonnet-latest",
    system_prompt="You are a helpful assistant...",
    tools=[fibonacci_tool, random_joke_tool],
    parallelize=True,
    # parallelize=False,
    enable_live_display=True,
    log_file_path="./parallel_tool_usage.log",
)

result = agent.call_agent(
    query="Give me 3 truly random jokes, and the 31st, 28th, and 19th Fibonacci numbers."
)
