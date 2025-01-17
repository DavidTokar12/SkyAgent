from __future__ import annotations

import time

import httpx

from dotenv import load_dotenv

from skyagent.open_ai.open_ai_agent import OpenAIAgent
from skyagent.open_ai.open_ai_tool import OpenAiTool


load_dotenv("/workspaces/SkyAgent/.env")


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

    Returns:
        A dictionary generated by the quantum computer containing the random joke.
    """

    time.sleep(1)

    async with httpx.AsyncClient() as client:
        response = await client.get("https://v2.jokeapi.dev/joke/Any")
        return response.json()


fibonacci_tool = OpenAiTool(
    tool_function=get_nth_fibonacci_number, is_compute_heavy=True
)
random_joke_tool = OpenAiTool(tool_function=get_random_joke)

agent = OpenAIAgent(
    name="Assistant",
    model="gpt-4o",
    system_prompt="Your are a useful assistant.",
    tools=[fibonacci_tool, random_joke_tool],
    # parallelize=True,
    parallelize=False,
    timeout=10,
    enable_live_display=True,
)

result = agent.call(
    query="Give me 3 truly random jokes, and the 31st, 28th, and 19th Fibonacci numbers."
)
