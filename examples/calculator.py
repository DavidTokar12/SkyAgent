from __future__ import annotations

from dotenv import load_dotenv
from openai import OpenAI

from skyagent.agent import Agent
from skyagent.agent_tool import AgentTool


load_dotenv("/workspaces/SkyAgent/.env")

client = OpenAI()


def evaluate_expression(expression: str) -> float:
    """Evaluates a math expression in Python format. Use this whenever you need to evaluate a math expression.

    :param expression: A math expression in Python format like 'sum([8, 16, 32])' or '2 ** 8'.
    """
    return eval(expression)


tool = AgentTool(func=evaluate_expression)
agent = Agent(
    name="Calculator",
    client=client,
    model="gpt-4o",
    system_prompt="Your are a precise math problem solver.",
    tools=[tool],
)


result = agent.call(
    query="""
A company produces 8,757 gadgets per month. Each gadget costs $237 to manufacture. The company distributes the gadgets to 15 stores equally each month.
How much does each store receive in product value (in dollars) every month?
"""
)

for message in result["history"]:
    print(message)
