from __future__ import annotations

from dotenv import load_dotenv

from skyagent.open_ai.open_ai_agent import OpenAIAgent
from skyagent.open_ai.open_ai_tool import OpenAITool


load_dotenv("/workspaces/SkyAgent/.env")


def evaluate_expression(expression: str) -> float:
    """Evaluates a math expression in Python format. Use this whenever you need to evaluate a math expression.

    :param expression: A math expression in Python format like 'sum([8, 16, 32])' or '2 ** 8'.
    """
    return eval(expression)


tool = OpenAITool(tool_function=evaluate_expression)

agent = OpenAIAgent(
    name="Calculator",
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

print(result)
