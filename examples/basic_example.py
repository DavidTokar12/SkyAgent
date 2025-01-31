from __future__ import annotations

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic import Field

from skyagent.open_ai.open_ai_agent import OpenAIAgent
from skyagent.open_ai.open_ai_tool import OpenAiTool


load_dotenv("/workspaces/SkyAgent/.env")  # Load Anthropic or OpenAI tokens.


def evaluate_expression(expression: str) -> float:
    """Evaluates a math expression in Python format. Use this whenever you need to evaluate a math expression.

    :param expression: A math expression in Python format like 'sum([8, 16, 32])' or '2 ** 8'.
    """
    return eval(expression)


tool = OpenAiTool(tool_function=evaluate_expression)


# You can define a response format for all models.
class CalculationResponse(BaseModel):
    chain_of_thought: str = Field(
        description="The chain of thought that led to the final result."
    )
    final_result: float
    your_favorite_number: str


system_prompt = """
Your are a precise math problem solver. 
To ensure precision, use the math tools at your disposal.
"""

# agent = AnthropicAgent(
#     name="Calculator",
#     model="claude-3-5-sonnet-latest",
#     system_prompt=system_prompt,
#     tools=[tool],
#     log_file_path="./basic_example.log",
#     enable_live_display=False,
# )
agent = OpenAIAgent(
    name="Calculator",
    model="gpt-4o",
    system_prompt=system_prompt,
    tools=[tool],
    log_file_path="./basic_example.log",
    enable_live_display=True,
)


result = agent.call_agent(
    query="""
- A company produces 8,757 gadgets per month. 
- Each gadget costs $237 to manufacture. 
- The company distributes the gadgets to 15 stores equally each month.

How much does each store receive in product value (in dollars) every month?
""",
    response_format=CalculationResponse,
)
