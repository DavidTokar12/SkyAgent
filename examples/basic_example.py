from __future__ import annotations

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic import Field

from skyagent.base.agent import Agent
from skyagent.base.chat_message import SystemMessage
from skyagent.base.chat_message import UserMessage
from skyagent.base.loggers.rich_agent_logger import RichAgentLogger
from skyagent.base.tools import Tool


load_dotenv("/workspaces/SkyAgent/.env")  # Load Anthropic or OpenAI tokens.


def evaluate_expression(expression: str) -> float:
    """Use this tool to evaluate a math expression. Use this whenever you have to calculate anything to avoid making mistakes.

    :param expression: A math expression in Python format like 'sum([8, 16, 32])' or '2 ** 8'.
    """
    return eval(expression)


tool = Tool(tool_function=evaluate_expression)


# You can define a response format for all models.
class CalculationResponse(BaseModel):
    chain_of_thought: str = Field(
        description="The chain of thought that led to the final result."
    )
    final_result: float
    your_favorite_number: str


system_prompt = """
Your are a precise math problem solver. 
To ensure precision, use the math tools at your disposal. Even if you know the answer, use the tools to avoid mistakes. 
"""

agent = Agent(
    agent_name="Calculator",
    # model="gpt-4o",
    # api_adapter="openai",
    model="claude-3-5-sonnet-latest",
    api_adapter="anthropic",
    system_prompt=SystemMessage(content=system_prompt),
    tools=[tool],
    logger=RichAgentLogger,
)

input_query = """
- A company produces 8,757 gadgets per month. 
- Each gadget costs $237 to manufacture. 
- The company distributes the gadgets to 15 stores equally each month.

How much does each store receive in product value (in dollars) every month?
"""


with agent._logger.live_dashboard_context():
    result = agent.call_agent_sync(
        input_chat_history=[UserMessage(content=input_query)],
        response_format=CalculationResponse,
    )
