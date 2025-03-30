from __future__ import annotations

import asyncio

from skyagent.agent import Agent
from skyagent.messages import ModelInput
from skyagent.messages import SystemPrompt
from skyagent.messages import UserPrompt
from skyagent.providers.predefined_providers.openai import OpenAiProvider
from skyagent.tool import Tool


def evaluate_expression(expression: str) -> float:
    """
    Use this tool to evaluate a Python math expression.

    Args:
        expression: expression in Python format like 'sum([8, 16, 32])' or '2 ** 8'.
    """
    return eval(expression)


tool = Tool(tool_function=evaluate_expression)


agent = Agent(
    model="gpt-4o",
    provider=OpenAiProvider,
    name="Calculator",
    system_prompt=SystemPrompt(content="system"),
    tools=[tool],
)


async def main():
    input_query = """user"""

    user_prompt = UserPrompt(content=input_query)

    # query = "user"

    input_chat_history = [ModelInput(message_parts=[user_prompt])]

    await agent.run(input_chat_history=input_chat_history)


asyncio.run(main())


# from dataclasses import dataclass
# import copy
# from openai import OpenAI
# from enum import Enum

# from dotenv import load_dotenv
# from pydantic import BaseModel
# from anthropic import Anthropic
# from skyagent.agent import Agent
# from skyagent.messages import SystemMessage
# from skyagent.messages import OutgoingMessage
# from skyagent.base.loggers.rich_agent_logger import RichAgentLogger
# from skyagent.tool import Tool


# load_dotenv("/workspaces/SkyAgent/.env")

# # if any description is None, ok
# # if any type is not parsable not ok


# def evaluate_expression(expression: str) -> float:
#     """
#     Use this tool to evaluate a Python math expression.

#     Args:
#         expression: expression in Python format like 'sum([8, 16, 32])' or '2 ** 8'.
#     """
#     return eval(expression)


# class IntNumber(int, Enum):
#     one = 1
#     two = 2
#     three = 3
#     four = 4
#     five = 5


# class StrNumber(str, Enum):
#     one = "1"
#     two = "2"
#     three = "3"
#     four = "4"
#     five = "5"


# class CallableNumber(list, Enum):
#     one = [1]


# class OtherStuff(BaseModel):
#     xd: int
#     dx: str


# @dataclass
# class Stuff:
#     other_stuff: OtherStuff

#     number: str

#     one_number: int

#     numbers: list[int]

#     str_number: str | None


# def tool_call_with_enum(xd: Stuff) -> int:
#     """
#     Call this tool with your favorite parameters.
#     """
#     return "Good job!"


# # client = OpenAI()
# client = Anthropic()

# tool = Tool(tool_function=tool_call_with_enum)
# schema = tool._tool_function_schema

# print(schema)

# print(
#     client.messages.create(
#         model="claude-3-5-sonnet-latest",
#         max_tokens=1000,
#         messages=[
#             {
#                 "role": "user",
#                 "content": "Please use the tool and fill it with your favorite values.",
#             },
#         ],
#         tools=[
#             {
#                 "name": schema.tool_name,
#                 "description": schema.tool_description,
#                 "input_schema": schema.params_json_schema,
#             }
#         ],
#     )
# )
# # print(
# #     client.chat.completions.create(
# #         model="gpt-4o",
# #         messages=[
# #             {
# #                 "role": "user",
# #                 "content": "Please use the tool and fill it with your favorite values.",
# #             },
# #         ],
# #         tools=[
# #             {
# #                 "type": "function",
# #                 "function": {
# #                     "name": schema.tool_name,
# #                     "description": schema.tool_description,
# #                     "strict": True,
# #                     "parameters": schema.params_json_schema,
# #                 },
# #             }
# #         ],
# #     )
# # )


# # print(tool)

# # # You can define a response format for all models.
# # class CalculationResponse(BaseModel):
# #     chain_of_thought: str = Field(
# #         description="The chain of thought that led to the final result."
# #     )
# #     final_result: float
# #     your_favorite_number: str


# # system_prompt = """
# # Your are a precise math problem solver.
# # To ensure precision, use the math tools at your disposal. Even if you know the answer, use the tools to avoid mistakes.
# # """

# # agent = Agent(
# #     agent_name="Calculator",
# #     # model="gpt-4o",
# #     # api_adapter="openai",
# #     model="claude-3-5-sonnet-latest",
# #     api_adapter="anthropic",
# #     system_prompt=SystemMessage(content=system_prompt),
# #     tools=[tool],
# #     logger=RichAgentLogger,
# # )

# # input_query = """
# # - A company produces 8,757 gadgets per month.
# # - Each gadget costs $237 to manufacture.
# # - The company distributes the gadgets to 15 stores equally each month.

# # How much does each store receive in product value (in dollars) every month?
# # """


# # with agent._logger.live_dashboard_context():
# #     result = agent.call_agent_sync(
# #         input_chat_history=[UserMessage(content=input_query)],
# #         response_format=CalculationResponse,
# #     )
