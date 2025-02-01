from __future__ import annotations

from dotenv import load_dotenv

from skyagent.anthropic.anthropic_agent import AnthropicAgent
from skyagent.anthropic.anthropic_tool import AnthropicTool
from skyagent.environment_interactors.unix_shell_environment_adapter import (
    UnixShellAdapter,
)


load_dotenv("/workspaces/SkyAgent/.env")

shell_adapter = UnixShellAdapter(
    base_dir="/",
    log_file_path="/workspaces/SkyAgent/examples/working_with_a_shell/unix_shell_usage.log",
)

tools = [
    AnthropicTool(tool_function=tool_function)
    for tool_function in shell_adapter.get_tool_functions()
]

system_prompt = """
You are an independent senior software engineer with access to a unix shell. 
    - Execute the task given to you. 
    - Avoid using complex chained commands, rather separate them into multiple smaller commands. 
    - Gathering context of your environment is your job. Look around the filesystem, read files whatever you need. 
    - Ensuring that your commands worked is your responsibility. Use additional commands to verify that your commands worked, and had their required effect.
"""


with shell_adapter:

    agent = AnthropicAgent(
        name="Unix Shell",
        model="claude-3-5-sonnet-latest",
        system_prompt=system_prompt,
        tools=tools,
        enable_live_display=True,
        max_turns=50,
    )

    result = agent.call_agent(
        query="""
    Your job is to create a bash script in the /workspaces/SkyAgent/examples/complex_sum directory that can calculate the sum of all numbers in its parameter list. 
    The script should output the result in a result.txt file. Test this script with a simple example, and ensure it works properly.
    Please give a short summary of your adventure executing this task.
    """
    )
