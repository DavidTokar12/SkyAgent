<div align="center">
    <picture>
      <img src="images/sky_agent_banner.png" alt="SkyAgent">
    </picture>
</div>

---

# 🤖🚀 SkyAgent: Open-Source LLM-Based Agent Framework

SkyAgent is an **open-source, developer-focused agent framework** built on Large Language Models (LLMs). Unlike "No Code" solutions, SkyAgent focuses on providing a solid foundation for key features while remaining highly extendable for any custom use case.

---

# ✨ Features

- 🔧 **Tool Use** – Just create any tool as a Python function, and SkyAgent takes care of the rest.
- 🧠 **Memory Management** – Store and recall relevant information across interactions, even if it doesn’t fit in the context window.
- 📚 **Knowledge Base Management** – Use any amount of structured, unstructured, binary or text data as a knowledge base for your agents.
- 🤝 **Multi-Agent Orchestration** – Create specialized agents, and SkyAgent will help you coordinate them to solve complex multi-step problems.
- 📜 **Logging** – Track agent behavior in real time effortlessly.
- 🛠 **Environment Interaction** – Give agents access to external systems, such as terminals, browsers, and codebases, that they can interact with.

---

# Why Choose SkyAgent?

- **Flexible & Extendable** – Customize and expand functionalities to suit your needs.
- **Developer-Focused** – Built by developers, for developers—no unnecessary abstractions.
- **Scalable & Modular** – Use only what you need, adapt as your project grows.

---

# Currently Supported LLM APIs

✅ **OpenAI (GPT)**  
✅ **Claude**

---

# Get Started

This simple example demonstrates how to create a basic agent using SkyAgent, equipped with a custom tool.

## 1. Import your API-specific Dependencies
```python
from skyagent.open_ai.open_ai_agent import OpenAIAgent
from skyagent.open_ai.open_ai_tool import OpenAiTool
```

## 2. Define a Tool
```python
def evaluate_expression(expression: str) -> float:
    """Use this tool whenever you need to evaluate a math expression.
    :param expression: A math expression in Python format.
    """
    return eval(expression)
```

## 3. Define a Structured Response
```python
class CalculationResponse(BaseModel):
    chain_of_thought: str = Field(description="The chain of thought that led to...")
    final_result: float
    your_favorite_number: str
```

## 4. Create an Agent
```python
agent = OpenAIAgent(
    name="Calculator",
    model="gpt-4o",
    system_prompt="...",
    tools=[OpenAiTool(evaluate_expression)],
    log_file_path="./basic_example.log",
    enable_live_display=True,
)
```

## 5. Call the Agent
```python
result = agent.call_agent(
    query="Solve this math problem: ...",
    response_format=CalculationResponse,
)
```

---

# 📟 Interacting with a Terminal

SkyAgent's **environment interactors** are tools that empower an agent to control external systems by executing actions and observing the system state afterward.

## Built-in Environment Interactors

SkyAgent includes common interactors by default, such as a **Unix shell adapter** that allows an agent to execute any command in a Unix-like environment.

⚠️ **WARNING:** SkyAgent does **not** restrict LLMs in any way. If the model decides to run a destructive command like `rm -rf ~`, it **will** be executed. 👉 **Ensure you set up a secure environment** (e.g., use a **Docker container**) to prevent unintended damage.

## Using the Unix Shell Adapter
```python
from skyagent.environment_interactors.unix_shell_environment_adapter import (
    UnixShellAdapter,
)

shell_adapter = UnixShellAdapter(
    base_dir="/",  # Root directory for the shell
    log_file_path="...",  # Path to store command logs
)

tools = [
    AnthropicTool(tool_function=tool_function)
    for tool_function in shell_adapter.get_tool_functions()
]

with shell_adapter:
    agent = AnthropicAgent(
        ...
        tools=tools,
    )
    result = agent.call_agent(query="...")
```

### Features of the Unix Shell Adapter
- ✔ **Execute shell commands**
- ✔ **Send control signals** (e.g., cancel long-running commands)
- ✔ **Handle interactive shell commands**

See `./examples/working_with_a_shell` for practical demonstrations.

## Creating Custom Environment Interactors
To build your own specialized environment interactor, extend the `EnvironmentAdapter` base class:
1. **Define your tools**: Implement methods that allow interaction with the environment.
2. **Return the updated environment state**: Ensure each tool provides a system state update.
3. **Override `get_tool_functions`**: This method should return all the defined tools.

---

# 📚 Example Features

## Parallel Tool Usage
SkyAgent supports running multiple tools in parallel, allowing for efficient multi-tasking. For example:
```python
def get_nth_fibonacci_number(n: int) -> int:
    ...

async def get_random_joke() -> dict:
    ...

fibonacci_tool = Tool(tool_function=get_nth_fibonacci_number, is_compute_heavy=True)
random_joke_tool = Tool(tool_function=get_random_joke)

agent = Agent(
    agent_name="Assistant",
    model="claude-3-5-sonnet-latest",
    api_provider="anthropic",
    tools=[fibonacci_tool, random_joke_tool],
    ...
)

result = agent.call_agent_sync(
    input_chat_history=[
        ModelInput(content="Give me 3 truly random jokes, and the 31st, 28th, and 19th Fibonacci numbers.")
    ]
)
```

## File, Directory, and Image Input
SkyAgent can process files, directories, and images as input attachments:
```python
# File input
from skyagent.input_loader.input_file_loader import InputFileLoader
file_attachment = InputFileLoader(...).load().to_attachment()
input_with_file = ModelInput(content="Explain this file...", attached_files=[file_attachment])

# Directory input
from skyagent.input_loader.input_directory_loader import InputDirectoryLoader
directory = InputDirectoryLoader(...).load()

# Image input
from skyagent.messages import ImageAttachment
input_with_image = ModelInput(content="Describe this image.", attached_images=[ImageAttachment.from_file_path("image.jpg")])
```

---

# TODO next
- MCP compatibility
- Gemini integration
- RAG integration

