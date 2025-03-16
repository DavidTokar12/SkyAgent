# 🤖🚀 SkyAgent: Open-Source LLM-Based Agent Framework  

SkyAgent is an **open-source, developer-focused agent framework** built on Large Language Models (LLMs).  

Unlike "No Code" solutions, **SkyAgent focuses on providing a solid foundation** for key features while remaining **highly extendable** for any custom use case.  

## ✨ Features  

🔧 **Tool Use** – Just create any tool as a Python function, and SkyAgent takes care of the rest.  
🧠 **Memory Management** – Store and recall relevant information across interactions, even if it doesn’t fit in the context window.  
📚 **Knowledge Base Management** – Use **any** amount of structured, unstructured, binary or text data as a knowledge base for your agents.  
🤝 **Multi-Agent Orchestration** – Create specialized agents, and SkyAgent will help you coordinate them to solve complex multi-step problems.  
📜 **Logging** – Track agent behavior in real time effortlessly.  
🛠 **Environment Interaction** – Give agents access to external systems, such as terminals, browsers, and codebases, that they can interact with.

## 🔧 Why Choose SkyAgent?  

- **Flexible & Extendable** – Customize and expand functionalities to suit your needs.  
- **Developer-Focused** – Built **by developers, for developers**—no unnecessary abstractions.  
- **Scalable & Modular** – Use only what you need, adapt as your project grows.  

## 🔮 Currently Supported LLM APIs  

✅ **OpenAI**  
✅ **Claude**  

### 🎯 Planned to Support next

🚧 **Gemini** (Coming soon)  
🚧 **Ollama** (Coming soon)

## 🚀 Get Started

This simple example demonstrates how to create a **basic agent** using `SkyAgent`, equipped with a **custom tool**.

---

### Step 1: Import your API-specific Dependencies

```python
from skyagent.open_ai.open_ai_agent import OpenAIAgent
from skyagent.open_ai.open_ai_tool import OpenAiTool
```

---

### 🛠 Step 2: Define a Tool

⚠ **WARNING:** Docstring for tool functions are part of your code! Make sure you clearly explain what the tool does and what its parameters are. Ideally use the 'use this tool to...' format, as LLMs prefer this structure.

```python
def evaluate_expression(expression: str) -> float:
    """Use this tool whenever you need to evaluate a math expression.

    :param expression: A math expression in Python format.
    """
    return eval(expression)
```

---

### Step 3: Define a Structured Response

SkyAgent implements structured outputs for all LLM APIs, even for ones that don't support it out of the box. Adding the description to the output fields serves the same purpose as the docstring of your tools.

```python
class CalculationResponse(BaseModel):
    chain_of_thought: str = Field(description="The chain of thought that led to...")
    final_result: float
    your_favorite_number: str
```

---

### 🤖 Step 4: Create an Agent

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

---

### 🎯 Step 5: Call the Agent

```python
result = agent.call_agent(
    query="Solve this math problem: ...",
    response_format=CalculationResponse,
)
```

If you enabled live display, you will see the following live view in your terminal of what your agent is doing:

![image info](/workspaces/SkyAgent/images/terminal_output.png)


---

# 📟 Interacting with a Terminal  

SkyAgent's **environment interactors** are tools that empower an agent to control external systems by executing actions and observing the system state afterward.

## Built-in Environment Interactors  

SkyAgent includes common interactors by default, such as a **Unix shell adapter** that allows an agent to execute any command in a Unix-like environment.  

⚠ **WARNING:** SkyAgent does **not** restrict LLMs in any way. If the model decides to run a destructive command like `rm -rf ~`, it **will** be executed.  
👉 **Ensure you set up a secure environment** (e.g., use a **Docker container**) to prevent unintended damage.  

---

## Using the Unix Shell Adapter  

To integrate a Unix shell interactor into your SkyAgent setup, use the `UnixShellAdapter`:  

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

# context manager is used to spawn/destroy a shell
with shell_adapter:
    agent = AnthropicAgent(
        ...
        tools=tools,
    )
    result = agent.call_agent(query="...")
```

---

## Features of the Unix Shell Adapter  

✔ **Execute shell commands**  
✔ **Send control signals** (e.g., cancel long-running commands)  
✔ **Handle interactive shell commands**  

📌 **Example usage:** Check out `./examples/working_with_a_shell` for practical demonstrations.

---

## 🎛 Creating Custom Environment Interactors  

To build your own specialized **environment interactor**, extend the `EnvironmentAdapter` base class:  

1. **Define your tools**: Implement methods that allow interaction with the environment.  
2. **Return the updated environment state**: Ensure each tool provides a system state update.  
3. **Override `get_tool_functions`**: This method should return all the defined tools.

