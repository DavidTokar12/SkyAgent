from skyagent.agent_tool import AgentTool


async def func(a: int, b: str, c: float) -> str:
    """A test function.
    This is a long description.

    Args:
        a: An integer.
        b: A string.
        c: A float.
    """
    return "Test string!"


tool = AgentTool(func=func)
print(tool.is_async)
