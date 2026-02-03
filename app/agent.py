import os
from collections.abc import AsyncIterable
from typing import Any, Literal

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

memory = MemorySaver()


@tool
def calculator(operation: str, a: float, b: float) -> str:
    """Perform basic math operations: add, subtract, multiply, or divide.

    Args:
        operation: One of 'add', 'subtract', 'multiply', 'divide'
        a: First number
        b: Second number

    Returns:
        The result of the operation
    """
    if operation == "add":
        return str(a + b)
    elif operation == "subtract":
        return str(a - b)
    elif operation == "multiply":
        return str(a * b)
    elif operation == "divide":
        if b == 0:
            return "Error: Division by zero"
        return str(a / b)
    else:
        return "Error: Unknown operation"


@tool
def get_greeting(name: str = "friend") -> str:
    """Generate a friendly greeting for someone.

    Args:
        name: The name of the person to greet

    Returns:
        A friendly greeting message
    """
    return f"Hello {name}! Nice to meet you! 👋"


class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str


class CalculatorGreetingAgent:
    """CalculatorGreetingAgent - a specialized assistant for greetings and calculations."""

    SYSTEM_INSTRUCTION = (
        "You are a helpful assistant that can greet people and help with math calculations. "
        "Use the calculator tool for math problems (add, subtract, multiply, divide) and the get_greeting tool to greet users. "
        "Be friendly and concise in your responses. "
        "If a user asks about something other than greetings or calculations, "
        "politely state that you can only help with greetings and basic arithmetic operations."
    )

    def __init__(self):
        self.model = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
        )
        self.tools = [calculator, get_greeting]

        # Create ReAct agent without response_format for better compatibility
        self.graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
        )

    async def stream(self, query, context_id) -> AsyncIterable[dict[str, Any]]:
        inputs = {"messages": [("user", query)]}
        config = {"configurable": {"thread_id": context_id}}

        for item in self.graph.stream(inputs, config, stream_mode="values"):
            message = item["messages"][-1]
            if (
                isinstance(message, AIMessage)
                and message.tool_calls
                and len(message.tool_calls) > 0
            ):
                # Agent is calling a tool
                tool_name = message.tool_calls[0].get("name", "tool")
                if tool_name == "calculator":
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": "Performing calculation...",
                    }
                elif tool_name == "get_greeting":
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": "Preparing greeting...",
                    }
            elif isinstance(message, ToolMessage):
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Processing result...",
                }

        yield self.get_agent_response(config)

    def get_agent_response(self, config):
        current_state = self.graph.get_state(config)

        # Get the last AI message content
        messages = current_state.values.get("messages", [])
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, AIMessage) and last_message.content:
                content = last_message.content
                # Check if it's a string (it should be)
                if isinstance(content, str):
                    content_lower = content.lower()
                    # Check if the message indicates more input is needed
                    if any(
                        keyword in content_lower
                        for keyword in [
                            "need more",
                            "please provide",
                            "please specify",
                            "which",
                            "clarify",
                        ]
                    ):
                        return {
                            "is_task_complete": False,
                            "require_user_input": True,
                            "content": content,
                        }
                    else:
                        # Task completed successfully
                        return {
                            "is_task_complete": True,
                            "require_user_input": False,
                            "content": content,
                        }

        return {
            "is_task_complete": False,
            "require_user_input": True,
            "content": (
                "We are unable to process your request at the moment. Please try again."
            ),
        }

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]
