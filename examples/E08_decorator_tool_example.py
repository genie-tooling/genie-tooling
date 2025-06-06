# examples/E08_decorator_tool_example.py
"""
Example: Using the @tool Decorator
----------------------------------
This example demonstrates defining a simple function, decorating it with
@genie_tooling.tool, registering it with Genie, and executing it.

To Run:
1. Ensure Genie Tooling is installed (`poetry install --all-extras`).
2. Run from the root of the project:
   `poetry run python examples/E08_decorator_tool_example.py`
"""
import asyncio
import logging
from typing import Optional

from genie_tooling import tool  # Import the decorator
from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie


# 1. Define your function and decorate it
@tool
async def greet_user(name: str, enthusiasm_level: int = 1) -> str:
    """
    Greets a user with a specified level of enthusiasm.

    Args:
        name (str): The name of the user to greet.
        enthusiasm_level (int): How enthusiastic the greeting should be (1-3).

    Returns:
        str: The generated greeting message.
    """
    if not 1 <= enthusiasm_level <= 3:
        return f"Sorry {name}, I can only be enthusiastic from level 1 to 3."

    greeting = f"Hello, {name}"
    greeting += "!" * enthusiasm_level
    return greeting

@tool
def simple_math_operation(a: float, b: float, operation: str = "add") -> float:
    """
    Performs a simple math operation.
    Args:
        a (float): First number.
        b (float): Second number.
        operation (str): 'add' or 'subtract'.
    Returns:
        float: The result of the operation.
    """
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    raise ValueError("Unknown operation for simple_math_operation")


async def run_decorator_tool_demo():
    print("--- @tool Decorator Example ---")

    app_config = MiddlewareConfig(
        features=FeatureSettings(
            llm="none",
            command_processor="none"
        ),
        tool_configurations={ # Enable the decorated tools by their function names
            "greet_user": {},
            "simple_math_operation": {}
        }
    )

    genie: Optional[Genie] = None
    try:
        print("\nInitializing Genie...")
        genie = await Genie.create(config=app_config)
        print("Genie initialized!")

        # 2. Register the decorated functions with Genie
        await genie.register_tool_functions([greet_user, simple_math_operation])
        print("Decorated functions registered as tools.")

        # 3. Execute the tools using their function names as identifiers
        print("\nExecuting 'greet_user' tool...")
        greeting_result = await genie.execute_tool(
            "greet_user",
            name="Alice",
            enthusiasm_level=3
        )
        print(f"Result from greet_user: {greeting_result}")

        print("\nExecuting 'simple_math_operation' tool...")
        math_result = await genie.execute_tool(
            "simple_math_operation",
            a=10.5,
            b=5.5,
            operation="subtract"
        )
        print(f"Result from simple_math_operation: {math_result}")


    except Exception as e:
        print(f"\nAn error occurred: {e}")
        logging.exception("Error details:")
    finally:
        if genie:
            await genie.close()
            print("\nGenie torn down.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_decorator_tool_demo())
