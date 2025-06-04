# examples/E06_run_command_simple_keyword_example.py
"""
Example: genie.run_command() with Simple Keyword Processor
---------------------------------------------------------
This example demonstrates using genie.run_command() with the
'simple_keyword' command processor.

To Run:
1. Ensure Genie Tooling is installed (`poetry install --all-extras`).
2. Run from the root of the project:
   `poetry run python examples/E06_run_command_simple_keyword_example.py`
"""
import asyncio
import json
import logging
from typing import Optional

from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie


async def run_simple_keyword_command_demo():
    print("--- Simple Keyword Command Processor Demo ---")

    app_config = MiddlewareConfig(
        features=FeatureSettings(
            llm="none", # No LLM needed for this processor
            command_processor="simple_keyword"
        ),
        command_processor_configurations={
            "simple_keyword_processor_v1": { # Canonical ID
                "keyword_map": {
                    "calculate": "calculator_tool",
                    "sum": "calculator_tool",
                    "add": "calculator_tool",
                    "plus": "calculator_tool",
                },
                "keyword_priority": ["calculate", "sum", "add", "plus"]
            }
        },
        tool_configurations={
            "calculator_tool": {} # Enable the calculator tool
        }
    )

    genie: Optional[Genie] = None
    try:
        print("\nInitializing Genie...")
        genie = await Genie.create(config=app_config)
        print("Genie initialized!")

        command_text = "calculate 25 plus 75"
        print(f"\nRunning command: '{command_text}'")
        print("The SimpleKeywordToolSelectorProcessorPlugin will prompt for parameters if a tool is matched.")

        # For non-interactive, you'd typically use execute_tool or an LLM-assisted processor.
        # This demo shows the prompting behavior.
        command_result = await genie.run_command(command_text)

        print("\nCommand Result:")
        if command_result:
            print(json.dumps(command_result, indent=2))
        else:
            print("Command did not produce a result.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        logging.exception("Error details:")
    finally:
        if genie:
            await genie.close()
            print("\nGenie torn down.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_simple_keyword_command_demo())
