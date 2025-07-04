# examples/E18_human_in_loop_example.py
"""
Example: Using Human-in-the-Loop (HITL) (`genie.human_in_loop`)
----------------------------------------------------------------
This example demonstrates how to configure and use the Human-in-the-Loop
feature for approvals. It uses the `CliApprovalPlugin` by default,
which will prompt on the command line.

This example will primarily show HITL integrated with `genie.run_command`.

To Run:
1. Ensure Genie Tooling is installed (`poetry install --all-extras`).
2. Ensure Ollama is running and 'mistral:latest' is pulled for the LLM-assisted processor.
3. Run from the root of the project:
   `poetry run python examples/E18_human_in_loop_example.py`
   You will be prompted in the console to approve/deny tool execution.
"""
import asyncio
import json
import logging
from typing import Optional

from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie


async def run_hitl_demo():
    print("--- Human-in-the-Loop (HITL) Example ---")
    logging.basicConfig(level=logging.INFO)

    app_config = MiddlewareConfig(
        features=FeatureSettings(
            llm="ollama",
            llm_ollama_model_name="mistral:latest",
            command_processor="llm_assisted",
            tool_lookup="embedding",
            hitl_approver="cli_hitl_approver"
        ),
        tool_configurations={
            "calculator_tool": {}
        }
    )

    genie: Optional[Genie] = None
    try:
        print("\nInitializing Genie with HITL enabled...")
        genie = await Genie.create(config=app_config)
        print("Genie initialized!")

        print("\n--- `run_command` with HITL ---")
        command_text = "What is 15 multiplied by 7?"
        print(f"Sending command: '{command_text}'")
        print("The system will now select the 'calculator_tool' and then prompt for human approval before execution.")

        command_result = await genie.run_command(command_text)

        print("\nCommand Result after HITL:")
        print(json.dumps(command_result, indent=2, default=str))

        if command_result and command_result.get("tool_result"):
            print(f"\nTool Result: {command_result['tool_result']}")
        elif command_result and "hitl_decision" in command_result and command_result["hitl_decision"].get("status") != "approved":
            print(f"\nTool execution was {command_result['hitl_decision']['status']}. Reason: {command_result['hitl_decision'].get('reason')}")
        elif command_result and command_result.get("error"):
             print(f"\nCommand Error: {command_result['error']}")
        else:
             print(f"\nCommand did not result in a tool call or expected HITL flow: {command_result}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        logging.exception("HITL demo error details:")
    finally:
        if genie:
            await genie.close()
            print("\nGenie torn down.")

if __name__ == "__main__":
    asyncio.run(run_hitl_demo())
