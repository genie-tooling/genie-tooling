# examples/E07_run_command_llm_assisted_example.py
"""
Example: genie.run_command() with LLM-Assisted Processor
--------------------------------------------------------
This example demonstrates using genie.run_command() with the
'llm_assisted' command processor, using Ollama as the backing LLM.

To Run:
1. Ensure Genie Tooling is installed (`poetry install --all-extras`).
2. Ensure Ollama is running and 'mistral:latest' (or your chosen model) is pulled.
3. Run from the root of the project:
   `poetry run python examples/E07_run_command_llm_assisted_example.py`
"""
import asyncio
import json
import logging
from typing import Optional

from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie


async def run_llm_assisted_command_demo():
    print("--- LLM-Assisted Command Processor Demo (Ollama) ---")

    app_config = MiddlewareConfig(
        features=FeatureSettings(
            llm="ollama",
            llm_ollama_model_name="mistral:latest",
            command_processor="llm_assisted",
            # Use default compact_text_formatter for command_processor_formatter_id_alias
            # Use default embedding tool lookup with default ST embedder and compact_text_formatter
            tool_lookup="embedding",
        ),
        # Optional: Configure tool_lookup_top_k for the LLM-assisted processor
        command_processor_configurations={
            "llm_assisted_tool_selection_processor_v1": { # Canonical ID
                "tool_lookup_top_k": 3
            }
        },
        tool_configurations={
            # Ensure tools the LLM might select are enabled
            "calculator_tool": {}
        }
    )

    genie: Optional[Genie] = None
    try:
        print("\nInitializing Genie...")
        genie = await Genie.create(config=app_config)
        print("Genie initialized!")

        command_text = "What is 123 multiplied by 4?"
        print(f"\nRunning command: '{command_text}'")

        command_result = await genie.run_command(command_text)

        print("\nCommand Result:")
        if command_result:
            print(json.dumps(command_result, indent=2, default=str))
            if command_result.get("tool_result", {}).get("error_message"):
                 print(f"Tool Execution Error: {command_result['tool_result']['error_message']}")
            elif command_result.get("error"):
                 print(f"Command Processing Error: {command_result['error']}")
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
    # logging.getLogger("genie_tooling").setLevel(logging.DEBUG)
    asyncio.run(run_llm_assisted_command_demo())
