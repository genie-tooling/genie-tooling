# examples/simple_agent_cli/main.py
"""
Example: Simple Agent CLI (Refactored for Genie Facade)
-------------------------------------------------------
This example demonstrates a basic command-line agent that uses the Genie
facade to interact with tools, configured via FeatureSettings.

To Run:
1. Ensure Genie Tooling is installed (`poetry install --all-extras`).
2. Set an environment variable for OpenWeatherMap API key if you want to test that tool:
   `export OPENWEATHERMAP_API_KEY="your_actual_key"`
3. Run from the root of the project:
   `poetry run python examples/simple_agent_cli/main.py`

The agent will:
- Initialize Genie with a simple keyword command processor.
- Ask for your input (e.g., "calculate 10 + 5", "weather in London").
- Use genie.run_command() to process the input and execute the tool.
- Display the result.
"""
import asyncio
import json
import logging
import os
from typing import Optional

from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie

# Genie uses EnvironmentKeyProvider by default if no key_provider_instance is given
# and key_provider_id is not set or set to "env_keys" in MiddlewareConfig.

async def run_simple_agent_cli():
    print("--- Simple Agent CLI (Genie Facade Version) ---")

    # 1. Configure Middleware using FeatureSettings
    app_config = MiddlewareConfig(
        features=FeatureSettings(
            llm="none", # Using simple_keyword, no LLM needed for command processing itself
            command_processor="simple_keyword",
            # No RAG or complex tool lookup needed for this basic CLI
            rag_embedder="none",
            tool_lookup="none",
        ),
        # Configure the simple_keyword_processor
        command_processor_configurations={
            "simple_keyword_processor_v1": { # Canonical ID
                "keyword_map": {
                    "calculate": "calculator_tool",
                    "math": "calculator_tool",
                    "add": "calculator_tool",
                    "plus": "calculator_tool",
                    "weather": "open_weather_map_tool",
                    "forecast": "open_weather_map_tool",
                },
                "keyword_priority": ["calculate", "math", "add", "plus", "weather", "forecast"]
            }
        },
        tool_configurations={
            # Explicitly enable the tools to be used
            "calculator_tool": {},
            "open_weather_map_tool": {}
            # No specific config needed for these tools if API keys are handled
            # by the default EnvironmentKeyProvider.
        }
    )

    # 2. Instantiate Genie
    genie: Optional[Genie] = None
    try:
        print("\nInitializing Genie facade...")
        # No key_provider_instance needed if using EnvironmentKeyProvider (default)
        genie = await Genie.create(config=app_config)
        print("Genie facade initialized successfully!")

        # 3. CLI Loop
        print("\nType 'quit' to exit.")
        if not os.getenv("OPENWEATHERMAP_API_KEY"):
            print("Note: OPENWEATHERMAP_API_KEY not set. 'weather' commands will select the tool but execution will fail.")

        while True:
            try:
                user_query = await asyncio.to_thread(input, "\n> Your query: ")
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            if user_query.lower() == "quit":
                print("Exiting...")
                break
            if not user_query.strip():
                continue

            print(f"Processing command: '{user_query}'")
            try:
                # genie.run_command will use the configured 'simple_keyword_processor_v1'
                # which will prompt for parameters if a tool is matched.
                command_result = await genie.run_command(user_query)

                print("\nCommand Result:")
                if command_result:
                    print(json.dumps(command_result, indent=2, default=str))
                    if command_result.get("tool_result") and command_result["tool_result"].get("error_message"):
                        print(f"Tool Execution Error: {command_result['tool_result']['error_message']}")
                    elif command_result.get("error"):
                         print(f"Command Processing Error: {command_result['error']}")
                else:
                    print("Command did not produce a result.")

            except Exception as e:
                print(f"An error occurred while processing the command: {e}")
                logging.exception("Error details:")

    except Exception as e:
        print(f"Failed to initialize or run agent: {e}")
        logging.exception("Initialization/runtime error details:")
    finally:
        if genie:
            print("\nTearing down Genie facade...")
            await genie.close()
            print("Genie facade torn down.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # For more detailed library logs:
    # logging.getLogger("genie_tooling").setLevel(logging.DEBUG)
    asyncio.run(run_simple_agent_cli())
