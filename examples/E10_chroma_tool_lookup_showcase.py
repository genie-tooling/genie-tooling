# examples/chroma_tool_lookup_showcase.py/main.py
"""
Example: ChromaDB-backed Tool Lookup Showcase (Updated)
-------------------------------------------------------
This example demonstrates using ChromaDB for tool lookup via the
LLMAssistedCommandProcessor, configured using FeatureSettings.

To Run:
1. Ensure Genie Tooling is installed (`poetry install --all-extras`).
   You'll need dependencies for chromadb-client, sentence-transformers, and ollama.
2. Ensure Ollama is running and the model specified (e.g., 'mistral:latest') is pulled:
   `ollama serve`
   `ollama pull mistral`
3. Set OPENWEATHERMAP_API_KEY if you want the weather tool to execute successfully:
   `export OPENWEATHERMAP_API_KEY="your_key"`
4. Run from the root of the project:
   `poetry run python examples/E10_chroma_tool_lookup_showcase.py`
   (Note: The path in the original text dump was examples/chroma_tool_lookup_showcase.py/main.py,
    assuming it's examples/E10_chroma_tool_lookup_showcase.py based on other examples)

The demo will:
- Initialize Genie with an LLM-assisted command processor.
- Configure tool lookup to use embeddings stored in ChromaDB.
- Process commands, demonstrating tool selection and execution.
"""
import asyncio
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie

# Genie uses EnvironmentKeyProvider by default.

async def run_chroma_tool_lookup_showcase():
    print("--- ChromaDB-backed Tool Lookup Showcase (FeatureSettings) ---")

    current_file_dir = Path(__file__).parent
    chroma_tool_embeddings_path_str = str(current_file_dir / "chroma_db_tool_embeddings_data_v2")

    # Clean up previous run's data
    if Path(chroma_tool_embeddings_path_str).exists():
        print(f"Cleaning up existing ChromaDB tool embeddings data at: {chroma_tool_embeddings_path_str}")
        shutil.rmtree(chroma_tool_embeddings_path_str)
    # Path(chroma_tool_embeddings_path_str).mkdir(parents=True, exist_ok=True) # Chroma client creates it

    tool_embeddings_collection_name = "genie_tool_definitions_chroma_v2"

    # 1. Configure Middleware using FeatureSettings
    app_config = MiddlewareConfig(
        features=FeatureSettings(
            llm="ollama",
            llm_ollama_model_name="mistral:latest", # Ensure this model is pulled in Ollama

            command_processor="llm_assisted",
            command_processor_formatter_id_alias="compact_text_formatter", # For LLM prompt

            tool_lookup="embedding", # Use EmbeddingSimilarityLookupProvider
            tool_lookup_formatter_id_alias="compact_text_formatter", # For indexing tools
            tool_lookup_embedder_id_alias="st_embedder", # For tool description embeddings
            # The ST embedder will use its default model (e.g., "all-MiniLM-L6-v2")

            # Specific ChromaDB settings for tool lookup via FeatureSettings
            tool_lookup_chroma_path=chroma_tool_embeddings_path_str,
            tool_lookup_chroma_collection_name=tool_embeddings_collection_name,
        ),
        # Explicit override for the command processor to set tool_lookup_top_k
        command_processor_configurations={
            "llm_assisted_tool_selection_processor_v1": { # Canonical ID
                "tool_lookup_top_k": 3 # Enable tool lookup pre-filtering
            }
        },
        tool_configurations={
            # Enable tools that the LLM might select
            "calculator_tool": {},
            "open_weather_map_tool": {}
            # No specific config needed for these built-in tools
            # if API keys are handled by EnvironmentKeyProvider.
        }
    )

    genie: Optional[Genie] = None
    try:
        # 2. Initialize Genie
        print("\nInitializing Genie facade...")
        genie = await Genie.create(config=app_config) # Uses default EnvironmentKeyProvider
        print("Genie facade initialized for ChromaDB-backed tool lookup.")
        print(f"Tool embeddings will be indexed in ChromaDB at: {chroma_tool_embeddings_path_str}")
        print(f"Using collection: {tool_embeddings_collection_name}")

        # 3. Using Tool Lookup via Command Processor
        print("\n--- Using Tool Lookup via LLM-Assisted Command Processor ---")

        commands_to_test = [
            "What is the sum of 15 and 30?",
            "Tell me the current weather in Paris."
            # Add more commands to test different tools
        ]

        if not os.getenv("OPENWEATHERMAP_API_KEY") and any("weather" in cmd.lower() for cmd in commands_to_test):
            print("\nWARNING: OPENWEATHERMAP_API_KEY not set. Weather tool lookup might select the tool, but execution will likely fail.")
            print('You can set it with: export OPENWEATHERMAP_API_KEY="your_key"\n')

        for cmd_text in commands_to_test:
            print(f"\nProcessing command: '{cmd_text}'")
            try:
                command_result = await genie.run_command(command=cmd_text)
                if command_result:
                    print(f"  Thought Process: {command_result.get('thought_process')}")
                    if command_result.get("tool_result"):
                        print(f"  Tool Result: {json.dumps(command_result['tool_result'], indent=2)}")
                    elif command_result.get("error"):
                        print(f"  Command Error: {command_result['error']}")
                    elif command_result.get("message"): # For "No tool selected"
                        print(f"  Message: {command_result['message']}")
                    else:
                        print(f"  Raw Command Result: {json.dumps(command_result, indent=2)}")
                else:
                    print("  Command processor returned no result.")
            except Exception as e:
                print(f"  Error processing command '{cmd_text}': {e}")
                logging.exception("Command processing error details:")

        print(f"\n[Persistence Demo] Tool definition embeddings are stored in ChromaDB at '{chroma_tool_embeddings_path_str}'.")
        print("You can inspect this directory after the script finishes.")

    except Exception as e:
        print(f"\nAn unexpected error occurred in the showcase: {e}")
        logging.exception("Showcase error details:")
    finally:
        if genie:
            print("\nGenie facade tearing down...")
            await genie.close()
            print("Genie facade torn down.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # For more detailed Genie logs:
    # logging.getLogger("genie_tooling").setLevel(logging.DEBUG)
    asyncio.run(run_chroma_tool_lookup_showcase())
