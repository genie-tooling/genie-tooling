# examples/chroma_tool_lookup_showcase.py
import asyncio
import os
import shutil
from pathlib import Path
from typing import Optional

from genie_tooling.config.features import FeatureSettings  # Import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.core.types import Plugin as CorePluginType
from genie_tooling.genie import Genie
from genie_tooling.security.key_provider import KeyProvider

# We might need to directly access ToolLookupService for a focused demo part
# from genie_tooling.lookup.service import ToolLookupService # Not used directly in this version

# --- 1. Application's KeyProvider ---
class ShowcaseKeyProvider(KeyProvider, CorePluginType):
    plugin_id = "showcase_chroma_tool_lookup_key_provider_v1"
    async def get_key(self, key_name: str) -> str | None:
        return os.environ.get(key_name)
    async def setup(self, config=None): pass
    async def teardown(self): pass

async def run_chroma_tool_lookup_showcase():
    print("--- ChromaDB-backed Tool Lookup Showcase ---")

    # --- Configuration ---
    current_file_dir = Path(__file__).parent
    # Separate ChromaDB path for tool embeddings
    chroma_tool_embeddings_path_str = str(current_file_dir / "chroma_db_tool_embeddings_data")

    if Path(chroma_tool_embeddings_path_str).exists():
        print(f"Cleaning up existing ChromaDB tool embeddings data at: {chroma_tool_embeddings_path_str}")
        shutil.rmtree(chroma_tool_embeddings_path_str)
    Path(chroma_tool_embeddings_path_str).mkdir(parents=True, exist_ok=True)

    key_provider = ShowcaseKeyProvider()
    tool_embeddings_collection_name = "genie_tool_definitions_persistent_v1"

    app_config = MiddlewareConfig(
        features=FeatureSettings(
            llm="ollama", # Needed by LLMAssistedToolSelectionProcessor
            llm_ollama_model_name="mistral:latest", # Ensure Ollama is running with this model

            tool_lookup="embedding", # Use EmbeddingSimilarityLookupProvider
            tool_lookup_formatter_id_alias="compact_text_formatter", # For indexing tools
            tool_lookup_embedder_id_alias="st_embedder", # For tool description embeddings
            # The ST embedder will use its default model (e.g., "all-MiniLM-L6-v2")
            # If a different ST model is needed for tool lookup, configure it explicitly in
            # embedding_generator_configurations for "sentence_transformer_embedder_v1".

            # Specific ChromaDB settings for tool lookup
            tool_lookup_chroma_path=chroma_tool_embeddings_path_str,
            tool_lookup_chroma_collection_name=tool_embeddings_collection_name,

            command_processor="llm_assisted", # Command Processor that will use the ToolLookupService
            command_processor_formatter_id_alias="compact_text_formatter" # For LLM prompt
        ),
        # Explicit override for the command processor to set tool_lookup_top_k
        command_processor_configurations={
            "llm_assisted_tool_selection_processor_v1": { # Canonical ID
                "tool_lookup_top_k": 3 # Enable tool lookup pre-filtering
            }
        },
        # If you needed to specify a non-default model for the SentenceTransformerEmbedder
        # used by tool_lookup, you would add it here:
        # embedding_generator_configurations={
        #     "sentence_transformer_embedder_v1": {"model_name": "paraphrase-MiniLM-L3-v2"}
        # }
    )

    genie: Optional[Genie] = None
    try:
        genie = await Genie.create(config=app_config, key_provider_instance=key_provider)
        print("Genie facade initialized for ChromaDB-backed tool lookup.")

        print(f"\nTool embeddings will be indexed in ChromaDB at: {chroma_tool_embeddings_path_str}")
        print(f"Using collection: {tool_embeddings_collection_name}")

        # --- 2. Using Tool Lookup via Command Processor ---
        print("\n--- Using Tool Lookup via LLM-Assisted Command Processor ---")

        commands_to_test = [
            "What is the sum of 15 and 30?",
            "Tell me the current weather in Paris."
        ]

        if not os.getenv("OPENWEATHERMAP_API_KEY") and any("weather" in cmd.lower() for cmd in commands_to_test):
            print("\nWARNING: OPENWEATHERMAP_API_KEY not set. Weather tool lookup might select the tool but execution will fail.")
            print('You can set it with: export OPENWEATHERMAP_API_KEY="your_key"\n')

        for cmd_text in commands_to_test:
            print(f"\nProcessing command: '{cmd_text}'")
            try:
                command_result = await genie.run_command(command=cmd_text)
                if command_result:
                    print(f"  Thought Process: {command_result.get('thought_process')}")
                    if command_result.get("tool_result"):
                        print(f"  Tool Result: {command_result['tool_result']}")
                    elif command_result.get("error"):
                        print(f"  Command Error: {command_result['error']}")
                    elif command_result.get("message"):
                        print(f"  Message: {command_result['message']}")
                else:
                    print("  Command processor returned no result.")
            except Exception as e:
                print(f"  Error processing command '{cmd_text}': {e}")

        print(f"\n[Persistence Demo] Tool definition embeddings are stored in ChromaDB at '{chroma_tool_embeddings_path_str}'.")

    except Exception as e:
        print(f"\nAn unexpected error occurred in the showcase: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if genie:
            await genie.close()
            print("\nGenie facade torn down.")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    # logging.getLogger("genie_tooling").setLevel(logging.DEBUG)
    asyncio.run(run_chroma_tool_lookup_showcase())
