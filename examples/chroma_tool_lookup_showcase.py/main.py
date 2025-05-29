# examples/chroma_tool_lookup_showcase.py
import asyncio
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.core.types import Plugin as CorePluginType
from genie_tooling.genie import Genie
from genie_tooling.security.key_provider import KeyProvider
# We might need to directly access ToolLookupService for a focused demo part
from genie_tooling.lookup.service import ToolLookupService

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
    # Separate ChromaDB path for tool embeddings to keep it distinct from RAG data
    chroma_tool_embeddings_path_str = str(current_file_dir / "chroma_db_tool_embeddings_data")
    
    if Path(chroma_tool_embeddings_path_str).exists():
        print(f"Cleaning up existing ChromaDB tool embeddings data at: {chroma_tool_embeddings_path_str}")
        shutil.rmtree(chroma_tool_embeddings_path_str)
    Path(chroma_tool_embeddings_path_str).mkdir(parents=True, exist_ok=True)

    key_provider = ShowcaseKeyProvider()

    # Define the collection name specifically for tool embeddings
    tool_embeddings_collection_name = "genie_tool_definitions_persistent_v1"

    app_config = MiddlewareConfig(
        # LLM Provider (needed by LLMAssistedToolSelectionProcessor)
        default_llm_provider_id="ollama_llm_provider_v1", # Assuming Ollama is running
        llm_provider_configurations={
            "ollama_llm_provider_v1": {"model_name": "mistral:latest"}
        },

        # Tool Lookup Configuration: Use EmbeddingSimilarityLookupProvider backed by ChromaDB
        default_tool_lookup_provider_id="embedding_similarity_lookup_v1",
        tool_lookup_provider_configurations={
            "embedding_similarity_lookup_v1": {
                "embedder_id": "sentence_transformer_embedder_v1", # Embedder for tool descriptions
                "vector_store_id": "chromadb_vector_store_v1",     # Vector store for tool embeddings
                "vector_store_config": {                           # Config for this VS instance
                    "path": chroma_tool_embeddings_path_str,       # Persistence path
                    "collection_name": tool_embeddings_collection_name # Crucial: specific collection
                },
                # The EmbeddingSimilarityLookupProvider's setup should use this collection_name
                # if it's passed within its own config or vector_store_config.
                # Let's assume it prioritizes collection_name from vector_store_config.
            }
        },
        # Default formatter for indexing tools (plugin ID)
        default_tool_indexing_formatter_id="compact_text_formatter_plugin_v1",

        # Command Processor that will use the ToolLookupService
        default_command_processor_id="llm_assisted_tool_selection_processor_v1",
        command_processor_configurations={
            "llm_assisted_tool_selection_processor_v1": {
                "tool_formatter_id": "compact_text_formatter_plugin_v1", # Formatter for LLM prompt
                "tool_lookup_top_k": 3 # Enable tool lookup
            }
        },
        
        # General configuration for ChromaDBVectorStore instances if not overridden
        # This might be used if ChromaDB is also the default_rag_vector_store_id
        vector_store_configurations={
            "chromadb_vector_store_v1": {
                "path": str(current_file_dir / "chroma_db_default_data"), # A default path
            }
        },
        embedding_generator_configurations={ # For the sentence_transformer_embedder_v1
            "sentence_transformer_embedder_v1": {"model_name": "all-MiniLM-L6-v2"}
        }
    )

    genie: Optional[Genie] = None
    try:
        genie = await Genie.create(config=app_config, key_provider_instance=key_provider)
        print("Genie facade initialized for ChromaDB-backed tool lookup.")

        # --- 1. Verify ToolLookupService is using ChromaDB (Conceptual Check) ---
        # The ToolLookupService itself doesn't expose its provider's backend directly.
        # We rely on the configuration being passed correctly.
        # The `EmbeddingSimilarityLookupProvider` will log if it uses a VectorStore.
        # For a direct check, we might need to access internal components if this were a test.
        # In an example, we demonstrate by effect.

        # The first time find_tools or reindex_tools_for_provider is called on ToolLookupService,
        # it will trigger index_tools on the EmbeddingSimilarityLookupProvider.
        # This provider, if configured with a vector_store_id, will then embed tool definitions
        # and add them to its configured ChromaDB collection.

        print(f"\nTool embeddings will be indexed in ChromaDB at: {chroma_tool_embeddings_path_str}")
        print(f"Using collection: {tool_embeddings_collection_name}")

        # --- 2. Using Tool Lookup via Command Processor ---
        # This will trigger tool indexing if it hasn't happened yet for the provider.
        # Built-in tools like 'calculator_tool' and 'open_weather_map_tool' should be indexed.
        print("\n--- Using Tool Lookup via LLM-Assisted Command Processor ---")
        
        commands_to_test = [
            "What is the sum of 15 and 30?",
            "Tell me the current weather in Paris."
        ]

        # Ensure OpenWeatherMap API key is set if testing that tool
        # export OPENWEATHERMAP_API_KEY="your_key"
        if not os.getenv("OPENWEATHERMAP_API_KEY") and any("weather" in cmd.lower() for cmd in commands_to_test):
            print("\nWARNING: OPENWEATHERMAP_API_KEY not set. Weather tool lookup might select the tool but execution will fail.")
            print("You can set it with: export OPENWEATHERMAP_API_KEY=\"your_key\"\n")


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
                    elif command_result.get("message"): # e.g., "No tool selected"
                        print(f"  Message: {command_result['message']}")
                else:
                    print("  Command processor returned no result.")
            except Exception as e:
                print(f"  Error processing command '{cmd_text}': {e}")
        
        # --- 3. Demonstrating Persistence of Tool Embeddings ---
        print(f"\n[Persistence Demo] Tool definition embeddings are stored in ChromaDB at '{chroma_tool_embeddings_path_str}'.")
        print("If you re-run this script (without cleaning that directory):")
        print("1. The ToolLookupService might not re-trigger a full re-embedding if its index is marked valid.")
        print("2. Or, if re-indexing occurs, ChromaDBVectorStore's `add` method would handle existing embeddings (typically ignoring duplicates by ID).")
        print("This demonstrates that the learned representation of tools can persist.")

        # --- 4. Forcing a Re-index (Conceptual - if needed for testing changes) ---
        # if genie._tool_lookup_service: # type: ignore
        # print("\n--- Forcing Re-index of Tools (for demonstration) ---")
        # genie._tool_lookup_service.invalidate_index(app_config.default_tool_lookup_provider_id) # type: ignore
        # print("Tool index invalidated. Next lookup will re-index.")
        # A subsequent call to run_command would trigger re-indexing.

    except Exception as e:
        print(f"\nAn unexpected error occurred in the showcase: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if genie:
            await genie.close()
            print("\nGenie facade torn down.")
        # For this demo, we might want to inspect the ChromaDB directory after runs.
        # print(f"ChromaDB tool embeddings data is at: {chroma_tool_embeddings_path_str}")
        # print("You can inspect it or delete it manually.")
        # For automated cleanup in a test script, you would uncomment:
        # if Path(chroma_tool_embeddings_path_str).exists():
        #     shutil.rmtree(chroma_tool_embeddings_path_str)

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    # Set specific loggers to DEBUG for more detail if needed:
    # logging.getLogger("genie_tooling.tool_lookup_providers.impl.embedding_similarity").setLevel(logging.DEBUG)
    # logging.getLogger("genie_tooling.vector_stores.impl.chromadb_store").setLevel(logging.DEBUG)
    # logging.getLogger("genie_tooling.lookup.service").setLevel(logging.DEBUG)
    # logging.getLogger("genie_tooling.command_processors.impl.llm_assisted_processor").setLevel(logging.DEBUG)

    asyncio.run(run_chroma_tool_lookup_showcase())