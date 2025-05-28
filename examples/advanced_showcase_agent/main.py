# examples/advanced_agent_showcase/main.py
"""
Example: Advanced Agent Showcase using Genie Facade
---------------------------------------------------
This example demonstrates configuring and using various components of the
Genie Tooling middleware through the Genie facade.

To Run:
1. Ensure Genie Tooling and all extras are installed (`poetry install --all-extras`).
2. Set necessary environment variables:
   - `OPENAI_API_KEY` (if using OpenAI LLM provider)
   - `GOOGLE_API_KEY` (if using Gemini LLM provider for RAG embedder or as LLM)
   - `OPENWEATHERMAP_API_KEY` (for the OpenWeatherMapTool)
3. Ensure you have Ollama running if you plan to use the OllamaLLMProviderPlugin.
   Pull a model, e.g., `ollama pull mistral`.
4. Run from the root of the project:
   `poetry run python examples/advanced_agent_showcase/main.py`
"""
import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.core.types import Plugin as CorePluginType
from genie_tooling.genie import Genie
from genie_tooling.llm_providers.types import ChatMessage # For type hinting
from genie_tooling.security.key_provider import KeyProvider

# --- 1. Custom KeyProvider ---
class DemoKeyProvider(KeyProvider, CorePluginType):
    plugin_id: str = "demo_showcase_key_provider_v1"
    description: str = "Provides API keys from environment variables for the advanced showcase."

    async def get_key(self, key_name: str) -> Optional[str]:
        print(f"[KeyProvider - Showcase] Requesting key: {key_name}")
        val = os.environ.get(key_name)
        if not val:
            print(f"[KeyProvider - Showcase] WARNING: Key '{key_name}' not found in environment.")
        return val

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        print(f"[{self.plugin_id}] Setup complete.")

    async def teardown(self) -> None:
        print(f"[{self.plugin_id}] Teardown complete.")

async def main():
    print("--- Advanced Agent Showcase ---")

    # --- Configuration ---
    key_provider_instance = DemoKeyProvider()

    # We need a CompactTextFormatter plugin to be available for tool lookup indexing
    # Assume plugin_id: "compact_text_formatter_plugin_v1", formatter_id: "llm_compact_text_v1"
    # This formatter should be discoverable (e.g., in src/genie_tooling/tools/formatters/impl/)

    app_config = MiddlewareConfig(
        default_log_level="DEBUG", # More verbose logging for the demo

        # LLM Provider: Default to Ollama, configure OpenAI and Gemini
        default_llm_provider_id="ollama_llm_provider_v1",
        llm_provider_configurations={
            "ollama_llm_provider_v1": {
                "model_name": "mistral:latest", # Make sure this model is pulled in Ollama
                "request_timeout_seconds": 180.0
            },
            "openai_llm_provider_v1": { # Assuming this plugin will exist (P1 task)
                "model_name": "gpt-3.5-turbo",
                "api_key_name": "OPENAI_API_KEY" # KeyProvider will fetch this
            },
            "gemini_llm_provider_v1": {
                "model_name": "gemini-1.5-flash-latest",
                "api_key_name": "GOOGLE_API_KEY"
            }
        },

        # Command Processor: Default to LLM-assisted, configure simple keyword as well
        default_command_processor_id="llm_assisted_tool_selection_processor_v1",
        command_processor_configurations={
            "simple_keyword_processor_v1": {
                "keyword_map": {
                    "calculate": "calculator_tool",
                    "math": "calculator_tool",
                    "sum": "calculator_tool",
                    "weather": "open_weather_map_tool",
                    "forecast": "open_weather_map_tool"
                },
                "keyword_priority": ["calculate", "weather", "math", "sum", "forecast"]
            },
            "llm_assisted_tool_selection_processor_v1": {
                # Use the default LLM (Ollama) for this processor by not specifying llm_provider_id
                "tool_formatter_id": "llm_compact_text_v1", # For LLM to understand tools
                "tool_lookup_top_k": 3 # Pre-filter tools before sending to LLM
            }
        },

        # Tool Lookup: Default to embedding similarity
        default_tool_lookup_provider_id="embedding_similarity_lookup_v1",
        # Provider specific config for tool_lookup_service if needed (none for embedding similarity by default)
        # default_tool_lookup_provider_configurations: {
        #     "embedding_similarity_lookup_v1": {"embedder_id": "my_custom_embedder_for_lookup"}
        # },
        default_tool_indexing_formatter_id="llm_compact_text_v1", # Crucial for tool lookup indexing


        # RAG Components: Configure defaults for simplified RAG calls
        default_rag_loader_id="web_page_loader_v1", # Default to web pages
        default_rag_splitter_id="character_recursive_text_splitter_v1",
        default_rag_embedder_id="sentence_transformer_embedder_v1", # Local for easy demo
        # default_rag_embedder_id="openai_embedding_generator_v1", # Or OpenAI if key is set and preferred
        default_rag_vector_store_id="faiss_vector_store_v1", # In-memory FAISS
        default_rag_retriever_id="basic_similarity_retriever_v1",

        # Tool Configurations (for specific tool plugins)
        tool_configurations={
            "open_weather_map_tool": {
                # No specific config needed if API_KEY_NAME is default
            },
            "generic_code_execution_tool": {
                # Configuration for the GenericCodeExecutionTool itself, if any.
                # Specific executors are discovered via PluginManager.
            }
        }
    )

    genie: Optional[Genie] = None
    try:
        print("\nInitializing Genie facade with showcase configuration...")
        genie = await Genie.create(
            config=app_config,
            key_provider_instance=key_provider_instance
        )
        print("Genie facade initialized.")

        # --- 1. LLM Interaction ---
        print("\n--- LLM Showcase ---")
        if os.getenv("OPENAI_API_KEY"):
            try:
                print("Trying OpenAI LLM for chat...")
                chat_messages: List[ChatMessage] = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the capital of France?"}
                ]
                openai_response = await genie.llm.chat(
                    messages=chat_messages,
                    provider_id="openai_llm_provider_v1" 
                )
                print(f"OpenAI Response: {openai_response['message']['content']}")
            except Exception as e:
                print(f"Error with OpenAI LLM: {e}")
        else:
            print("Skipping OpenAI LLM showcase (OPENAI_API_KEY not set).")

        try:
            print("\nTrying default LLM (Ollama/Mistral) for generation...")
            ollama_response = await genie.llm.generate(
                prompt="Explain the concept of Retrieval Augmented Generation in one sentence.",
                # No provider_id needed, uses default from config
                options={"temperature": 0.7} # Pass-through options
            )
            print(f"Ollama/Mistral Response: {ollama_response['text']}")
        except Exception as e:
            print(f"Error with Ollama LLM: {e}. Is Ollama running with 'mistral' model available?")


        # --- 2. RAG Showcase ---
        print("\n--- RAG Showcase ---")
        rag_collection = "showcase_rag_collection"
        # For this demo, we'll use a known public page.
        # Ensure your network allows access.
        # If you have issues, replace with a local file using index_directory and default_rag_loader_id="file_system_loader_v1"
        dummy_url_to_index = "https://www.python.org/about/gettingstarted/" # A relatively stable page
        try:
            print(f"Indexing web page: {dummy_url_to_index} into collection '{rag_collection}'...")
            # Uses defaults: web_page_loader, sentence_transformer_embedder, faiss_vector_store
            index_result = await genie.rag.index_web_page(dummy_url_to_index, collection_name=rag_collection)
            print(f"Web page indexing result: {json.dumps(index_result, indent=2)}")

            if index_result.get("status") == "success":
                search_query = "What are some Python libraries?"
                print(f"\nSearching RAG for: '{search_query}'")
                rag_results = await genie.rag.search(search_query, collection_name=rag_collection, top_k=1)
                if rag_results:
                    print("Top RAG Search Result:")
                    print(f"  Content: {rag_results[0].content[:300]}...")
                    print(f"  Score: {rag_results[0].score:.4f}")
                    print(f"  Source: {rag_results[0].metadata.get('url')}")
                else:
                    print("No RAG results found.")
            else:
                print("Skipping RAG search due to indexing failure.")
        except Exception as e:
            print(f"Error during RAG showcase: {e}")


        # --- 3. Command Processing Showcase ---
        print("\n--- Command Processing Showcase ---")
        commands_to_try = [
            "calculate 100 times 3.14",
            "what's the weather like in London?"
        ]
        for cmd_text in commands_to_try:
            print(f"\nProcessing command: '{cmd_text}' (using default LLM-assisted processor)")
            try:
                # This will use default_command_processor_id (llm_assisted_tool_selection_processor_v1)
                # which in turn uses the default_llm_provider_id (ollama_llm_provider_v1)
                # and default_tool_indexing_formatter_id (llm_compact_text_v1)
                cmd_result = await genie.run_command(command=cmd_text)
                print("Command Result:")
                print(json.dumps(cmd_result, indent=2, default=str)) # default=str for any non-serializable parts
            except Exception as e:
                print(f"Error processing command '{cmd_text}': {e}")

        print("\nProcessing command: 'sum 5 and 7' (using simple_keyword processor)")
        try:
            # Explicitly use the simple keyword processor
            keyword_cmd_result = await genie.run_command(
                command="sum 5 and 7",
                processor_id="simple_keyword_processor_v1"
            )
            print("Keyword Command Result:")
            print(json.dumps(keyword_cmd_result, indent=2, default=str))
        except Exception as e:
            print(f"Error processing keyword command: {e}")


        # --- 4. Direct Tool Invocation Showcase (if OpenWeatherMap API key is set) ---
        print("\n--- Direct Tool Invocation Showcase ---")
        if os.getenv("OPENWEATHERMAP_API_KEY"):
            try:
                print("Invoking OpenWeatherMapTool directly for 'New York'...")
                weather_result = await genie.execute_tool(
                    tool_identifier="open_weather_map_tool",
                    city="New York, US",
                    units="metric"
                )
                print("Direct Weather Tool Result:")
                print(json.dumps(weather_result, indent=2))
            except Exception as e:
                print(f"Error invoking weather tool directly: {e}")
        else:
            print("Skipping direct OpenWeatherMapTool invocation (OPENWEATHERMAP_API_KEY not set).")

        # --- 5. Tool Lookup Service (Conceptual) ---
        # In a full agent, the CommandProcessor (especially the LLM-assisted one)
        # would internally use the ToolLookupService. We configured it above.
        # Here's how you *could* access it if needed directly (though less common from app layer):
        print("\n--- Tool Lookup Service (Conceptual Usage) ---")
        if hasattr(genie, '_tool_lookup_service') and genie._tool_lookup_service: # type: ignore
            try:
                lookup_query = "find tools for math calculations"
                print(f"Looking up tools for query: '{lookup_query}'")
                # This uses default_tool_lookup_provider_id and default_tool_indexing_formatter_id
                # from MiddlewareConfig.
                ranked_tools = await genie._tool_lookup_service.find_tools(lookup_query, top_k=2) # type: ignore
                if ranked_tools:
                    print("Tool Lookup Results:")
                    for rt in ranked_tools:
                        print(f"  - ID: {rt.tool_identifier}, Score: {rt.score:.4f}, Snippet: {rt.description_snippet}")
                else:
                    print("No tools found by lookup service for this query.")
            except Exception as e:
                print(f"Error during conceptual tool lookup: {e}")
        else:
            print("ToolLookupService not directly accessible on Genie for this demo.")


    except Exception as e:
        print(f"\nAn UNEXPECTED CRITICAL error occurred in the showcase: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if genie:
            print("\n--- Tearing down Genie facade ---")
            await genie.close()
            print("Genie facade teardown complete.")

if __name__ == "__main__":
    asyncio.run(main())