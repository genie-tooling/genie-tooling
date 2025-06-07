# examples/E11_advanced_showcase_agent.py
"""
Example: Advanced Agent Showcase using Genie Facade
---------------------------------------------------
This example demonstrates configuring and using various components of the
Genie Tooling middleware through the Genie facade.
"""
import asyncio
import json
import logging
import os
from typing import List, Optional

from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.core.types import Plugin as CorePluginType
from genie_tooling.genie import Genie
from genie_tooling.llm_providers.types import ChatMessage
from genie_tooling.security.key_provider import KeyProvider

logger = logging.getLogger(__name__)

# --- 1. Custom KeyProvider ---
class DemoKeyProvider(KeyProvider, CorePluginType):
    plugin_id: str = "demo_showcase_key_provider_v1"
    async def get_key(self, key_name: str) -> Optional[str]:
        val = os.environ.get(key_name)
        if not val:
            print(f"[KeyProvider - Showcase] WARNING: Key '{key_name}' not found.")
        return val
    async def setup(self, config=None): print(f"[{self.plugin_id}] Setup.")
    async def teardown(self): print(f"[{self.plugin_id}] Teardown.")

async def main():
    print("--- Advanced Agent Showcase ---")
    key_provider_instance = DemoKeyProvider()

    app_config = MiddlewareConfig(
        features=FeatureSettings(
            llm="ollama", # Default LLM
            llm_ollama_model_name="mistral:latest",
            llm_openai_model_name="gpt-3.5-turbo", # For OpenAI provider if used
            llm_gemini_model_name="gemini-1.5-flash-latest", # For Gemini provider if used

            command_processor="llm_assisted", # Default command processor
            command_processor_formatter_id_alias="compact_text_formatter", # For LLM-assisted

            tool_lookup="embedding", # Default tool lookup
            tool_lookup_formatter_id_alias="compact_text_formatter", # For indexing
            tool_lookup_embedder_id_alias="st_embedder", # For tool lookup embeddings

            rag_loader="web_page", # Default RAG loader
            rag_embedder="sentence_transformer", # Default RAG embedder
            rag_vector_store="faiss", # Default RAG vector store
        ),
        default_log_level="INFO",

        llm_provider_configurations={
            "ollama": { "request_timeout_seconds": 180.0 },
            "openai": { "model_name": "gpt-4-turbo-preview" },
            "gemini": {}
        },
        command_processor_configurations={
            "simple_keyword_cmd_proc": { # Alias for simple_keyword_processor_v1
                "keyword_map": {
                    "calculate": "calculator_tool", "math": "calculator_tool",
                    "weather": "open_weather_map_tool", "forecast": "open_weather_map_tool"
                }
            },
            "llm_assisted_cmd_proc": { # Alias for llm_assisted_tool_selection_processor_v1
                "tool_lookup_top_k": 3
            }
        },
        tool_configurations={ # Tools must be enabled
            "calculator_tool": {},
            "open_weather_map_tool": {},
            "generic_code_execution_tool": {}
        }
    )

    genie: Optional[Genie] = None
    try:
        print("\nInitializing Genie facade with showcase configuration...")
        genie = await Genie.create(config=app_config, key_provider_instance=key_provider_instance)
        print("Genie facade initialized.")

        # --- 1. LLM Interaction ---
        print("\n--- LLM Showcase ---")
        if os.getenv("OPENAI_API_KEY"):
            try:
                print("Trying OpenAI LLM (gpt-4-turbo-preview) for chat...")
                chat_messages: List[ChatMessage] = [{"role": "user", "content": "What is the capital of France?"}]
                openai_response = await genie.llm.chat(messages=chat_messages, provider_id="openai")
                print(f"OpenAI Response: {openai_response['message']['content']}")
            except Exception as e:
                print(f"Error with OpenAI LLM: {e}")
        else:
            print("Skipping OpenAI LLM showcase (OPENAI_API_KEY not set).")

        try:
            print("\nTrying default LLM (Ollama/Mistral) for generation...")
            ollama_response = await genie.llm.generate("Explain RAG in one sentence.", options={"temperature": 0.7})
            print(f"Ollama/Mistral Response: {ollama_response['text']}")
        except Exception as e:
            print(f"Error with Ollama LLM: {e}")

        # --- 2. RAG Showcase ---
        print("\n--- RAG Showcase ---")
        rag_collection = "showcase_rag_collection_e11"
        dummy_url = "https://www.python.org/about/gettingstarted/"
        try:
            print(f"Indexing web page: {dummy_url} into collection '{rag_collection}'...")
            index_result = await genie.rag.index_web_page(dummy_url, collection_name=rag_collection)
            print(f"Web page indexing result: {json.dumps(index_result, indent=2)}")
            if index_result.get("status") == "success":
                rag_results = await genie.rag.search("What are Python libraries?", collection_name=rag_collection, top_k=1)
                if rag_results:
                    print(f"Top RAG: {rag_results[0].content[:300]}... (Source: {rag_results[0].metadata.get('url')})")
        except Exception as e:
            print(f"Error during RAG showcase: {e}")

        # --- 3. Command Processing Showcase ---
        print("\n--- Command Processing Showcase ---")
        commands = ["calculate 100 times 3.14", "what's the weather like in London?"]
        for cmd_text in commands:
            print(f"\nProcessing command: '{cmd_text}' (using default LLM-assisted processor)")
            try:
                cmd_result = await genie.run_command(command=cmd_text)
                print(f"Result: {json.dumps(cmd_result, indent=2, default=str)}")
            except Exception as e:
                print(f"Error: {e}")

        print("\nProcessing command: 'sum 5 and 7' (using simple_keyword processor)")
        try:
            keyword_cmd_result = await genie.run_command("sum 5 and 7", processor_id="simple_keyword_cmd_proc")
            print(f"Keyword Result: {json.dumps(keyword_cmd_result, indent=2, default=str)}")
        except Exception as e:
            print(f"Error: {e}")

        # --- 4. Direct Tool Invocation ---
        print("\n--- Direct Tool Invocation Showcase ---")
        if os.getenv("OPENWEATHERMAP_API_KEY"):
            try:
                weather_result = await genie.execute_tool("open_weather_map_tool", city="New York, US", units="metric")
                print(f"Direct Weather: {json.dumps(weather_result, indent=2)}")
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Skipping direct OpenWeatherMapTool (OPENWEATHERMAP_API_KEY not set).")

    except Exception as e:
        print(f"\nUNEXPECTED CRITICAL error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if genie:
            await genie.close()
            print("\nGenie facade torn down.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # logging.getLogger("genie_tooling").setLevel(logging.DEBUG)
    asyncio.run(main())
