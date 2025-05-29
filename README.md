# Genie Tooling

[![Pytest Status](https://github.com/genie-tooling/genie-tooling/actions/workflows/python_ci.yml/badge.svg)](https://github.com/genie-tooling/genie-tooling/actions/workflows/python_ci.yml)
<!-- TODO: Replace with actual CI badge URL once repo is public -->

A hyper-pluggable Python middleware for building sophisticated Agentic AI and LLM-powered applications.

## Vision

Genie Tooling empowers developers to construct complex AI agents by providing a modular, async-first framework. It emphasizes clear interfaces and interchangeable components, allowing for rapid iteration and customization of agent capabilities. The `Genie` facade offers a simplified, high-level API for common agentic tasks.

## Core Concepts

*   **Plugins**: Genie is built around a plugin architecture. Almost every piece of functionality (LLM interaction, tool definition, data retrieval, caching, etc.) is a plugin that can be swapped or extended.
*   **Managers**: Specialized managers (e.g., `ToolManager`, `RAGManager`, `LLMProviderManager`) orchestrate their respective plugin types.
*   **`Genie` Facade**: The primary entry point for most applications. It simplifies interaction with all underlying managers and plugins, providing easy access to:
    *   LLM chat and generation (`genie.llm`)
    *   Retrieval Augmented Generation (`genie.rag`)
    *   Tool execution (`genie.execute_tool`)
    *   Natural language command processing (`genie.run_command`)
*   **Configuration**: Applications provide runtime configuration (e.g., API keys, default plugin choices, plugin-specific settings) via a `MiddlewareConfig` object and a custom `KeyProvider` implementation.

## Key Plugin Categories

Genie Tooling supports a wide array of plugin types:

*   **LLM Providers**: Interface with LLM APIs (e.g., OpenAI, Ollama, Gemini).
*   **Command Processors**: Interpret user commands to select tools and extract parameters.
*   **Tools**: Define discrete actions the agent can perform (e.g., calculator, web search, file operations).
*   **Key Provider**: Securely supplies API keys (implemented by the application).
*   **RAG Components**:
    *   **Document Loaders**: Load data from various sources (files, web pages).
    *   **Text Splitters**: Divide documents into manageable chunks.
    *   **Embedding Generators**: Create vector embeddings for text.
    *   **Vector Stores**: Store and search embeddings (e.g., FAISS, ChromaDB).
    *   **Retrievers**: Orchestrate the RAG retrieval process.
*   **Tool Lookup Providers**: Help find relevant tools based on natural language.
*   **Invocation Strategies**: Define the lifecycle of a tool call (validation, execution, caching, error handling).
*   **Caching Providers**: Offer caching mechanisms (e.g., in-memory, Redis).
*   **Code Executors**: Securely execute code (e.g., for a generic code execution tool).
*   **Logging & Redaction**: Adapt logging and sanitize sensitive data.
*   **Definition Formatters**: Format tool metadata for different uses (e.g., LLM prompts, UI).

*(Refer to `pyproject.toml` for a list of built-in plugin entry points and their default identifiers).*

## Installation

1.  **Clone the repository:**
    ```bash
    # TODO: Update with the correct repository URL when available
    git clone https://github.com/YourNameOrOrg/genie-tooling.git
    cd genie-tooling
    ```

2.  **Install dependencies using Poetry:**
    (Ensure [Poetry](https://python-poetry.org/docs/#installation) is installed.)
    ```bash
    poetry install --all-extras # Install with all optional dependencies
    ```
    To install only core dependencies:
    ```bash
    poetry install
    ```
    You can install specific extras like `poetry install --extras ollama openai` as needed.

## Quick Start with the `Genie` Facade

The `Genie` facade is the recommended way to get started. A quick example flexing many parts:

```python
import asyncio
import os
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.core.types import Plugin as CorePluginType

# 1. Implement your application's KeyProvider
# This example uses environment variables. In production, use a secure vault.
class MyAppKeyProvider(KeyProvider, CorePluginType):
    plugin_id = "my_app_key_provider_v1" # Must be unique

    async def get_key(self, key_name: str) -> str | None:
        return os.environ.get(key_name)

    async def setup(self, config=None): # config is Optional[Dict[str, Any]]
        print(f"[{self.plugin_id}] KeyProvider setup.")

    async def teardown(self):
        print(f"[{self.plugin_id}] KeyProvider teardown.")

async def run_genie_demo():
    # 2. Create MiddlewareConfig
    # This example configures Ollama as the default LLM.
    app_config = MiddlewareConfig(
        default_llm_provider_id="ollama_llm_provider_v1",
        llm_provider_configurations={
            "ollama_llm_provider_v1": {"model_name": "mistral:latest"} # Ensure this model is pulled in Ollama
        },
        # Set a default command processor (e.g., for LLM-assisted tool use)
        default_command_processor_id="llm_assisted_tool_selection_processor_v1",
        command_processor_configurations={
            "llm_assisted_tool_selection_processor_v1": {
                "tool_formatter_id": "compact_text_formatter_plugin_v1", # Plugin ID of the formatter
            }
        },
        # Defaults for ToolLookupService (used by LLMAssistedToolSelectionProcessor)
        default_tool_indexing_formatter_id="compact_text_formatter_plugin_v1", # Plugin ID
        default_tool_lookup_provider_id="embedding_similarity_lookup_v1", # Plugin ID

        # Example RAG defaults (can be overridden in RAGInterface calls)
        default_rag_embedder_id="sentence_transformer_embedder_v1",
        default_rag_vector_store_id="faiss_vector_store_v1",
    )

    # 3. Instantiate your KeyProvider
    my_key_provider = MyAppKeyProvider()

    # 4. Create Genie instance
    # Genie.create handles initializing all managers and plugins based on the config.
    genie = await Genie.create(config=app_config, key_provider_instance=my_key_provider)
    print("Genie facade initialized!")

    # --- Example: LLM Chatting ---
    print("\n--- LLM Chat Example (Ollama/Mistral) ---")
    try:
        chat_response = await genie.llm.chat([{"role": "user", "content": "Hello, Genie! Tell me a short story."}])
        print(f"Genie LLM says: {chat_response['message']['content']}")
    except Exception as e:
        print(f"LLM Chat Error: {e}")

    # --- Example: RAG Indexing & Search ---
    print("\n--- RAG Example ---")
    try:
        # Create a dummy doc for demo
        with open("temp_doc_for_genie.txt", "w") as f:
            f.write("Genie Tooling makes building AI agents easier and more flexible.")
        
        # Index the current directory (will pick up temp_doc_for_genie.txt)
        # Uses default RAG components configured in MiddlewareConfig or Genie's own defaults.
        await genie.rag.index_directory(".", collection_name="my_docs_collection")
        
        rag_results = await genie.rag.search("What is Genie Tooling?", collection_name="my_docs_collection")
        if rag_results:
            print(f"RAG found: '{rag_results[0].content}' (Score: {rag_results[0].score:.2f})")
        else:
            print("RAG: No relevant documents found.")
        os.remove("temp_doc_for_genie.txt") # Cleanup
    except Exception as e:
        print(f"RAG Error: {e}")

    # --- Example: Running a Command (Tool Use) ---
    # This uses the 'llm_assisted_tool_selection_processor_v1' by default.
    # Ensure the 'calculator_tool' is discoverable (it's a built-in).
    # The LLM will need to understand how to call it based on its formatted definition.
    print("\n--- Command Execution Example (Calculator) ---")
    try:
        # Ensure your default LLM (Ollama/Mistral) is capable of function/tool calling
        # or can follow the JSON output format instruction in the processor's prompt.
        # You might need to adjust the system prompt for llm_assisted_tool_selection_processor_v1
        # if your LLM needs very specific instructions for tool selection JSON.
        command_text = "What is 123 plus 456?"
        print(f"Sending command: '{command_text}'")
        command_result = await genie.run_command(command_text)
        
        if command_result and command_result.get("tool_result"):
            print(f"Tool Result: {command_result['tool_result']}")
        elif command_result and command_result.get("error"):
             print(f"Command Error: {command_result['error']}")
        else:
             print(f"Command did not result in a tool call or error: {command_result}")
    except Exception as e:
        print(f"Command Execution Error: {e}")

    # 5. Teardown
    # This gracefully closes clients and releases resources.
    await genie.close()
    print("\nGenie facade torn down.")

if __name__ == "__main__":
    # For the demo to run, ensure any required API keys are set as environment variables
    # if your chosen plugins need them (e.g., OPENAI_API_KEY, OPENWEATHERMAP_API_KEY).
    # Also, ensure Ollama is running if using it: `ollama serve`
    # And that the model is pulled: `ollama pull mistral` (or your configured model)
    
    # Setup basic logging to see Genie's operations
    import logging
    logging.basicConfig(level=logging.INFO) # Use INFO for less verbosity, DEBUG for more
    # logging.getLogger("genie_tooling").setLevel(logging.DEBUG) # Uncomment for detailed library logs

    asyncio.run(run_genie_demo())
```

## Documentation

For more detailed information, guides, and API references, please refer to our [Full Documentation Site](https://your-docs-site-url.com). <!-- TODO: Update link -->

*   User Guide
*   Developer Guide (Creating Plugins)
*   API Reference
*   Tutorials

## Contributing

We welcome contributions!

Key contributor: [colonelpanik](https://github.com/colonelpanik)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
