# Genie Tooling

[![Pytest Status](https://github.com/genie-tooling/genie-tooling/actions/workflows/python_ci.yml/badge.svg)](https://github.com/genie-tooling/genie-tooling/actions/workflows/python_ci.yml)
<!-- TODO: Replace with actual CI badge URL once repo is public -->

A hyper-pluggable Python middleware for building sophisticated Agentic AI and LLM-powered applications.

## Vision

Genie Tooling empowers developers to construct complex AI agents by providing a modular, async-first framework. It emphasizes clear interfaces and interchangeable components, allowing for rapid iteration and customization of agent capabilities. The `Genie` facade offers a simplified, high-level API for common agentic tasks.

## Core Concepts

*   **`Genie` Facade**: The primary entry point for most applications. It simplifies interaction with all underlying managers and plugins, providing easy access to:
    *   LLM chat and generation (`genie.llm`)
    *   Retrieval Augmented Generation (`genie.rag`)
    *   Tool execution (`genie.execute_tool`)
    *   Natural language command processing (`genie.run_command`)
*   **Plugins**: Genie is built around a plugin architecture. Almost every piece of functionality (LLM interaction, tool definition, data retrieval, caching, etc.) is a plugin that can be swapped or extended.
*   **Managers**: Specialized managers (e.g., `ToolManager`, `RAGManager`, `LLMProviderManager`) orchestrate their respective plugin types, typically managed internally by the `Genie` facade.
*   **Configuration**: Applications provide runtime configuration (e.g., API keys, default plugin choices, plugin-specific settings) via a `MiddlewareConfig` object, often simplified using `FeatureSettings`, and a custom `KeyProvider` implementation (or the default `EnvironmentKeyProvider`).
*   **`@tool` Decorator**: Easily turn your Python functions into Genie-compatible tools with automatic metadata generation.

## Key Plugin Categories

Genie Tooling supports a wide array of plugin types:

*   **LLM Providers**: Interface with LLM APIs (e.g., OpenAI, Ollama, Gemini).
*   **Command Processors**: Interpret user commands to select tools and extract parameters.
*   **Tools**: Define discrete actions the agent can perform (e.g., calculator, web search, file operations, or functions decorated with `@tool`).
*   **Key Provider**: Securely supplies API keys (implemented by the application or using the default environment provider).
*   **RAG Components**: Document Loaders, Text Splitters, Embedding Generators, Vector Stores, Retrievers.
*   **Tool Lookup Providers**: Help find relevant tools based on natural language.
*   **And more**: Invocation Strategies, Caching Providers, Code Executors, Logging & Redaction Adapters, Definition Formatters.

*(Refer to `pyproject.toml` for a list of built-in plugin entry points and their default identifiers, and `src/genie_tooling/config/resolver.py` for available aliases).*

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

The `Genie` facade is the recommended way to get started.

```python
import asyncio
import os
import logging
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.features import FeatureSettings
from genie_tooling.genie import Genie
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.core.types import Plugin as CorePluginType

# Optional: Define a custom KeyProvider if needed
# class MyAppKeyProvider(KeyProvider, CorePluginType):
#     plugin_id = "my_app_key_provider_v1"
#     async def get_key(self, key_name: str) -> str | None:
#         # Your secure key retrieval logic here
#         return os.environ.get(key_name.upper()) # Example
#     async def setup(self, config=None): pass
#     async def teardown(self): pass

async def run_genie_quick_start():
    # Configure logging to see Genie's operations
    logging.basicConfig(level=logging.INFO)
    # For more detailed library logs:
    # logging.getLogger("genie_tooling").setLevel(logging.DEBUG)

    # 1. Configure Middleware using FeatureSettings for simplicity
    # Ensure Ollama is running (ollama serve) and mistral model is pulled (ollama pull mistral)
    # Set OPENAI_API_KEY environment variable if testing with OpenAI.
    app_config = MiddlewareConfig(
        features=FeatureSettings(
            llm="ollama", # Default LLM
            llm_ollama_model_name="mistral:latest",
            llm_openai_model_name="gpt-3.5-turbo", # For OpenAI if selected
            
            command_processor="llm_assisted", # Use LLM for tool selection
            tool_lookup="embedding", # Use embedding-based tool lookup
            
            rag_embedder="sentence_transformer", # For RAG
            rag_vector_store="faiss" # For RAG
        ),
        # Example of overriding a specific plugin's configuration
        tool_configurations={
            "sandboxed_fs_tool_v1": {"sandbox_base_path": "./my_agent_sandbox"}
        }
    )

    # 2. Instantiate Genie
    # Genie uses EnvironmentKeyProvider by default if key_provider_instance is not given.
    # my_custom_kp = MyAppKeyProvider() # Uncomment if using a custom one
    genie = await Genie.create(config=app_config) #, key_provider_instance=my_custom_kp)
    print("Genie facade initialized!")

    # --- Example: LLM Chatting (using default 'ollama') ---
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
        dummy_doc_path = Path("./temp_doc_for_genie.txt")
        dummy_doc_path.write_text("Genie Tooling makes building AI agents easier and more flexible.")
        
        await genie.rag.index_directory(".", collection_name="my_docs_collection")
        
        rag_results = await genie.rag.search("What is Genie Tooling?", collection_name="my_docs_collection")
        if rag_results:
            print(f"RAG found: '{rag_results[0].content}' (Score: {rag_results[0].score:.2f})")
        else:
            print("RAG: No relevant documents found.")
        dummy_doc_path.unlink(missing_ok=True)
    except Exception as e:
        print(f"RAG Error: {e}")

    # --- Example: Running a Command (Tool Use via LLM-assisted processor) ---
    print("\n--- Command Execution Example (Calculator) ---")
    try:
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
    
    # --- Example: Using a specific tool directly ---
    print("\n--- Direct Tool Execution (Sandboxed File System) ---")
    try:
        await genie.execute_tool(
            "sandboxed_fs_tool_v1",
            operation="write_file",
            path="readme_quickstart.txt",
            content="File written by Genie Quick Start!"
        )
        print("Wrote 'readme_quickstart.txt' to sandbox.")
        read_back = await genie.execute_tool(
            "sandboxed_fs_tool_v1",
            operation="read_file",
            path="readme_quickstart.txt"
        )
        if read_back.get("success"):
            print(f"Read back: {read_back.get('content')}")
    except Exception as e:
        print(f"Filesystem Tool Error: {e}")


    # 3. Teardown
    await genie.close()
    print("\nGenie facade torn down.")

if __name__ == "__main__":
    from pathlib import Path # For dummy doc path in quick start
    asyncio.run(run_genie_quick_start())
```

## Documentation

For more detailed information, guides, and API references, please refer to our [Full Documentation Site](https://your-docs-site-url.com). <!-- TODO: Update link -->

*   User Guide (including Simplified Configuration)
*   Developer Guide (Creating Plugins, Using `@tool`)
*   API Reference
*   Tutorials & Examples

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` (TODO: create this file) for guidelines.

Key contributor: [colonelpanik](https://github.com/colonelpanik)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
