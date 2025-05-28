![Pytest Status](https://github.com/genie-tooling/genie-tooling/actions/workflows/python_ci.yml/badge.svg)

# Genie Tooling

A hyper-pluggable Python middleware for Agentic AI and LLM applications.

## Overview

Genie Tooling provides a comprehensive, async-first suite of components for building sophisticated AI agents and LLM-powered applications. It emphasizes modularity and clear interfaces, allowing developers to easily swap implementations for various functionalities. The core of the library is the `Genie` facade, which simplifies interaction with all underlying managers and plugins.

## Core Plugin Categories & Default Implementations

Genie Tooling is built around a plugin architecture. Here's a look at the main categories and example/default plugins:

*   **`LLMProviderPlugin`**: Interfaces with Large Language Model APIs.
    *   *Example/Default (to be configured)*: `ollama_llm_provider_v1` (for local Ollama models), `gemini_llm_provider_v1` (for Google Gemini), `openai_llm_provider_v1` (for OpenAI models). Your application configures the default.

*   **`CommandProcessorPlugin`**: Interprets user commands to select tools and extract parameters.
    *   *Example/Default (to be configured)*: `llm_assisted_tool_selection_processor_v1` (uses an LLM for selection), or `simple_keyword_processor_v1` (for basic keyword matching).

*   **`ToolPlugin`**: Defines discrete capabilities or actions the agent can perform.
    *   *Built-in Examples*: `calculator_tool`, `open_weather_map_tool`, `generic_code_execution_tool`.

*   **`KeyProvider`**: Securely supplies API keys to other plugins.
    *   *Note*: This plugin **must be implemented by the consuming application** (e.g., `EnvironmentKeyProvider` in examples).

*   **RAG (Retrieval Augmented Generation) Plugins**:
    *   **`DocumentLoaderPlugin`**: Loads data from various sources.
        *   *Default Example*: `file_system_loader_v1`, `web_page_loader_v1`.
    *   **`TextSplitterPlugin`**: Divides documents into manageable chunks.
        *   *Default Example*: `character_recursive_text_splitter_v1`.
    *   **`EmbeddingGeneratorPlugin`**: Creates vector embeddings for text.
        *   *Default Example*: `sentence_transformer_embedder_v1` (local), `openai_embedding_generator_v1`.
    *   **`VectorStorePlugin`**: Stores and searches embeddings.
        *   *Default Example*: `faiss_vector_store_v1` (local in-memory/file), `chromadb_vector_store_v1`.
    *   **`RetrieverPlugin`**: Orchestrates the retrieval process (embedding query + vector store search).
        *   *Default Example*: `basic_similarity_retriever_v1`.

*   **`ToolLookupProviderPlugin`**: Helps find relevant tools based on a natural language query.
    *   *Default Example*: `embedding_similarity_lookup_v1`.
    *   **`DefinitionFormatterPlugin`** (used by ToolLookupService for indexing): Formats tool metadata for lookup.
        *   *Default for Lookup*: `llm_compact_text_v1` (from `compact_text_formatter_plugin_v1`).

*   **`InvocationStrategy`**: Defines the lifecycle of a tool call (validation, execution, transformation, caching).
    *   *Default*: `default_async_invocation_strategy_v1`.
    *   Sub-plugins used by strategies:
        *   **`InputValidator`**: *Default*: `jsonschema_input_validator_v1`.
        *   **`OutputTransformer`**: *Default*: `passthrough_output_transformer_v1`.
        *   **`ErrorHandler`**: *Default*: `default_error_handler_v1`.
        *   **`ErrorFormatter`**: *Default*: `llm_error_formatter_v1`.

*   **`CacheProviderPlugin`**: Provides caching mechanisms.
    *   *Examples*: `in_memory_cache_provider_v1`, `redis_cache_provider_v1`.

*   **`CodeExecutorPlugin`** (used by `GenericCodeExecutionTool`): Executes code in a sandboxed environment.
    *   *Built-in Example (Stub)*: `pysandbox_executor_stub_v1` (**Warning: Insecure stub**).

*   **`LogAdapterPlugin`**: Adapts logging to different systems or formats.
    *   *Default Example*: `default_log_adapter_v1`.
*   **`RedactorPlugin`**: Sanitizes data to remove sensitive information.
    *   *Default Example*: `noop_redactor_v1`.

## Installation

```bash
# Ensure poetry is installed
git clone https://github.com/YourName/genie-tooling.git # TODO: Update URL
cd genie-tooling
poetry install --all-extras # Install with all optional dependencies for full functionality
```

## Configuration

The consuming application is responsible for providing API keys via a `KeyProvider`
implementation and other configurations (via `MiddlewareConfig`) at runtime.
See `examples/` for demonstrations.

## Usage with Genie Facade

The primary way to interact with the library is through the `Genie` facade:

```python
import asyncio
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie
# Import your application's KeyProvider implementation

async def my_app():
    # 1. Define your KeyProvider
    # class MyAppKeyProvider(KeyProvider, CorePluginType): ...

    # 2. Create MiddlewareConfig
    config = MiddlewareConfig(
        default_llm_provider_id="your_chosen_llm_provider_id",
        llm_provider_configurations={
            "your_chosen_llm_provider_id": {"model_name": "your_model"}
        }
        # ... other configurations
    )

    # 3. Create Genie instance
    # my_key_provider = MyAppKeyProvider()
    # genie = await Genie.create(config=config, key_provider_instance=my_key_provider)

    # 4. Use Genie!
    # result = await genie.llm.chat([{"role": "user", "content": "Hello!"}])
    # print(result['message']['content'])

    # rag_search_results = await genie.rag.search("What is pluggability?")
    # tool_output = await genie.execute_tool("calculator_tool", num1=5, num2=10, operation="add")

    # 5. Teardown
    # await genie.close()

# if __name__ == "__main__":
#     asyncio.run(my_app())
```
(Full examples are available in the `/examples` directory.)

## Contributing

[colonelpanik](https://github.com/colonelpanik)

## License
MIT
