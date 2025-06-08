# Genie Tooling

[![Pytest Status](https://github.com/genie-tooling/genie-tooling/actions/workflows/python_ci.yml/badge.svg)](https://github.com/genie-tooling/genie-tooling/actions/workflows/python_ci.yml)

A hyper-pluggable Python middleware for building sophisticated Agentic AI and LLM-powered applications.

## Vision

Genie Tooling empowers developers to construct complex AI agents by providing a modular, async-first framework. It emphasizes clear interfaces and interchangeable components, allowing for rapid iteration and customization of agent capabilities. The `Genie` facade offers a simplified, high-level API for common agentic tasks.

## Core Concepts

*   **`Genie` Facade**: The primary entry point for most applications. It simplifies interaction with all underlying managers and plugins, providing easy access to:
    *   LLM chat and generation (`genie.llm`)
    *   Retrieval Augmented Generation (`genie.rag`)
    *   Tool execution (`genie.execute_tool`)
    *   Natural language command processing (`genie.run_command`)
    *   Prompt management (`genie.prompts`)
    *   Conversation state (`genie.conversation`)
    *   Observability tracing (`genie.observability`)
    *   Human-in-the-loop approvals (`genie.human_in_loop`)
    *   Token usage tracking (`genie.usage`)
    *   Distributed task submission (`genie.task_queue`)
*   **Plugins**: Genie is built around a plugin architecture. Almost every piece of functionality (LLM interaction, tool definition, data retrieval, caching, guardrails, task queuing, etc.) is a plugin that can be swapped or extended.
*   **Explicit Tool Enablement**: Tools are only loaded and made available if they are explicitly configured in `MiddlewareConfig.tool_configurations`. An entry like `{"my_tool_id": {}}` is sufficient to enable a tool that requires no specific configuration. See the [Using Tools Guide](docs/guides/using_tools.md) for details.
*   **Managers**: Specialized managers (e.g., `ToolManager`, `RAGManager`, `LLMProviderManager`, `GuardrailManager`, `DistributedTaskQueueManager`) orchestrate their respective plugin types, typically managed internally by the `Genie` facade.
*   **Configuration**: Applications provide runtime configuration (e.g., API keys, default plugin choices, plugin-specific settings) via a `MiddlewareConfig` object, often simplified using `FeatureSettings`, and a custom `KeyProvider` implementation (or the default `EnvironmentKeyProvider`).
*   **`@tool` Decorator**: Easily turn your Python functions into Genie-compatible tools with automatic metadata generation. Remember to register these decorated functions using `await genie.register_tool_functions([...])` and **enable them via `tool_configurations`** in your `MiddlewareConfig`.

## Key Plugin Categories

Genie Tooling supports a wide array of plugin types:

*   **LLM Providers**: Interface with LLM APIs (e.g., OpenAI, Ollama, Gemini, Llama.cpp server, Llama.cpp internal).
*   **Command Processors**: Interpret user commands to select tools and extract parameters.
*   **Tools**: Define discrete actions the agent can perform (e.g., calculator, web search, file operations, or functions decorated with `@tool`). **Must be enabled in `tool_configurations`.**
*   **Key Providers**: Securely supply API keys.
*   **RAG Components**: Document Loaders, Text Splitters, Embedding Generators, Vector Stores (e.g., FAISS, ChromaDB, Qdrant), Retrievers.
*   **Tool Lookup Providers**: Help find relevant tools based on natural language (embedding, keyword-based, or hybrid).
*   **Definition Formatters**: Format tool metadata for LLMs or indexing (e.g., compact text, OpenAI functions JSON).
*   **Invocation Strategies**: Define the lifecycle of a tool call (e.g., default async, distributed task offloading).
*   **Input Validators**: Validate parameters passed to tools (e.g., JSON Schema).
*   **Output Transformers**: Transform raw tool output.
*   **Error Handlers & Formatters**: Process and format errors from tool execution.
*   **Caching Providers**: Cache tool results or other data (e.g., in-memory, Redis).
*   **Logging & Redaction Adapters**: Customize logging behavior and redact sensitive data.
*   **Code Executors**: Securely execute code snippets (e.g., Docker-based, pysandbox stub).
*   **Observability Tracers**: Record interaction traces (e.g., console, OpenTelemetry).
*   **HITL Approvers**: Handle human approval steps (e.g., CLI-based).
*   **Token Usage Recorders**: Track LLM token consumption (e.g., in-memory, OpenTelemetry metrics).
*   **Guardrail Plugins**: Enforce input, output, and tool usage policies (e.g., keyword blocklists).
*   **Prompt System**: Prompt Registries (e.g., file system) and Prompt Template Engines (e.g., basic string format, Jinja2).
*   **Conversation State Providers**: Manage conversation history (e.g., in-memory, Redis).
*   **LLM Output Parsers**: Structure LLM text responses (e.g., JSON, Pydantic models).
*   **Distributed Task Queues**: Interface with systems like Celery or RQ for offloading tasks.

*(Refer to `pyproject.toml` for a list of built-in plugin entry points and their default identifiers, and `src/genie_tooling/config/resolver.py` for available aliases).*

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/genie-tooling/genie-tooling.git
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
    You can install specific extras like `poetry install --extras ollama openai qdrant celery llama_cpp_internal` as needed. Check `pyproject.toml` for available extras like `web_tools`, `local_rag`, `distributed_rag`, `observability`, `task_queues`, `llama_cpp_server`, `llama_cpp_internal`, etc.

## Quick Start with the `Genie` Facade (Local-Only)

The `Genie` facade is the recommended way to get started. This example showcases local LLM chat (Llama.cpp internal), RAG with local FAISS, and local code execution.

```python
import asyncio
import os
import logging
import json # Added for token usage output
from pathlib import Path
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.features import FeatureSettings
from genie_tooling.genie import Genie

async def run_genie_quick_start():
    logging.basicConfig(level=logging.INFO)
    # For detailed library logs:
    # logging.getLogger("genie_tooling").setLevel(logging.DEBUG)

    # --- IMPORTANT: Local Model Configuration ---
    # !!! USER ACTION REQUIRED !!!
    # Download a GGUF model (e.g., from Hugging Face) and update the path below.
    # Example models: mistral-7b-instruct-v0.2.Q4_K_M.gguf, llama-2-7b-chat.Q4_K_M.gguf
    # Ensure the model chosen is compatible with the chat_format specified.
    local_gguf_model_path_str = "/path/to/your/model.gguf"  # <--- !!! CHANGE THIS PATH !!!
    # --- End of User Action Required ---

    local_gguf_model_path = Path(local_gguf_model_path_str)
    if local_gguf_model_path_str == "/path/to/your/model.gguf" or not local_gguf_model_path.exists():
        print("\nERROR: Local GGUF model path not configured or file does not exist.")
        print("Please edit the 'local_gguf_model_path_str' variable in this script")
        print(f"to point to a valid GGUF model file on your system. Current path: '{local_gguf_model_path_str}'")
        return

    app_config = MiddlewareConfig(
        features=FeatureSettings(
            # LLM: Use internal Llama.cpp
            llm="llama_cpp_internal",
            llm_llama_cpp_internal_model_path=str(local_gguf_model_path.resolve()),
            llm_llama_cpp_internal_n_gpu_layers=-1, # Offload all layers to GPU if available, 0 for CPU
            llm_llama_cpp_internal_chat_format="mistral-instruct", # Corrected from "mistral"
            llm_llama_cpp_internal_n_ctx=2048, # Context size

            # Command Processing & Tool Lookup (local)
            command_processor="llm_assisted",
            tool_lookup="hybrid", # Uses embedding + keyword search. In-memory by default.
            tool_lookup_embedder_id_alias="st_embedder", # Local sentence-transformer

            # RAG (local)
            rag_embedder="sentence_transformer", # Local sentence-transformer
            rag_vector_store="faiss",            # Local FAISS

            # Other local features
            cache="in-memory",                   # In-memory cache
            observability_tracer="console_tracer",
            hitl_approver="none",                # HITL disabled
            token_usage_recorder="in_memory_token_recorder",
            input_guardrails=["keyword_blocklist_guardrail"],
            task_queue="none",                   # No distributed task queue
        ),
        tool_configurations={
            "calculator_tool": {},
            "sandboxed_fs_tool_v1": {"sandbox_base_path": "./my_agent_sandbox"},
            "generic_code_execution_tool": {}, # Uses PySandboxExecutorStub by default (local, insecure)
        },
        guardrail_configurations={
            "keyword_blocklist_guardrail_v1": {
                "blocklist": ["secret_project_alpha", "highly_classified_info"],
                "action_on_match": "block"
            }
        },
    )

    genie = await Genie.create(config=app_config)
    print(f"Genie facade initialized with Llama.cpp model: {local_gguf_model_path.name}")

    # --- Example: LLM Chatting ---
    print("\n--- LLM Chat Example (Llama.cpp Internal) ---")
    try:
        chat_response = await genie.llm.chat([{"role": "user", "content": "Hello, Genie! Tell me a short story about a friendly local AI."}])
        print(f"Genie LLM says: {chat_response['message']['content']}")
    except PermissionError as e_perm:
        print(f"LLM Chat Blocked by Guardrail: {e_perm}")
    except Exception as e:
        print(f"LLM Chat Error: {e} (Is your GGUF model path correct and model compatible with 'mistral-instruct' format?)")

    # --- Example: RAG Indexing & Search ---
    print("\n--- RAG Example (Local FAISS) ---")
    try:
        dummy_doc_path = Path("./my_agent_sandbox/temp_doc_for_genie.txt")
        dummy_doc_path.parent.mkdir(parents=True, exist_ok=True)
        dummy_doc_path.write_text("Genie Tooling makes building local AI agents easier and more flexible.")

        await genie.rag.index_directory("./my_agent_sandbox", collection_name="my_local_docs_collection")

        rag_results = await genie.rag.search("What is Genie Tooling?", collection_name="my_local_docs_collection")
        if rag_results:
            print(f"RAG found: '{rag_results[0].content}' (Score: {rag_results[0].score:.2f})")
        else:
            print("RAG: No relevant documents found.")
        dummy_doc_path.unlink(missing_ok=True)
    except Exception as e:
        print(f"RAG Error: {e}")

    # --- Example: Running a Command (Code Execution, HITL Disabled) ---
    print("\n--- Command Execution Example (Code Execution, HITL Disabled) ---")
    try:
        command_text = "Execute the following Python code: print(f'The sum of 7 and 8 is {{7 + 8}}')"
        print(f"Sending command: '{command_text}' (Code execution, HITL disabled)")
        command_result = await genie.run_command(command_text)

        if command_result and command_result.get("tool_result"):
            tool_res_data = command_result['tool_result']
            print(f"Tool Result (Code Execution):")
            if tool_res_data.get('stdout'):
                print(f"  Stdout: {tool_res_data.get('stdout', '').strip()}")
            if tool_res_data.get('stderr'):
                print(f"  Stderr: {tool_res_data.get('stderr', '').strip()}")
            if tool_res_data.get('error'):
                print(f"  Executor Error: {tool_res_data.get('error')}")
            if tool_res_data.get('result') is not None:
                 print(f"  Result object: {tool_res_data.get('result')}")
            # Ensure execution_time_ms is handled if it's None or not a float
            exec_time = tool_res_data.get('execution_time_ms')
            if isinstance(exec_time, (int, float)):
                print(f"  Execution Time (ms): {exec_time:.2f}")
            else:
                print(f"  Execution Time (ms): N/A")


        elif command_result and command_result.get("error"):
             print(f"Command Error: {command_result['error']}")
        else:
             print(f"Command did not result in a tool call or error: {command_result}")
    except Exception as e:
        print(f"Command Execution Error: {e}")

    # --- Example: Token Usage Summary ---
    print("\n--- Token Usage Summary ---")
    usage_summary = await genie.usage.get_summary()
    print(json.dumps(usage_summary, indent=2))

    await genie.close()
    print("\nGenie facade torn down.")

if __name__ == "__main__":
    asyncio.run(run_genie_quick_start())
```

This quick start demonstrates several key features. For more examples, including `genie.prompts`, `genie.conversation`, `genie.task_queue`, direct `genie.execute_tool`, the `@tool` decorator, and advanced configurations, please explore the `/examples` directory and the full [Documentation](https://genie-tooling.github.io/genie-tooling/).

## Dive Deeper

*   **Comprehensive Examples**: Explore the `/examples` directory in the repository for detailed code examples showcasing various features.
*   **Full Documentation**: For more detailed information, guides, and API references, please refer to the [Genie Tooling Documentation Site](https://genie-tooling.github.io/genie-tooling/) (or browse the `docs/` directory locally). Key guides include:
    *   User Guide (Installation, Configuration, Using LLMs, Tools, RAG, Command Processors, Prompts, Conversation State, Observability, HITL, Token Usage, Guardrails, Distributed Tasks)
    *   Developer Guide (Creating Plugins, Using `@tool`)
    *   API Reference
    *   Tutorials & Examples

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Key contributor: [colonelpanik](https://github.com/colonelpanik)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
