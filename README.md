# Genie Tooling

[![Pytest Status](https://github.com/genie-tooling/genie-tooling/actions/workflows/python_ci.yml/badge.svg)](https://github.com/genie-tooling/genie-tooling/actions/workflows/python_ci.yml)

A hyper-pluggable Python middleware for building sophisticated Agentic AI and LLM-powered applications.

## Vision

Genie Tooling empowers developers to construct complex AI agents by providing a modular, async-first framework. It emphasizes clear interfaces and interchangeable components, allowing for rapid iteration and customization of agent capabilities. The `Genie` facade offers a simplified, high-level API for common agentic tasks.

## Core Concepts

*   **`Genie` Facade**: The primary entry point for most applications. It simplifies interaction with all underlying managers and plugins.
*   **Plugins & Extensibility**: Genie is built around a plugin architecture. Almost every piece of functionality (LLM interaction, tool definition, data retrieval, caching, guardrails, task queuing, etc.) is a plugin that can be swapped or extended. The framework includes a **Bootstrap Plugin** system for creating self-contained extensions.
*   **Explicit Tool Enablement (Production Safety)**: Tools are only active if they are explicitly enabled in the configuration (`tool_configurations`). This provides a clear, secure manifest of an agent's capabilities, preventing accidental exposure of development tools in production. The `auto_enable_registered_tools` flag can be set to `True` for rapid development, but `False` is the recommended production setting.
*   **`@tool` Decorator**: Easily turn your Python functions into Genie-compatible tools with automatic metadata generation.
*   **Zero-Effort Observability**: The framework is deeply instrumented. By simply enabling a tracer (e.g., `observability_tracer="console_tracer"`), developers get detailed, correlated traces for all internal operations. This is achieved by decoupling tracing from logging: tracers emit events, which are then processed by a configurable `LogAdapterPlugin` (e.g., `DefaultLogAdapter` or the rich `PyviderTelemetryLogAdapter`).

## Key Plugin Categories

Genie Tooling supports a wide array of plugin types:

*   **LLM Providers**: Interface with LLM APIs (e.g., OpenAI, Ollama, Gemini, Llama.cpp server, **Llama.cpp internal**).
*   **Command Processors**: Interpret user commands to select tools and extract parameters (e.g., `llm_assisted`, `rewoo`).
*   **Tools**: Define discrete actions the agent can perform.
*   **Key Providers**: Securely supply API keys.
*   **RAG Components**: Document Loaders, Text Splitters, Embedding Generators, Vector Stores.
*   **Observability**: Tracers (`ConsoleTracerPlugin`, `OpenTelemetryTracerPlugin`), Log Adapters (`DefaultLogAdapter`, `PyviderTelemetryLogAdapter`), and Token Recorders (`in_memory`, `otel_metrics`).
*   ...and many more, including Caching, Guardrails, HITL, Prompts, and **Distributed Task Queues (Celery, RQ)**.

*(Refer to `pyproject.toml` for a list of built-in plugin entry points and `src/genie_tooling/config/resolver.py` for available aliases).*

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/genie-tooling/genie-tooling.git
    cd genie-tooling
    ```

2.  **Install dependencies using Poetry:**
    (Ensure [Poetry](https://python-poetry.org/docs/#installation) is installed.)
    ```bash
    poetry install --all-extras
    ```

## Quick Start with the `Genie` Facade (Local-Only)

This example showcases a fully local setup: LLM chat (via the internal Llama.cpp provider), RAG with local FAISS, and local code execution, with rich console tracing enabled for visibility.

```python
import asyncio
import json
import logging
from pathlib import Path

from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie


async def run_genie_quick_start():
    logging.basicConfig(level=logging.INFO)
    # For detailed library logs:
    # logging.getLogger("genie_tooling").setLevel(logging.DEBUG)

    # --- IMPORTANT: Local Model Configuration ---
    # !!! USER ACTION REQUIRED !!!
    # Download a GGUF model (e.g., from Hugging Face) and update the path below.
    # Example models: mistral-7b-instruct-v0.2.Q4_K_M.gguf, llama-3-8b-instruct.Q4_K_M.gguf
    # Ensure the model chosen is compatible with the specified chat_format.
    local_gguf_model_path_str = "/path/to/your/model.gguf"  # <--- !!! CHANGE THIS PATH !!!
    # --- End of User Action Required ---

    local_gguf_model_path = Path(local_gguf_model_path_str)
    if not local_gguf_model_path.exists() or "/path/to/your/model.gguf" in local_gguf_model_path_str:
        print("\nERROR: Local GGUF model path not configured or file does not exist.")
        print(f"Please edit the 'local_gguf_model_path_str' variable in '{__file__}'")
        print(f"to point to a valid GGUF model file on your system. Current path: '{local_gguf_model_path_str}'")
        return

    app_config = MiddlewareConfig(
        # For production, set to False and list all tools in tool_configurations
        auto_enable_registered_tools=True,
        features=FeatureSettings(
            # LLM: Use internal Llama.cpp
            llm="llama_cpp_internal",
            llm_llama_cpp_internal_model_path=str(local_gguf_model_path.resolve()),
            llm_llama_cpp_internal_n_gpu_layers=-1, # Offload all layers to GPU if available
            llm_llama_cpp_internal_n_ctx=4096, # Context size

            # Command Processing & Tool Lookup (local)
            command_processor="llm_assisted",
            tool_lookup="embedding",

            # RAG (local)
            rag_embedder="sentence_transformer",
            rag_vector_store="faiss",

            # Observability & Logging
            observability_tracer="console_tracer",
            logging_adapter="pyvider_log_adapter", # Use rich logging
        ),
        # Explicitly enable the tools we want to use.
        tool_configurations={
            "calculator_tool": {},
            "sandboxed_fs_tool_v1": {"sandbox_base_path": "./my_agent_sandbox"},
            "generic_code_execution_tool": {},
        },
    )

    genie = await Genie.create(config=app_config)
    print(f"Genie facade initialized with Llama.cpp model: {local_gguf_model_path.name}")

    # --- Example: LLM Chatting ---
    print("\n--- LLM Chat Example (Llama.cpp Internal) ---")
    try:
        chat_response = await genie.llm.chat([{"role": "user", "content": "Hello, Genie! Tell me a short story about a friendly local AI."}])
        print(f"Genie LLM says: {chat_response['message']['content']}")
    except Exception as e:
        print(f"LLM Chat Error: {e} (Is your GGUF model path correct and model compatible?)")

    # --- Example: RAG Indexing & Search ---
    print("\n--- RAG Example (Local FAISS) ---")
    try:
        dummy_doc_path = Path("./my_agent_sandbox/temp_doc_for_genie.txt")
        dummy_doc_path.parent.mkdir(parents=True, exist_ok=True)
        dummy_doc_path.write_text("Genie Tooling makes building local AI agents easier and more flexible.")
        await genie.rag.index_directory("./my_agent_sandbox", collection_name="my_local_docs")

        rag_results = await genie.rag.search("What is Genie Tooling?", collection_name="my_local_docs")
        if rag_results:
            print(f"RAG found: '{rag_results[0].content}' (Score: {rag_results[0].score:.2f})")
        else:
            print("RAG: No relevant documents found.")
        dummy_doc_path.unlink(missing_ok=True)
    except Exception as e:
        print(f"RAG Error: {e}")

    # --- Example: Running a Command ---
    print("\n--- Command Execution Example ---")
    try:
        command_text = "Execute the following Python code: print(f'The sum of 7 and 8 is {{7 + 8}}')"
        command_result = await genie.run_command(command_text)

        if command_result and command_result.get("tool_result"):
            tool_res_data = command_result["tool_result"]
            print("Tool Result (Code Execution):")
            if tool_res_data.get("stdout"):
                print(f"  Stdout: {tool_res_data.get('stdout', '').strip()}")
        elif command_result and command_result.get("error"):
             print(f"Command Error: {command_result['error']}")
        else:
             print(f"Command did not result in a tool call or error: {command_result}")
    except Exception as e:
        print(f"Command Execution Error: {e}")

    await genie.close()
    print("\nGenie facade torn down.")

if __name__ == "__main__":
    asyncio.run(run_genie_quick_start())
```
