# Welcome to Genie Tooling

Genie Tooling is a hyper-pluggable Python middleware designed to empower developers in building sophisticated Agentic AI and LLM-powered applications. It provides a modular, async-first framework that emphasizes clear interfaces and interchangeable components, allowing for rapid iteration and customization of agent capabilities.

## Key Features

*   **Simplified Development with the `Genie` Facade**: The primary entry point for most applications, offering a high-level API for common agentic tasks like LLM interaction, Retrieval Augmented Generation (RAG), tool execution, and natural language command processing.
*   **Hyper-Pluggable Architecture**: Almost every piece of functionality—from LLM providers and tools to data loaders and caching mechanisms—is a plugin. This allows you to easily swap, extend, or create custom components.
*   **Simplified Configuration**: Utilize `FeatureSettings` within `MiddlewareConfig` for quick setup of common features, with the flexibility for detailed overrides.
*   **Async First**: Built with `asyncio` for high-performance, I/O-bound operations common in AI applications.
*   **`@tool` Decorator**: Effortlessly convert your existing Python functions into Genie-compatible tools with automatic metadata and schema generation. By default, these tools are automatically enabled upon registration, but it is strongly recommended to explicitly enable them via `tool_configurations` in production for enhanced security.

## Quick Start

Get up and running quickly with the `Genie` facade:

```python
import asyncio
import logging
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.features import FeatureSettings
from genie_tooling.genie import Genie

async def main():
    logging.basicConfig(level=logging.INFO)

    app_config = MiddlewareConfig(
        features=FeatureSettings(
            llm="ollama", # Default: ollama, ensure it's running
            llm_ollama_model_name="mistral:latest",
            command_processor="llm_assisted",
            tool_lookup="embedding" 
        ),
        # Explicitly enable the built-in calculator tool.
        # If auto_enable_registered_tools=False, any custom @tool
        # would also need to be listed here after registration.
        tool_configurations={
            "calculator_tool": {} 
        }
    )
    genie = await Genie.create(config=app_config)
    print("Genie initialized!")

    # LLM Chat
    chat_response = await genie.llm.chat([{"role": "user", "content": "Tell me about Genie Tooling."}])
    print(f"LLM: {chat_response['message']['content']}")

    # Command Execution (e.g., calculator tool is built-in)
    cmd_result = await genie.run_command("What is 5 times 12?")
    print(f"Command Result: {cmd_result.get('tool_result')}")

    await genie.close()
    print("Genie torn down.")

if __name__ == "__main__":
    asyncio.run(main())
```

## Dive Deeper

*   **User Guide**: Learn how to install, configure, and use Genie Tooling for your projects.
    *   [Installation](guides/installation.md)
    *   [Configuration](guides/configuration.md) & [Simplified Configuration](guides/simplified_configuration.md)
    *   [Using LLM Providers](guides/using_llm_providers.md)
    *   [Using Tools](guides/using_tools.md)
    *   [Using RAG](guides/using_rag.md)
    *   [Using Command Processors](guides/using_command_processors.md)
    *   [Tool Lookup](guides/tool_lookup.md)
    *   [Logging](guides/logging.md)
    *   [Prompt Management](guides/using_prompts.md)
    *   [Conversation State](guides/using_conversation_state.md)
    *   [Observability & Tracing](guides/observability_tracing.md)
    *   [Human-in-the-Loop (HITL)](guides/using_human_in_loop.md)
    *   [Token Usage Tracking](guides/token_usage_tracking.md)
    *   [Guardrails](guides/using_guardrails.md)
    *   [Distributed Tasks](guides/distributed_tasks.md)
*   **Developer Guide**: Understand the plugin architecture and learn how to create your own custom plugins and tools.
    *   [Plugin Architecture](guides/plugin_architecture.md)
    *   [Creating Tool Plugins](guides/creating_tool_plugins.md)
    *   [Creating RAG Plugins](guides/creating_rag_plugins.md)
    *   [Creating Other Plugins](guides/creating_other_plugins.md)
*   **API Reference**: Detailed reference for all public modules and classes.
*   **Tutorials & Examples**: Step-by-step guides and practical examples to get you started.

We encourage you to explore the documentation and examples to unlock the full potential of Genie Tooling!
