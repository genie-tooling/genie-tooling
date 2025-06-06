# Creating Other Plugins

Beyond tools and RAG components, Genie Tooling's pluggable architecture allows for customization of many other functionalities. This guide provides an overview of how to create various other types of plugins.

## General Plugin Principles

All plugins in Genie Tooling typically adhere to the `genie_tooling.core.types.Plugin` protocol:

```python
from typing import Protocol, Optional, Dict, Any

class Plugin(Protocol):
    @property
    def plugin_id(self) -> str:
        ... # Unique identifier for the plugin

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        pass # Optional async setup

    async def teardown(self) -> None:
        pass # Optional async teardown
```

Your custom plugin class should:
1.  Define a unique `plugin_id` as a class attribute.
2.  Implement the specific protocol for the type of plugin you are creating (e.g., `LLMProviderPlugin`, `CacheProviderPlugin`).
3.  Implement `async def setup(self, config: Optional[Dict[str, Any]] = None)` if your plugin requires initialization with configuration.
4.  Implement `async def teardown(self)` if your plugin needs to release resources.

## Registering Your Plugin

Make your plugin discoverable by Genie:
*   **Entry Points**: Define an entry point in your `pyproject.toml` under the `[tool.poetry.plugins."genie_tooling.plugins"]` group.
    ```toml
    [tool.poetry.plugins."genie_tooling.plugins"]
    "my_custom_cache_v1" = "my_package.my_module:MyCustomCacheProvider"
    ```
*   **Plugin Development Directories**: Place your plugin's Python file in a directory specified in `MiddlewareConfig.plugin_dev_dirs`.

## Common Plugin Categories and Their Protocols

Refer to the `src/genie_tooling/` subdirectories for the specific abstract base classes (ABCs) or protocols for each plugin type. Key examples include:

*   **Key Providers**: `genie_tooling.security.key_provider.KeyProvider`
    *   Implement `async def get_key(self, key_name: str) -> Optional[str]`
*   **LLM Providers**: `genie_tooling.llm_providers.abc.LLMProviderPlugin`
    *   Implement `async def generate(...)` and/or `async def chat(...)`.
*   **Command Processors**: `genie_tooling.command_processors.abc.CommandProcessorPlugin`
    *   Implement `async def process_command(...)`.
*   **Definition Formatters**: `genie_tooling.definition_formatters.abc.DefinitionFormatter`
    *   Implement `def format(...)`.
*   **Caching Providers**: `genie_tooling.cache_providers.abc.CacheProvider`
    *   Implement `async def get(...)`, `async def set(...)`, `async def delete(...)`, etc.
*   **Invocation Strategies**: `genie_tooling.invocation_strategies.abc.InvocationStrategy`
    *   Implement `async def invoke(...)`.
*   **Input Validators**: `genie_tooling.input_validators.abc.InputValidator`
    *   Implement `def validate(...)`.
*   **Output Transformers**: `genie_tooling.output_transformers.abc.OutputTransformer`
    *   Implement `def transform(...)`.
*   **Error Handlers & Formatters**: `genie_tooling.error_handlers.abc.ErrorHandler`, `genie_tooling.error_formatters.abc.ErrorFormatter`
*   **Log Adapters & Redactors**: `genie_tooling.log_adapters.abc.LogAdapter`, `genie_tooling.redactors.abc.Redactor`
*   **Code Executors**: `genie_tooling.code_executors.abc.CodeExecutor`
*   **Observability Tracers**: `genie_tooling.observability.abc.InteractionTracerPlugin`
*   **HITL Approvers**: `genie_tooling.hitl.abc.HumanApprovalRequestPlugin`
*   **Token Usage Recorders**: `genie_tooling.token_usage.abc.TokenUsageRecorderPlugin`
*   **Guardrail Plugins**: `genie_tooling.guardrails.abc.InputGuardrailPlugin`, `OutputGuardrailPlugin`, `ToolUsageGuardrailPlugin`
*   **Prompt System Plugins**: `genie_tooling.prompts.abc.PromptRegistryPlugin`, `PromptTemplatePlugin`
*   **Conversation State Providers**: `genie_tooling.prompts.conversation.impl.abc.ConversationStateProviderPlugin`
*   **LLM Output Parsers**: `genie_tooling.prompts.llm_output_parsers.abc.LLMOutputParserPlugin`
*   **Distributed Task Queues**: `genie_tooling.task_queues.abc.DistributedTaskQueuePlugin`

## Configuration

Once your plugin is created and registered, you can configure it in `MiddlewareConfig` using the appropriate `*_configurations` dictionary, keyed by your plugin's `plugin_id`.

```python
# In your MiddlewareConfig
app_config = MiddlewareConfig(
    # ...
    cache_provider_configurations={
        "my_custom_cache_v1": {
            "connection_string": "...",
            "default_ttl": 300
        }
    },
    default_cache_provider_id="my_custom_cache_v1" # If it should be the default
    # ...
)
```

By following the relevant protocol and registering your plugin, you can extend Genie Tooling to meet your specific application needs.
