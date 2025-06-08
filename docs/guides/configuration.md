# Configuration

Genie Tooling is configured at runtime using a `MiddlewareConfig` object. For ease of use, especially for common setups, `MiddlewareConfig` integrates a `FeatureSettings` model.

## Simplified Configuration with `FeatureSettings`

The recommended way to start configuring Genie is by using the `features` attribute of `MiddlewareConfig`. `FeatureSettings` provides high-level toggles and default choices for major components like LLM providers, RAG components, caching, tool lookup, logging adapter, observability, HITL, token usage, guardrails, prompt system, conversation state, and distributed task queues.

```python
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.features import FeatureSettings

app_config = MiddlewareConfig(
    features=FeatureSettings(
        llm="ollama",
        command_processor="llm_assisted",
        # ... other features ...
    ),
    # Even in automatic mode, class-based tools often need to be enabled.
    # Tools requiring specific config must always be listed.
    tool_configurations={
        "sandboxed_fs_tool_v1": {"sandbox_base_path": "./my_agent_workspace"}
    }
)
```

### How `FeatureSettings` Works

When you initialize `Genie` with a `MiddlewareConfig` containing `FeatureSettings`, an internal `ConfigResolver` processes these settings. It translates your high-level choices into specific plugin IDs and default configurations for those plugins.

For a full list of available aliases and more details on simplified configuration, please see the [Simplified Configuration Guide](simplified_configuration.md).

## Explicit Overrides and Detailed Configuration

While `FeatureSettings` provides a convenient starting point, you can always provide more detailed, explicit configurations that will override or augment the settings derived from features.

### The `auto_enable_registered_tools` Flag

`MiddlewareConfig` includes a crucial flag for managing tool enablement:

*   **`auto_enable_registered_tools: bool`**
    *   **Default**: `True`
    *   **Behavior**:
        *   When `True`, any tool registered via `@tool` and `genie.register_tool_functions()` is **automatically enabled** and ready for use. This is convenient for development and rapid prototyping.
        *   When `False`, a tool is only active if its identifier is explicitly listed as a key in the `tool_configurations` dictionary.
    *   **Recommendation**: For production environments, it is **strongly recommended to set this to `False`** to maintain a clear, secure manifest of the agent's capabilities.

### Configuring Specific Plugins

You can provide specific configurations for individual plugins using the various `*_configurations` dictionaries in `MiddlewareConfig`. These dictionaries are keyed by the **canonical plugin ID** or a recognized alias.

```python
app_config = MiddlewareConfig(
    auto_enable_registered_tools=False, # Production-safe setting
    features=FeatureSettings(
        llm="openai",
        llm_openai_model_name="gpt-3.5-turbo", 
    ),
    # Override configuration for the OpenAI LLM provider
    llm_provider_configurations={
        "openai_llm_provider_v1": { # Canonical ID
            "model_name": "gpt-4-turbo-preview", 
            "request_timeout_seconds": 120
        }
    },
    # Explicitly enable and configure tools
    tool_configurations={
        "calculator_tool": {}, # Enable calculator
        "sandboxed_fs_tool_v1": {"sandbox_base_path": "./agent_workspace"}
    },
    # ... other specific configurations ...
)
```

**Key Points for Explicit Configuration:**

*   **Precedence**: Explicit configurations in `default_*_id` fields or within the `*_configurations` dictionaries take precedence over settings derived from `FeatureSettings`.
*   **Tool Enablement**: A tool plugin's availability is controlled by `auto_enable_registered_tools` and the `tool_configurations` dictionary.
*   **Canonical IDs vs. Aliases**: You can use either the canonical plugin ID (e.g., `"ollama_llm_provider_v1"`) or a recognized alias (e.g., `"ollama"`) as keys in configuration dictionaries.

## Plugin Development Directories

If you have custom plugins located outside your main Python path or not installed as entry points, you can specify their location:

```python
app_config = MiddlewareConfig(
    plugin_dev_dirs=["/path/to/my/custom_plugins", "./project_plugins"]
)
```
The `PluginManager` will scan these directories for valid plugin classes.
