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
    )
)
```

### How `FeatureSettings` Works

When you initialize `Genie` with a `MiddlewareConfig` containing `FeatureSettings`, an internal `ConfigResolver` processes these settings. It translates your high-level choices into specific plugin IDs and default configurations for those plugins. For a full list of available aliases and more details on simplified configuration, please see the [Simplified Configuration Guide](simplified_configuration.md).

## Explicit Overrides and Detailed Configuration

While `FeatureSettings` provides a convenient starting point, you can always provide more detailed, explicit configurations that will override or augment the settings derived from features.

### Explicit Tool Enablement: `auto_enable_registered_tools` and `tool_configurations`

This is a critical security and configuration concept in Genie Tooling.

*   **`auto_enable_registered_tools: bool`** (Default: `True`)
    *   When `True` (for development), tools registered via the `@tool` decorator and `genie.register_tool_functions()` are **automatically enabled**. Class-based tools still generally need to be listed in `tool_configurations` to be enabled.
    *   When `False` (for production), a tool is **only** active if its identifier is explicitly listed as a key in the `tool_configurations` dictionary. This applies to both class-based plugins and `@tool` decorated functions.
    *   **Recommendation**: For production environments, it is **strongly recommended to set `auto_enable_registered_tools=False`** to maintain a clear, secure manifest of the agent's capabilities.

*   **`tool_configurations: Dict[str, Dict[str, Any]]`**
    *   This dictionary serves two purposes:
        1.  **Enabling Tools**: If a tool's `identifier` is a key in this dictionary, it is considered enabled.
        2.  **Configuring Tools**: The value associated with the key is a dictionary passed to the tool's `setup()` method. Use an empty dictionary `{}` to enable a tool that requires no configuration.

**Production Example:**
```python
app_config = MiddlewareConfig(
    auto_enable_registered_tools=False, # Production-safe setting
    features=FeatureSettings(
        llm="openai",
        llm_openai_model_name="gpt-4-turbo-preview",
    ),
    # Explicitly enable and configure tools
    tool_configurations={
        "calculator_tool": {}, # Enable calculator with no config
        "sandboxed_fs_tool_v1": {"sandbox_base_path": "./agent_workspace"},
        "my_registered_tool_function": {} # Enable a decorated tool
    },
    # ... other specific configurations ...
)
```

### Configuring Other Plugins

You can provide specific configurations for individual plugins using the various `*_configurations` dictionaries in `MiddlewareConfig`. These dictionaries are keyed by the **canonical plugin ID** or a recognized alias.

```python
app_config = MiddlewareConfig(
    features=FeatureSettings(llm="openai"),
    llm_provider_configurations={
        "openai_llm_provider_v1": { # Canonical ID
            "model_name": "gpt-4o", # Overrides model from features
            "request_timeout_seconds": 120
        }
    },
    # ...
)
```

## Plugin Development Directories

If you have custom plugins located outside your main Python path or not installed as entry points, you can specify their location:

```python
app_config = MiddlewareConfig(
    plugin_dev_dirs=["/path/to/my/custom_plugins", "./project_plugins"]
)
```
The `PluginManager` will scan these directories for valid plugin classes.
