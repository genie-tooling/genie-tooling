# Creating Plugins (Overview)

Genie Tooling is designed to be highly extensible through its plugin architecture. Almost every functional component can be replaced or augmented with custom implementations.

This page provides a general overview. For specific plugin types, refer to:
*   [Creating Tool Plugins](creating_tool_plugins.md)
*   [Creating RAG Plugins](creating_rag_plugins.md)
*   [Creating Other Plugins](creating_other_plugins.md) (for caches, LLM providers, etc.)

## Core Plugin Protocol

All plugins in Genie Tooling should ideally adhere to the `genie_tooling.core.types.Plugin` protocol:

```python
from typing import Protocol, Optional, Dict, Any

class Plugin(Protocol):
    @property
    def plugin_id(self) -> str:
        """A unique identifier for this plugin instance/type."""
        ...

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Optional asynchronous setup method.
        Called after the plugin is instantiated, with its specific configuration.
        """
        pass # Default implementation

    async def teardown(self) -> None:
        """
        Optional asynchronous teardown method.
        Called when the Genie facade is closing down, to release resources.
        """
        pass # Default implementation
```

**Key Requirements for a Plugin Class:**

1.  **`plugin_id` (Class Attribute)**:
    *   A **unique string identifier** for your plugin.
    *   Convention: `your_plugin_name_v1` (e.g., `my_custom_llm_provider_v1`, `advanced_calculator_tool_v1`).
    *   This ID is used in configuration files and for discovery.

2.  **Implement the Specific Protocol**:
    *   Your plugin class must implement the methods defined by the protocol for its type (e.g., `Tool`, `LLMProviderPlugin`, `CacheProviderPlugin`).
    *   These protocols are typically found in the `abc.py` file of the relevant submodule (e.g., `genie_tooling.tools.abc.Tool`).

3.  **`async def setup(self, config: Optional[Dict[str, Any]] = None)` (Optional)**:
    *   If your plugin needs initialization with configuration values, implement this asynchronous method.
    *   The `config` dictionary will contain the settings provided for your plugin's `plugin_id` in the `MiddlewareConfig` (e.g., in `llm_provider_configurations["my_custom_llm_v1"]`).

4.  **`async def teardown(self)` (Optional)**:
    *   If your plugin acquires resources (e.g., network connections, file handles) that need to be released, implement this asynchronous method.

## Plugin Discovery

Genie's `PluginManager` discovers plugins through two primary mechanisms:

1.  **Entry Points (Recommended for distributable plugins)**:
    *   Define an entry point in your package's `pyproject.toml` under the group `[tool.poetry.plugins."genie_tooling.plugins"]`.
    *   Example:
        ```toml
        [tool.poetry.plugins."genie_tooling.plugins"]
        "my_awesome_tool_v1" = "my_package.my_module:MyAwesomeToolClass"
        "custom_cache_provider_alpha" = "my_other_package.cache:CustomCache"
        ```
    *   The key is the `plugin_id` Genie will use to refer to your plugin.
    *   The value is the import path to your plugin class.

2.  **Plugin Development Directories (For local/project-specific plugins)**:
    *   Specify a list of directories in `MiddlewareConfig.plugin_dev_dirs`.
    *   The `PluginManager` will scan these directories for Python files (`*.py`).
    *   It will attempt to import these files as modules and look for classes that implement the `Plugin` protocol and have a `plugin_id`.
    *   Files starting with `_` or `.` (e.g., `__init__.py`, `_internal_utils.py`) are ignored.

## Configuration and Usage

Once your plugin is discoverable, you can:
*   **Configure it**: Provide settings in the appropriate `*_configurations` dictionary within `MiddlewareConfig`, keyed by your plugin's `plugin_id`.
*   **Set it as a default**: If applicable, set its `plugin_id` in fields like `default_llm_provider_id`, `default_cache_provider_id`, etc., in `MiddlewareConfig` or via `FeatureSettings`.
*   **Use it explicitly**: Some `Genie` facade methods allow specifying a `plugin_id` at runtime (e.g., `genie.llm.chat(..., provider_id="my_llm_v1")`).

By adhering to these principles, you can create robust and reusable extensions for Genie Tooling.
