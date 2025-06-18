# Plugin Architecture

Genie Tooling is built upon a highly modular and extensible plugin architecture. This design principle allows developers to easily swap, customize, or add new functionalities without altering the core framework.

## Core Idea

The central idea is that most significant functionalities within Genie are implemented as **plugins**. Each plugin adheres to a specific **protocol** (an interface defined using Python's `typing.Protocol` or an Abstract Base Class). This ensures that different implementations of a particular functionality can be used interchangeably as long as they conform to the defined contract.

## Key Components

1.  **`Plugin` Protocol (`genie_tooling.core.types.Plugin`)**:
    This is the base protocol that all plugins should ideally implement. It defines:
    *   `plugin_id` (property): A unique string identifier for the plugin.
    *   `async def setup(config)` (optional): For initialization with configuration.
    *   `async def teardown()` (optional): For resource cleanup.

2.  **`BootstrapPlugin` Protocol (`genie_tooling.bootstrap.abc.BootstrapPlugin`)**:
    A special type of plugin that runs once after the `Genie` facade is fully initialized. It's the primary mechanism for creating first-class extensions to the framework, allowing new interfaces to be attached directly to the `genie` object.

3.  **`PluginManager` (`genie_tooling.core.plugin_manager.PluginManager`)**:
    *   **Discovery**: Responsible for finding available plugin classes. It searches:
        *   **Entry Points**: Defined in `pyproject.toml` under the `[tool.poetry.plugins."genie_tooling.plugins"]` and `[tool.poetry.plugins."genie_tooling.bootstrap"]` groups. This is the standard way for third-party libraries or separate modules to provide plugins.
        *   **Plugin Development Directories**: Specified in `MiddlewareConfig.plugin_dev_dirs`. Useful for project-specific plugins or during development.
    *   **Instantiation & Setup**: When a plugin is requested (e.g., by a manager or the `Genie` facade), the `PluginManager` instantiates the plugin class and calls its `async setup(config)` method, passing any relevant configuration.
    *   **Caching**: It caches instantiated plugins to avoid re-creating them on every request.
    *   **Teardown**: Manages the `async teardown()` lifecycle of all loaded plugins.

4.  **Specialized Managers (e.g., `ToolManager`, `LLMProviderManager`, `RAGManager`)**:
    *   These components orchestrate plugins of a specific type.
    *   They use the `PluginManager` to get instances of the plugins they need based on configuration.
    *   For example, `LLMProviderManager` uses `PluginManager` to load the configured `LLMProviderPlugin` (like `OllamaLLMProviderPlugin`).

5.  **`Genie` Facade**:
    *   The `Genie` facade simplifies interaction with the underlying managers and, by extension, the plugins.
    *   When you configure `Genie` (e.g., `features.llm = "ollama"`), it internally directs the `LLMProviderManager` to load and use the "ollama" LLM provider plugin.

## Benefits of the Plugin Architecture

*   **Extensibility**: Easily add new LLM providers, tools, data sources, caching backends, or even entirely new agent capabilities as self-contained extensions.
*   **Customization**: Replace default implementations with your own specialized versions.
*   **Modularity**: Keeps different functionalities decoupled, making the system easier to understand, maintain, and test.
*   **Community Contributions**: Facilitates sharing and using plugins developed by the community.
*   **Flexibility**: Allows applications to select and configure only the components they need.

## How to Create a Plugin

1.  **Identify the Protocol**: Find the relevant plugin protocol in `genie_tooling` (e.g., `Tool` for a new tool, `CacheProvider` for a new cache).
2.  **Implement the Class**: Create a Python class that:
    *   Defines a unique `plugin_id` class attribute.
    *   Implements all methods and properties required by the chosen protocol.
    *   Optionally implements `async setup(config)` and `async teardown()`.
3.  **Register the Plugin**:
    *   For distributable plugins: Add an entry point in your `pyproject.toml` in the appropriate group.
    *   For local plugins: Place the file in a directory specified in `MiddlewareConfig.plugin_dev_dirs`.
4.  **Configure Genie**: Update your `MiddlewareConfig` to use your new plugin.

Refer to the specific "Creating ... Plugins" guides and the new [Extending Genie Tooling](extending_genie_tooling.md) guide for more detailed instructions.
