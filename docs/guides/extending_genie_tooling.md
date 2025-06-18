# Extending Genie Tooling: A Developer's Guide

This guide provides the official "blessed path" for creating robust, maintainable extensions for the `genie-tooling` ecosystem.

#### **Core Philosophy: The Bootstrap Plugin**

An extension is a self-contained package that enhances the `Genie` facade with new capabilities. It achieves this by providing a **Bootstrap Plugin** that automatically discovers and initializes the extension's components when the application starts.

This approach keeps your extension logic decoupled from the core framework while allowing it to cleanly attach its public interface to the `genie` object (e.g., `genie.my_extension`).

---

#### **Step 1: Project Setup (`pyproject.toml`)**

Your extension is a standard Python package that depends on `genie-tooling`. The magic happens in the `[tool.poetry.plugins]` section.

1.  **Bootstrap Plugin Entry Point:** This is the most important part. It registers your main extension setup class with Genie's bootstrap discovery mechanism.
    ```toml
    [tool.poetry.plugins."genie_tooling.bootstrap"]
    "my_extension_bootstrap_v1" = "my_extension.bootstrap:MyExtensionBootstrapPlugin"
    ```

2.  **Standard Plugin Entry Points:** If your extension also provides standard plugin types (Tools, Command Processors, LLM Providers, etc.), register them in the standard `plugins` group. The core framework's `PluginManager` will discover them automatically.
    ```toml
    [tool.poetry.plugins."genie_tooling.plugins"]
    "my_custom_tool_v1" = "my_extension.tools:my_custom_tool_function"
    "my_custom_dispatcher_v1" = "my_extension.dispatchers:MyCustomDispatcher"
    ```

#### **Step 2: The Bootstrap Plugin**

The Bootstrap Plugin is a class that implements the `genie_tooling.bootstrap.BootstrapPlugin` protocol. Its single `bootstrap` method is called at the end of `Genie.create()`, giving it access to the fully initialized framework.

```python
# my_extension/bootstrap.py
from genie_tooling.bootstrap import BootstrapPlugin
from genie_tooling.genie import Genie
from .manager import MyExtensionManager  # Your internal logic
from .interface import MyExtensionInterface # Your public API

class MyExtensionBootstrapPlugin(BootstrapPlugin):
    plugin_id: str = "my_extension_bootstrap_v1"
    description: str = "Initializes and attaches the MyExtension feature to the Genie facade."

    async def bootstrap(self, genie: Genie) -> None:
        # 1. Access Extension-Specific Configuration
        # Namespace your extension's config to avoid conflicts.
        # This part of the config model is not yet standardized, but this is the recommended pattern.
        # For now, you might need to pass it into your manager's setup.
        # extension_configs = genie.config.extension_configurations.get("my_extension", {})
        extension_configs = {} # Placeholder

        # 2. Access Shared Core Services from the Genie Facade
        # Use public interfaces like genie.llm, genie.rag.
        # Avoid accessing private managers like `genie._rag_manager`.
        llm_interface = genie.llm
        rag_interface = genie.rag

        # 3. Initialize Your Extension's Core Logic (Manager)
        # Pass dependencies explicitly. This makes your manager easy to unit test.
        manager = MyExtensionManager(
            config=extension_configs,
            llm_interface=llm_interface,
            rag_interface=rag_interface,
            plugin_manager=genie._plugin_manager # Pass the PM for discovering your own internal plugins
        )
        await manager.setup()

        # 4. Create the Public Interface for your Extension
        interface = MyExtensionInterface(manager=manager)

        # 5. Attach the Interface to the Genie Facade
        # This is the "magic" that makes `genie.my_extension` available to the user.
        setattr(genie, "my_extension", interface)
        print("MyExtension has been successfully bootstrapped and attached to Genie.")
```

#### **Step 3: The Manager and Interface**

*   **Manager (`MyExtensionManager`)**: This class contains the internal orchestration logic for your extension. It should take its dependencies (like `llm_interface`) in its `__init__`, making it independent of the `Genie` object and easy to test in isolation.
*   **Interface (`MyExtensionInterface`)**: This is the clean, public-facing API you expose to the end developer. It holds a reference to your manager and delegates calls to it (e.g., a call to `genie.my_extension.summarize(...)` would call `manager.summarize(...)`).

#### **Step 4: Providing Your Own Internal Plugins**

Your extension can provide its own set of internal plugins (e.g., custom `CommandProcessorPlugin`s or `Tool`s). Your `MyExtensionManager` can use the `PluginManager` instance passed to it during initialization to discover and load these internal plugins as needed. This follows the same robust pattern as the core framework, ensuring your extension is itself modular.

#### **Step 5: Testing Your Extension**

Rigorous testing is non-negotiable.

1.  **Unit Tests:**
    *   Test your tools and other components in isolation. Use `pytest.fixture` to provide mocked dependencies (e.g., a mock `genie.llm` interface).
    *   Test your `MyExtensionManager` by passing it mocked interfaces and plugin managers. Verify it calls them correctly.

2.  **Integration Tests:**
    *   Test the `MyExtensionBootstrapPlugin`'s `bootstrap` method. Create a mock `Genie` object and assert that `setattr` is called correctly on it. This verifies the handshake between your extension and the core framework.

3.  **End-to-End (E2E) Tests:**
    *   This is the final specification. Create a test that initializes the *real* `Genie` framework with your extension installed in the environment.
    *   Provide a `MiddlewareConfig` that enables the necessary core features (like an LLM provider).
    *   Call `genie = await Genie.create(config=...)`.
    *   Assert that `hasattr(genie, "my_extension")` is `True`.
    *   Call your extension's public methods (e.g., `await genie.my_extension.do_something()`) and verify the output. This proves that the entire discovery, bootstrap, and execution pipeline works with live components.
