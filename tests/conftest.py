"""Pytest fixtures and global test configuration for Genie Tooling."""
import logging
from typing import Any, Dict, Optional

import pytest
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.manager import ToolManager

# pytest-asyncio provides an event_loop fixture automatically.

@pytest.fixture
async def plugin_manager() -> PluginManager: # Changed to direct return
    """Provides a PluginManager instance, initialized and with plugins discovered."""
    pm = PluginManager(plugin_dev_dirs=[]) # No dev dirs for most unit tests
    await pm.discover_plugins() # Discover built-ins etc.
    # Teardown can be handled by a finalizer if specific instances need it,
    # or rely on Python's GC for simple cases if no external resources are held.
    # For tests creating many instances, explicit teardown in test or fixture finalizer is better.
    return pm


@pytest.fixture
async def mock_key_provider() -> KeyProvider:
    # Ensure it implements all methods from Protocol, including Plugin if KeyProvider is also a Plugin
    # Assuming KeyProvider is a Plugin for this example based on previous structure
    from genie_tooling.core.types import (
        Plugin as BasePluginForKP,  # Alias to avoid conflict
    )
    from genie_tooling.security.key_provider import KeyProvider

    class MockKeyProviderImpl(KeyProvider, BasePluginForKP):
        def __init__(self, keys: Dict[str, str]):
            self.keys = keys
            self.plugin_id = "mock_key_provider_fixture_v1_from_conftest"
            self.description = "Mock KeyProvider from conftest." # Added description

        async def get_key(self, key_name: str) -> Optional[str]:
            logging.debug(f"MockKeyProviderFixture: Requesting key '{key_name}'")
            return self.keys.get(key_name)

        async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
            logging.debug(f"{self.plugin_id} setup (mock).")

        async def teardown(self) -> None:
            logging.debug(f"{self.plugin_id} teardown (mock).")

    provider = MockKeyProviderImpl({
        "OPENWEATHERMAP_API_KEY": "test_owm_key_from_conftest_fixture",
        "OPENAI_API_KEY": "test_openai_key_from_conftest_fixture",
    })
    await provider.setup()
    return provider


@pytest.fixture
async def tool_manager(plugin_manager: PluginManager) -> ToolManager: # plugin_manager is already awaited
    from genie_tooling.tools.manager import ToolManager
    tm = ToolManager(plugin_manager=plugin_manager)
    # Initialize with empty configs; this will use discovered tools from plugin_manager
    await tm.initialize_tools(tool_configurations={})
    return tm
