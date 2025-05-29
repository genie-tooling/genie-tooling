"""Pytest fixtures and global test configuration for Genie Tooling."""
import logging
from typing import Any, Dict, Optional

import pytest
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.manager import ToolManager
from genie_tooling.core.types import Plugin as BasePluginForKP

@pytest.fixture
async def plugin_manager() -> PluginManager:
    pm = PluginManager(plugin_dev_dirs=[])
    await pm.discover_plugins()
    return pm

@pytest.fixture
async def mock_key_provider() -> KeyProvider:
    class MockKeyProviderImpl(KeyProvider, BasePluginForKP):
        _plugin_id_storage: str
        _description_storage: str

        def __init__(self, keys: Dict[str, str]):
            self.keys: Dict[str, str] = keys
            self._plugin_id_storage = "mock_key_provider_fixture_v1_from_conftest"
            self._description_storage = "Mock KeyProvider from conftest."

        @property
        def plugin_id(self) -> str:
            return self._plugin_id_storage

        @property
        def description(self) -> str: # Added for consistency, though not the cause of the error
            return self._description_storage

        async def get_key(self, key_name: str) -> Optional[str]:
            logging.debug(f"MockKeyProviderFixture: Requesting key '{key_name}'")
            return self.keys.get(key_name)

        async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
            logging.debug(f"{self.plugin_id} setup (mock).") # Accesses the property

        async def teardown(self) -> None:
            logging.debug(f"{self.plugin_id} teardown (mock).") # Accesses the property

    provider = MockKeyProviderImpl({
        "OPENWEATHERMAP_API_KEY": "test_owm_key_from_conftest_fixture",
        "OPENAI_API_KEY": "test_openai_key_from_conftest_fixture",
        "GOOGLE_API_KEY": "test_google_key_from_conftest_fixture"
    })
    await provider.setup()
    return provider

@pytest.fixture
async def tool_manager(plugin_manager: PluginManager) -> ToolManager:
    # plugin_manager fixture is async, so await it
    actual_plugin_manager = await plugin_manager
    tm = ToolManager(plugin_manager=actual_plugin_manager)
    await tm.initialize_tools(tool_configurations={})
    return tm
