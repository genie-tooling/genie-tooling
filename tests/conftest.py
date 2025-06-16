"""Pytest fixtures and global test configuration for Genie Tooling."""
import logging
import os
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest


# This prevents background threads from making network calls (for telemetry,
# safetensors conversion checks, etc.) that can outlive the test process.
# This is the most robust way to avoid hangs and "I/O operation on closed file"
# errors during pytest shutdown.
os.environ["HF_HUB_OFFLINE"] = "1"


from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import Plugin as BasePluginForKP
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.manager import ToolManager


@pytest.fixture()
async def plugin_manager() -> PluginManager:
    # This is a basic PluginManager, tests might need to mock its methods further
    # or use a more specialized fixture if discovery is part of the test.
    pm = PluginManager(plugin_dev_dirs=[])
    # await pm.discover_plugins() # Discovery might not be needed for all unit tests
    return pm


@pytest.fixture()
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
        def description(self) -> str:
            return self._description_storage

        async def get_key(self, key_name: str) -> Optional[str]:
            logging.debug(f"MockKeyProviderFixture: Requesting key '{key_name}'")
            return self.keys.get(key_name)

        async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
            logging.debug(f"{self.plugin_id} setup (mock).")

        async def teardown(self) -> None:
            logging.debug(f"{self.plugin_id} teardown (mock).")

    provider = MockKeyProviderImpl(
        {
            "OPENWEATHERMAP_API_KEY": "test_owm_key_from_conftest_fixture",
            "OPENAI_API_KEY": "test_openai_key_from_conftest_fixture",
            "GOOGLE_API_KEY": "test_google_key_from_conftest_fixture",
            "GOOGLE_CSE_ID": "test_google_cse_id_from_conftest_fixture",
            "TEST_KEY": "test_value_from_conftest_fixture",
            "LLAMA_CPP_API_KEY_TEST": "test_llama_cpp_api_key_from_conftest",
            "QDRANT_API_KEY_TEST": "test_qdrant_api_key_from_conftest",
        }
    )
    await provider.setup()
    return provider


@pytest.fixture()
async def tool_manager(plugin_manager: PluginManager) -> ToolManager:
    # This fixture provides a ToolManager that has already discovered tools.
    # For tests focusing on ToolManager's initialization, a different setup might be needed.
    # For now, assume plugin_manager is already populated or discovery is mocked.
    actual_plugin_manager = plugin_manager  # No await needed if plugin_manager is sync fixture
    tm = ToolManager(plugin_manager=actual_plugin_manager)
    # In a real scenario, initialize_tools would be called.
    # For unit tests of other components, we might mock tm.get_tool directly.
    # await tm.initialize_tools(tool_configurations={})
    return tm


# Add a generic mock plugin fixture that can be specialized
@pytest.fixture()
def generic_mock_plugin_instance(mocker) -> MagicMock:
    """Returns a generic MagicMock that can be used to simulate any plugin instance."""
    mock_instance = mocker.MagicMock(
        spec=BasePluginForKP
    )  # Use BasePluginForKP for common attributes
    mock_instance.plugin_id = "generic_mock_plugin_v1"
    mock_instance.setup = mocker.AsyncMock()
    mock_instance.teardown = mocker.AsyncMock()
    return mock_instance