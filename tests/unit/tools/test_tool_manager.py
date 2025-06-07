import logging
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.tools.abc import Tool as ToolPlugin
from genie_tooling.tools.manager import ToolManager

TOOL_MANAGER_LOGGER_NAME = "genie_tooling.tools.manager"

class MockTool(ToolPlugin):
    _identifier_value: str
    _plugin_id_value: str
    def __init__(self, identifier_val: str, metadata: Dict[str, Any], execute_result: Any = "tool_executed", plugin_manager=None):
        self._identifier_value = identifier_val
        self._plugin_id_value = identifier_val
        self._metadata = metadata
        self._execute_result = execute_result
        self.setup_called_with_config: Optional[Dict[str, Any]] = None
        self.injected_plugin_manager = plugin_manager
        self.teardown_called = False
    @property
    def identifier(self) -> str: return self._identifier_value
    @property
    def plugin_id(self) -> str: return self._plugin_id_value
    async def get_metadata(self) -> Dict[str, Any]:
        if "raise_in_get_metadata" in self._metadata:
            raise RuntimeError("Metadata retrieval failed")
        return self._metadata
    async def execute(self, params, key_provider, context=None) -> Any:
        return self._execute_result
    async def setup(self, config=None):
        self.setup_called_with_config = config
    async def teardown(self):
        self.teardown_called = True

@pytest.fixture
def mock_plugin_manager_fixture(mocker) -> PluginManager:
    pm = mocker.MagicMock(spec=PluginManager)
    pm.list_discovered_plugin_classes = MagicMock(return_value={})
    pm.get_plugin_instance = AsyncMock(return_value=None)
    pm.get_plugin_source = MagicMock(return_value="mock_source")
    pm._discovered_plugin_classes = {}
    pm.discover_plugins = AsyncMock()
    return pm

@pytest.mark.asyncio
async def test_initialize_tools_no_discovered_plugins(mock_plugin_manager_fixture: PluginManager, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger=TOOL_MANAGER_LOGGER_NAME)
    pm = mock_plugin_manager_fixture
    pm.list_discovered_plugin_classes.return_value = {}
    pm._discovered_plugin_classes = {}
    tm = ToolManager(plugin_manager=pm)
    await tm.initialize_tools(tool_configurations={})
    assert len(await tm.list_tools()) == 0
    assert "ToolManager initialized. Loaded 0 explicitly configured class-based tools." in caplog.text
