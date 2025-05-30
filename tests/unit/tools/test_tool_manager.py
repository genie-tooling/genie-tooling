import logging
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.definition_formatters.abc import (
    DefinitionFormatter as DefinitionFormatterPlugin,
)
from genie_tooling.tools.abc import Tool as ToolPlugin
from genie_tooling.tools.manager import ToolManager


class MockTool(ToolPlugin):
    _identifier_value: str; _plugin_id_value: str
    def __init__(self, identifier_val: str, metadata: Dict[str, Any], execute_result: Any = "tool_executed", plugin_manager=None):
        self._identifier_value = identifier_val; self._plugin_id_value = identifier_val
        self._metadata = metadata; self._execute_result = execute_result
        self.setup_called_with_config: Optional[Dict[str, Any]] = None
        self.injected_plugin_manager = plugin_manager; self.teardown_called = False
    @property
    def identifier(self) -> str: return self._identifier_value
    @property
    def plugin_id(self) -> str: return self._plugin_id_value
    async def get_metadata(self) -> Dict[str, Any]:
        if "raise_in_get_metadata" in self._metadata: raise RuntimeError("Metadata retrieval failed")
        return self._metadata
    async def execute(self, params, key_provider, context=None) -> Any: return self._execute_result
    async def setup(self, config=None): self.setup_called_with_config = config
    async def teardown(self): self.teardown_called = True

class MockFormatter(DefinitionFormatterPlugin):
    _plugin_id_value: str
    def __init__(self):
        self._plugin_id_value = "mock_formatter_plugin_v1"
        self.formatter_id: str = "mock_format_v1"; self.description: str = "A mock definition formatter."; self.raise_in_format: bool = False
    @property
    def plugin_id(self) -> str: return self._plugin_id_value
    def format(self, tool_metadata: Dict[str, Any]) -> Any:
        if self.raise_in_format: raise RuntimeError("Formatter format failed")
        return {"formatted": True, "original_id": tool_metadata.get("identifier")}
    async def setup(self,config: Optional[Dict[str, Any]]=None): pass
    async def teardown(self): pass

@pytest.fixture
def mock_plugin_manager_fixture(mocker) -> PluginManager:
    pm = mocker.MagicMock(spec=PluginManager); pm.list_discovered_plugin_classes = MagicMock(return_value={}); pm.get_plugin_instance = AsyncMock(return_value=None); pm.get_plugin_source = MagicMock(return_value="mock_source"); pm._discovered_plugin_classes = {}; pm.discover_plugins = AsyncMock()
    return pm

@pytest.fixture
async def initialized_tool_manager(mock_plugin_manager_fixture: PluginManager) -> ToolManager:
    actual_mock_pm: PluginManager = mock_plugin_manager_fixture
    tool1_meta = {"identifier": "tool1", "name": "Tool One", "description_llm": "Desc1"}
    tool2_meta = {"identifier": "tool2", "name": "Tool Two", "description_llm": "Desc2"}
    actual_mock_pm.list_discovered_plugin_classes.return_value = {"tool1_plugin_id_from_discovery": MockTool, "tool2_plugin_id_from_discovery": MockTool, MockFormatter.plugin_id: MockFormatter}
    actual_mock_pm._discovered_plugin_classes = actual_mock_pm.list_discovered_plugin_classes.return_value
    async def get_instance_side_effect(plugin_id: str, config: Optional[Dict[str, Any]] = None, **kwargs):
        init_kwargs = kwargs
        if plugin_id == "tool1_plugin_id_from_discovery":
            tool = MockTool(identifier_val="tool1", metadata=tool1_meta, **init_kwargs)
            await tool.setup(config=config); return tool
        if plugin_id == "tool2_plugin_id_from_discovery":
            tool = MockTool(identifier_val="tool2", metadata=tool2_meta, **init_kwargs)
            await tool.setup(config=config); return tool
        if plugin_id == MockFormatter.plugin_id:
            formatter = MockFormatter(); await formatter.setup(config=config); return formatter
        return None
    actual_mock_pm.get_plugin_instance.side_effect = get_instance_side_effect
    tm = ToolManager(plugin_manager=actual_mock_pm)
    await tm.initialize_tools(tool_configurations={"tool1_plugin_id_from_discovery": {"setup_key": "setup_val1"}, "tool2_plugin_id_from_discovery": {"setup_key": "setup_val2"}})
    return tm

@pytest.mark.asyncio
async def test_tool_manager_initializes_discovered_tools(initialized_tool_manager: ToolManager):
    tm = await initialized_tool_manager
    tools: List[ToolPlugin] = await tm.list_tools()
    assert len(tools) == 2, f"Expected 2 tools, got {len(tools)}. Tools: {[t.identifier for t in tools]}"
    tool1_instance = await tm.get_tool("tool1")
    assert tool1_instance is not None
    assert tool1_instance.setup_called_with_config == {"setup_key": "setup_val1"} # type: ignore
    if isinstance(tool1_instance, MockTool):
        assert tool1_instance.injected_plugin_manager is not None

@patch("importlib.metadata.entry_points", return_value=MagicMock(select=MagicMock(return_value=[])))
@pytest.mark.asyncio
async def test_tool_manager_init_handles_duplicate_identifier(mock_eps, mock_plugin_manager_fixture: PluginManager, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING)
    pm = mock_plugin_manager_fixture
    tool_meta = {"identifier": "duplicate_id", "name": "Duplicate Tool"}
    class MockToolDupA(MockTool): pass
    class MockToolDupB(MockTool): pass
    pm.list_discovered_plugin_classes.return_value = {"plugin_a": MockToolDupA, "plugin_b": MockToolDupB}
    pm._discovered_plugin_classes = pm.list_discovered_plugin_classes.return_value
    async def get_instance_side_effect_dup(plugin_id: str, config=None, **kwargs):
        return MockTool(identifier_val="duplicate_id", metadata=tool_meta, plugin_manager=pm)
    pm.get_plugin_instance.side_effect = get_instance_side_effect_dup
    tm = ToolManager(plugin_manager=pm)
    await tm.initialize_tools()
    tools = await tm.list_tools()
    assert len(tools) == 1 , f"Expected 1 tool after duplicate, got {len(tools)}"
    assert tools[0].identifier == "duplicate_id"
    assert "Duplicate tool identifier 'duplicate_id' encountered" in caplog.text
    assert "Overwriting previous tool with same identifier." in caplog.text


# --- Tests for ToolManager with FunctionToolWrapper ---
# Assuming FunctionToolWrapper and @tool are accessible for testing
# These might need to be imported from their actual locations (e.g., genie_tooling.genie, genie_tooling.decorators)
from genie_tooling.decorators import tool
from genie_tooling.genie import FunctionToolWrapper


@tool
def sample_func_for_tm_test(data: str) -> str:
    """A sample function for ToolManager testing."""
    return data.upper()

@pytest.mark.asyncio
async def test_tool_manager_handles_function_tool_wrapper(mock_plugin_manager_fixture: PluginManager):
    pm = mock_plugin_manager_fixture
    tm = ToolManager(plugin_manager=pm)

    # Manually create a FunctionToolWrapper instance for testing
    # This simulates what Genie.register_tool_functions would do

    # The @tool decorator attaches metadata to the *wrapped* function it returns.
    # The original function is stored as _original_function_ on that wrapper.
    decorated_func = sample_func_for_tm_test
    metadata = getattr(decorated_func, "_tool_metadata_", None)
    assert metadata is not None, "Tool metadata not found on decorated function"

    original_func_to_call = getattr(decorated_func, "_original_function_", decorated_func)
    assert callable(original_func_to_call), "Original function not found or not callable"

    func_tool_wrapper = FunctionToolWrapper(original_func_to_call, metadata)

    # Add it to the ToolManager's internal store (simulating registration)
    tm._tools[func_tool_wrapper.identifier] = func_tool_wrapper

    # Test get_tool
    retrieved_tool = await tm.get_tool(func_tool_wrapper.identifier)
    assert retrieved_tool is func_tool_wrapper
    assert retrieved_tool.identifier == "sample_func_for_tm_test"

    # Test list_tools
    all_tools = await tm.list_tools()
    assert len(all_tools) == 1
    assert all_tools[0] is func_tool_wrapper

    # Test get_formatted_tool_definition (requires a mock formatter)
    mock_formatter_instance = MockFormatter() # Using existing mock from the file
    # Ensure get_plugin_instance is set up to return this formatter for its ID
    async def get_formatter_side_effect(plugin_id_req: str, config=None, **kwargs):
        if plugin_id_req == mock_formatter_instance.plugin_id:
            return mock_formatter_instance
        return None # Fallback for other plugin IDs
    pm.get_plugin_instance.side_effect = get_formatter_side_effect

    formatted_def = await tm.get_formatted_tool_definition(
        tool_identifier=func_tool_wrapper.identifier,
        formatter_id=mock_formatter_instance.plugin_id
    )
    assert formatted_def is not None
    assert formatted_def.get("formatted") is True
    assert formatted_def.get("original_id") == "sample_func_for_tm_test"

    pm.get_plugin_instance.assert_any_call(mock_formatter_instance.plugin_id)

# END_OF_EXISTING_TOOL_MANAGER_TESTS_MARKER
