### tests/unit/tools/test_tool_manager.py
import logging
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import Plugin  # For NotATool
from genie_tooling.definition_formatters.abc import (
    DefinitionFormatter as DefinitionFormatterPlugin,
)
from genie_tooling.tools.abc import Tool as ToolPlugin
from genie_tooling.tools.manager import ToolManager

# Logger name for the module under test
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

class MockFormatter(DefinitionFormatterPlugin):
    _plugin_id_value: str
    def __init__(self):
        self._plugin_id_value = "mock_formatter_plugin_v1"
        self.formatter_id: str = "mock_format_v1"
        self.description: str = "A mock definition formatter."
        self.raise_in_format: bool = False
    @property
    def plugin_id(self) -> str: return self._plugin_id_value
    def format(self, tool_metadata: Dict[str, Any]) -> Any:
        if self.raise_in_format:
            raise RuntimeError("Formatter format failed")
        return {"formatted": True, "original_id": tool_metadata.get("identifier")}
    async def setup(self,config: Optional[Dict[str, Any]]=None):
        pass
    async def teardown(self):
        pass

class NotATool(Plugin): # Does not implement ToolPlugin
    plugin_id: str = "not_a_tool_v1"
    description: str = "This is not a tool."
    async def setup(self, config=None): pass
    async def teardown(self): pass

class SetupFailsTool(MockTool):
    async def setup(self, config=None):
        await super().setup(config)
        raise RuntimeError("Tool setup failed deliberately")


@pytest.fixture
def mock_plugin_manager_fixture(mocker) -> PluginManager:
    pm = mocker.MagicMock(spec=PluginManager)
    pm.list_discovered_plugin_classes = MagicMock(return_value={})
    pm.get_plugin_instance = AsyncMock(return_value=None)
    pm.get_plugin_source = MagicMock(return_value="mock_source")
    pm._discovered_plugin_classes = {}
    pm.discover_plugins = AsyncMock()
    return pm

@pytest.fixture
async def initialized_tool_manager(mock_plugin_manager_fixture: PluginManager) -> ToolManager:
    actual_mock_pm: PluginManager = mock_plugin_manager_fixture
    tool1_meta = {"identifier": "tool1", "name": "Tool One", "description_llm": "Desc1"}
    tool2_meta = {"identifier": "tool2", "name": "Tool Two", "description_llm": "Desc2"}
    # Define plugin IDs that will be used as keys in tool_configurations
    tool1_plugin_id_key = "tool1_plugin_id_from_discovery"
    tool2_plugin_id_key = "tool2_plugin_id_from_discovery"

    actual_mock_pm.list_discovered_plugin_classes.return_value = {
        tool1_plugin_id_key: MockTool,
        tool2_plugin_id_key: MockTool,
        MockFormatter.plugin_id: MockFormatter
    }
    actual_mock_pm._discovered_plugin_classes = actual_mock_pm.list_discovered_plugin_classes.return_value

    async def get_instance_side_effect(plugin_id: str, config: Optional[Dict[str, Any]] = None, **kwargs):
        init_kwargs = kwargs
        if plugin_id == tool1_plugin_id_key:
            tool = MockTool(identifier_val="tool1", metadata=tool1_meta, **init_kwargs)
            await tool.setup(config=config)
            return tool
        if plugin_id == tool2_plugin_id_key:
            tool = MockTool(identifier_val="tool2", metadata=tool2_meta, **init_kwargs)
            await tool.setup(config=config)
            return tool
        if plugin_id == MockFormatter.plugin_id: # Default behavior for the formatter
            formatter = MockFormatter()
            await formatter.setup(config=config)
            return formatter
        return None
    actual_mock_pm.get_plugin_instance.side_effect = get_instance_side_effect

    tm = ToolManager(plugin_manager=actual_mock_pm)
    # Initialize with tool_configurations that match the keys used by get_instance_side_effect
    await tm.initialize_tools(tool_configurations={
        tool1_plugin_id_key: {"setup_key": "setup_val1"},
        tool2_plugin_id_key: {"setup_key": "setup_val2"}
    })
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
    caplog.set_level(logging.WARNING, logger=TOOL_MANAGER_LOGGER_NAME)
    pm = mock_plugin_manager_fixture
    tool_meta = {"identifier": "duplicate_id", "name": "Duplicate Tool"}

    # Define unique plugin IDs for discovery
    plugin_a_id = "plugin_a_for_dup_test"
    plugin_b_id = "plugin_b_for_dup_test"

    class MockToolDupA(MockTool):
        pass # Will be instantiated with identifier="duplicate_id"
    class MockToolDupB(MockTool):
        pass # Will also be instantiated with identifier="duplicate_id"

    pm.list_discovered_plugin_classes.return_value = {
        plugin_a_id: MockToolDupA,
        plugin_b_id: MockToolDupB
    }
    pm._discovered_plugin_classes = pm.list_discovered_plugin_classes.return_value

    async def get_instance_side_effect_dup(plugin_id: str, config=None, **kwargs):
        # Both plugin IDs will result in a tool with the same identifier
        return MockTool(identifier_val="duplicate_id", metadata=tool_meta, plugin_manager=pm)
    pm.get_plugin_instance.side_effect = get_instance_side_effect_dup

    tm = ToolManager(plugin_manager=pm)
    # Explicitly configure both plugin IDs to be loaded
    await tm.initialize_tools(tool_configurations={
        plugin_a_id: {},
        plugin_b_id: {} # This one should overwrite the first due to same identifier
    })

    tools = await tm.list_tools()
    assert len(tools) == 1 , f"Expected 1 tool after duplicate, got {len(tools)}"
    assert tools[0].identifier == "duplicate_id"
    assert "Duplicate tool identifier 'duplicate_id' encountered" in caplog.text
    assert "Overwriting previous tool with same identifier." in caplog.text


# --- Tests for ToolManager with FunctionToolWrapper ---
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
    # Note: FunctionToolWrappers are typically added via genie.register_tool_functions,
    # which directly manipulates tm._tools. For this unit test, we simulate that.
    decorated_func = sample_func_for_tm_test
    metadata = getattr(decorated_func, "_tool_metadata_", None)
    assert metadata is not None, "Tool metadata not found on decorated function"
    original_func_to_call = getattr(decorated_func, "_original_function_", decorated_func)
    assert callable(original_func_to_call), "Original function not found or not callable"

    func_tool_wrapper = FunctionToolWrapper(original_func_to_call, metadata)
    # Manually add to the internal store for this test, as initialize_tools won't pick it up
    # unless it's configured via tool_configurations (which is not how @tool is typically used with ToolManager directly)
    tm._tools[func_tool_wrapper.identifier] = func_tool_wrapper

    retrieved_tool = await tm.get_tool(func_tool_wrapper.identifier)
    assert retrieved_tool is func_tool_wrapper
    assert retrieved_tool.identifier == "sample_func_for_tm_test"

    all_tools = await tm.list_tools()
    assert len(all_tools) == 1
    assert all_tools[0] is func_tool_wrapper

    mock_formatter_instance = MockFormatter()
    async def get_formatter_side_effect(plugin_id_req: str, config=None, **kwargs):
        if plugin_id_req == mock_formatter_instance.plugin_id:
            return mock_formatter_instance
        return None
    pm.get_plugin_instance.side_effect = get_formatter_side_effect

    formatted_def = await tm.get_formatted_tool_definition(
        tool_identifier=func_tool_wrapper.identifier,
        formatter_id=mock_formatter_instance.plugin_id
    )
    assert formatted_def is not None
    assert formatted_def.get("formatted") is True
    assert formatted_def.get("original_id") == "sample_func_for_tm_test"
    pm.get_plugin_instance.assert_any_call(mock_formatter_instance.plugin_id)

# --- Additional Tests for ToolManager ---

@pytest.mark.asyncio
async def test_initialize_tools_no_discovered_plugins(mock_plugin_manager_fixture: PluginManager, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger=TOOL_MANAGER_LOGGER_NAME)
    pm = mock_plugin_manager_fixture
    pm.list_discovered_plugin_classes.return_value = {}
    pm._discovered_plugin_classes = {}
    tm = ToolManager(plugin_manager=pm)
    await tm.initialize_tools(tool_configurations={}) # Pass empty config
    assert len(await tm.list_tools()) == 0
    assert "ToolManager initialized. Loaded 0 explicitly configured tools." in caplog.text

@pytest.mark.asyncio
async def test_initialize_tools_skips_non_tool_plugins(mock_plugin_manager_fixture: PluginManager, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG, logger=TOOL_MANAGER_LOGGER_NAME)
    pm = mock_plugin_manager_fixture
    tool_a_plugin_id = "tool_a_plugin_id"
    not_a_tool_plugin_id = "not_a_tool_plugin_id"

    pm.list_discovered_plugin_classes.return_value = {
        tool_a_plugin_id: MockTool,
        not_a_tool_plugin_id: NotATool
    }
    pm._discovered_plugin_classes = pm.list_discovered_plugin_classes.return_value

    async def get_instance_side_effect(plugin_id: str, config=None, **kwargs):
        if plugin_id == tool_a_plugin_id:
            tool = MockTool("tool_a_id", {"identifier": "tool_a_id"}, plugin_manager=pm)
            await tool.setup(config)
            return tool
        if plugin_id == not_a_tool_plugin_id:
            instance = NotATool() # Not a ToolPlugin
            # await instance.setup(config) # NotATool doesn't have async setup
            return instance
        return None
    pm.get_plugin_instance.side_effect = get_instance_side_effect

    tm = ToolManager(plugin_manager=pm)
    # Explicitly configure both, ToolManager should only load MockTool
    await tm.initialize_tools(tool_configurations={
        tool_a_plugin_id: {},
        not_a_tool_plugin_id: {} # This will be attempted but skipped
    })
    tools = await tm.list_tools()
    assert len(tools) == 1
    assert tools[0].identifier == "tool_a_id"
    assert f"Plugin '{not_a_tool_plugin_id}' (class NotATool) instantiated but is not a Tool." in caplog.text

@pytest.mark.asyncio
async def test_initialize_tools_get_plugin_instance_returns_none(mock_plugin_manager_fixture: PluginManager, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING, logger=TOOL_MANAGER_LOGGER_NAME)
    pm = mock_plugin_manager_fixture
    tool_b_plugin_id = "tool_b_plugin_id_for_none_test"
    pm.list_discovered_plugin_classes.return_value = {tool_b_plugin_id: MockTool}
    pm._discovered_plugin_classes = pm.list_discovered_plugin_classes.return_value
    pm.get_plugin_instance.return_value = None # Simulate get_plugin_instance failing

    tm = ToolManager(plugin_manager=pm)
    await tm.initialize_tools(tool_configurations={tool_b_plugin_id: {}}) # Configure it
    assert len(await tm.list_tools()) == 0
    assert f"Plugin '{tool_b_plugin_id}' (class MockTool) did not yield a valid instance or failed setup." in caplog.text

@pytest.mark.asyncio
async def test_initialize_tools_tool_setup_fails(mock_plugin_manager_fixture: PluginManager, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=TOOL_MANAGER_LOGGER_NAME)
    pm = mock_plugin_manager_fixture
    setup_fail_tool_plugin_id = "setup_fail_tool_plugin_id_test"
    pm.list_discovered_plugin_classes.return_value = {setup_fail_tool_plugin_id: SetupFailsTool}
    pm._discovered_plugin_classes = pm.list_discovered_plugin_classes.return_value
    # Simulate that get_plugin_instance itself raises because SetupFailsTool.setup raises
    async def get_instance_setup_fail_side_effect(plugin_id: str, config=None, **kwargs):
        if plugin_id == setup_fail_tool_plugin_id:
            tool = SetupFailsTool(setup_fail_tool_plugin_id, {}) # identifier, metadata
            await tool.setup(config) # This will raise "Tool setup failed deliberately"
            return tool # Should not be reached
        return None
    pm.get_plugin_instance.side_effect = get_instance_setup_fail_side_effect

    tm = ToolManager(plugin_manager=pm)
    await tm.initialize_tools(tool_configurations={setup_fail_tool_plugin_id: {}}) # Configure it
    assert len(await tm.list_tools()) == 0
    assert f"Error initializing tool plugin from ID/alias '{setup_fail_tool_plugin_id}' (class SetupFailsTool): Tool setup failed deliberately" in caplog.text


@pytest.mark.asyncio
async def test_list_available_formatters_no_formatters(mock_plugin_manager_fixture: PluginManager):
    pm = mock_plugin_manager_fixture
    pm.list_discovered_plugin_classes.return_value = {"tool_a": MockTool} # Only a tool, no formatters
    pm._discovered_plugin_classes = pm.list_discovered_plugin_classes.return_value
    tm = ToolManager(plugin_manager=pm)
    formatters = await tm.list_available_formatters()
    assert len(formatters) == 0

@pytest.mark.asyncio
async def test_list_available_formatters_skips_non_formatter(mock_plugin_manager_fixture: PluginManager, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG, logger=TOOL_MANAGER_LOGGER_NAME)
    pm = mock_plugin_manager_fixture
    formatter1_id = "formatter1_plugin_id"
    not_a_formatter_id = "not_a_formatter_plugin_id"

    pm.list_discovered_plugin_classes.return_value = {
        formatter1_id: MockFormatter,
        not_a_formatter_id: NotATool
    }
    pm._discovered_plugin_classes = pm.list_discovered_plugin_classes.return_value

    async def get_instance_side_effect(plugin_id: str, config=None, **kwargs):
        if plugin_id == formatter1_id:
            return MockFormatter()
        if plugin_id == not_a_formatter_id:
            return NotATool() # Instantiates, but not a DefinitionFormatter
        return None
    pm.get_plugin_instance.side_effect = get_instance_side_effect

    tm = ToolManager(plugin_manager=pm)
    formatters = await tm.list_available_formatters()
    assert len(formatters) == 1
    assert formatters[0]["id"] == "mock_format_v1"
    # No error/warning should be logged for skipping NotATool, just debug if anything.

@pytest.mark.asyncio
async def test_list_available_formatters_get_instance_fails(mock_plugin_manager_fixture: PluginManager, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG, logger=TOOL_MANAGER_LOGGER_NAME)
    pm = mock_plugin_manager_fixture
    formatter_plugin_id_for_fail = "formatter_plugin_id_for_fail_test"
    pm.list_discovered_plugin_classes.return_value = {formatter_plugin_id_for_fail: MockFormatter}
    pm._discovered_plugin_classes = pm.list_discovered_plugin_classes.return_value
    pm.get_plugin_instance.side_effect = RuntimeError("Failed to get instance")

    tm = ToolManager(plugin_manager=pm)
    formatters = await tm.list_available_formatters()
    assert len(formatters) == 0
    assert f"Could not instantiate or check plugin '{formatter_plugin_id_for_fail}' as DefinitionFormatter: Failed to get instance" in caplog.text


@pytest.mark.asyncio
async def test_get_tool_not_found(initialized_tool_manager: ToolManager, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG, logger=TOOL_MANAGER_LOGGER_NAME)
    tm = await initialized_tool_manager
    tool = await tm.get_tool("non_existent_tool")
    assert tool is None
    assert "Tool with identifier 'non_existent_tool' not found in ToolManager (not explicitly configured or loaded)." in caplog.text

@pytest.mark.asyncio
async def test_list_tool_summaries_no_tools(mock_plugin_manager_fixture: PluginManager):
    pm = mock_plugin_manager_fixture
    tm = ToolManager(plugin_manager=pm)
    await tm.initialize_tools(tool_configurations={}) # Ensure initialized with no tools
    summaries, meta = await tm.list_tool_summaries()
    assert len(summaries) == 0
    assert meta["total_items"] == 0

@pytest.mark.asyncio
async def test_list_tool_summaries_get_metadata_fails(initialized_tool_manager: ToolManager, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=TOOL_MANAGER_LOGGER_NAME)
    tm = await initialized_tool_manager # This fixture loads tool1 and tool2
    tool1 = await tm.get_tool("tool1")
    assert tool1 is not None
    tool1._metadata["raise_in_get_metadata"] = True # Make tool1 fail metadata retrieval

    summaries, _ = await tm.list_tool_summaries()
    assert len(summaries) == 1 # Only tool2 should be summarized
    assert summaries[0]["identifier"] == "tool2"
    assert "Error getting metadata for tool 'tool1': Metadata retrieval failed" in caplog.text

@pytest.mark.asyncio
async def test_list_tool_summaries_pagination(initialized_tool_manager: ToolManager):
    tm = await initialized_tool_manager # Loads tool1, tool2
    # Add more tools manually to tm._tools for this test, as initialize_tools won't pick them up
    # without tool_configurations entries.
    tool3 = MockTool("tool3", {"identifier": "tool3", "name": "Tool Three"}, plugin_manager=tm._plugin_manager)
    tool4 = MockTool("tool4", {"identifier": "tool4", "name": "Tool Four"}, plugin_manager=tm._plugin_manager)
    tm._tools["tool3"] = tool3
    tm._tools["tool4"] = tool4

    summaries, meta = await tm.list_tool_summaries(pagination_params={"page_size": 0})
    assert meta["page_size"] == 20
    summaries, meta = await tm.list_tool_summaries(pagination_params={"page_size": "invalid"})
    assert meta["page_size"] == 20

    summaries, meta = await tm.list_tool_summaries(pagination_params={"page": 1, "page_size": 2})
    assert len(summaries) == 2
    assert meta["current_page"] == 1
    assert meta["total_pages"] == 2
    assert meta["has_next"] is True
    assert meta["has_prev"] is False

    summaries, meta = await tm.list_tool_summaries(pagination_params={"page": 2, "page_size": 2})
    assert len(summaries) == 2
    assert meta["current_page"] == 2
    assert meta["total_pages"] == 2
    assert meta["has_next"] is False
    assert meta["has_prev"] is True

    summaries, meta = await tm.list_tool_summaries(pagination_params={"page": 3, "page_size": 2})
    assert len(summaries) == 0


@pytest.mark.asyncio
async def test_get_formatted_tool_definition_tool_not_found(initialized_tool_manager: ToolManager):
    tm = await initialized_tool_manager
    formatted_def = await tm.get_formatted_tool_definition("non_existent", "mock_format_v1")
    assert formatted_def is None

@pytest.mark.asyncio
async def test_get_formatted_tool_definition_formatter_not_found(initialized_tool_manager: ToolManager, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING, logger=TOOL_MANAGER_LOGGER_NAME)
    tm = await initialized_tool_manager
    # Modify the side_effect of the PM instance used by tm
    original_side_effect = tm._plugin_manager.get_plugin_instance.side_effect
    async def side_effect_formatter_none(plugin_id: str, config=None, **kwargs):
        if plugin_id == "unknown_formatter_id": # The one we request
            return None
        # Call original for other plugins like tools or default formatter if needed by other parts of the test
        if callable(original_side_effect):
             return await original_side_effect(plugin_id, config=config, **kwargs)
        return MockFormatter() # Fallback
    tm._plugin_manager.get_plugin_instance.side_effect = side_effect_formatter_none

    formatted_def = await tm.get_formatted_tool_definition("tool1", "unknown_formatter_id")
    assert formatted_def is None
    assert "DefinitionFormatter plugin 'unknown_formatter_id' not found or invalid." in caplog.text
    tm._plugin_manager.get_plugin_instance.side_effect = original_side_effect # Restore

@pytest.mark.asyncio
async def test_get_formatted_tool_definition_formatter_format_fails(initialized_tool_manager: ToolManager, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=TOOL_MANAGER_LOGGER_NAME)
    tm = await initialized_tool_manager
    mock_formatter_fails = MockFormatter()
    mock_formatter_fails.raise_in_format = True

    original_side_effect = tm._plugin_manager.get_plugin_instance.side_effect
    async def side_effect_formatter_fails(plugin_id: str, config=None, **kwargs):
        if plugin_id == mock_formatter_fails.plugin_id:
            return mock_formatter_fails
        if callable(original_side_effect):
             return await original_side_effect(plugin_id, config=config, **kwargs)
        return None
    tm._plugin_manager.get_plugin_instance.side_effect = side_effect_formatter_fails

    formatted_def = await tm.get_formatted_tool_definition("tool1", mock_formatter_fails.plugin_id)
    assert formatted_def is None
    assert f"Error formatting tool 'tool1' with formatter '{mock_formatter_fails.plugin_id}': Formatter format failed" in caplog.text
    tm._plugin_manager.get_plugin_instance.side_effect = original_side_effect # Restore

@pytest.mark.asyncio
async def test_get_formatted_tool_definition_tool_get_metadata_fails(initialized_tool_manager: ToolManager, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=TOOL_MANAGER_LOGGER_NAME)
    tm = await initialized_tool_manager

    tool1 = await tm.get_tool("tool1")
    assert tool1 is not None
    tool1._metadata["raise_in_get_metadata"] = True # Make tool1 fail metadata retrieval

    mock_formatter_good = MockFormatter()
    original_side_effect = tm._plugin_manager.get_plugin_instance.side_effect
    async def side_effect_good_formatter(plugin_id: str, config=None, **kwargs):
        if plugin_id == mock_formatter_good.plugin_id:
            return mock_formatter_good
        if callable(original_side_effect):
             return await original_side_effect(plugin_id, config=config, **kwargs)
        return None
    tm._plugin_manager.get_plugin_instance.side_effect = side_effect_good_formatter

    formatted_def = await tm.get_formatted_tool_definition("tool1", mock_formatter_good.plugin_id)
    assert formatted_def is None
    assert f"Error formatting tool 'tool1' with formatter '{mock_formatter_good.plugin_id}': Metadata retrieval failed" in caplog.text
    tm._plugin_manager.get_plugin_instance.side_effect = original_side_effect # Restore

@pytest.mark.asyncio
async def test_tool_manager_constructor_injects_pm_if_tool_accepts_it(mock_plugin_manager_fixture: PluginManager):
    pm = mock_plugin_manager_fixture
    tool_accept_pm_plugin_id = "tool_accept_pm_plugin_id_test"

    class ToolAcceptingPM(MockTool):
        def __init__(self, identifier_val: str, metadata: Dict[str, Any], plugin_manager: PluginManager):
            super().__init__(identifier_val, metadata, plugin_manager=plugin_manager)

    pm.list_discovered_plugin_classes.return_value = {tool_accept_pm_plugin_id: ToolAcceptingPM}
    pm._discovered_plugin_classes = pm.list_discovered_plugin_classes.return_value

    async def get_instance_side_effect(plugin_id: str, config=None, **kwargs):
        # kwargs here will include plugin_manager passed by ToolManager
        if plugin_id == tool_accept_pm_plugin_id:
            # Pass kwargs to the constructor of ToolAcceptingPM
            tool = ToolAcceptingPM(tool_accept_pm_plugin_id, {"identifier": tool_accept_pm_plugin_id}, **kwargs)
            await tool.setup(config)
            return tool
        return None
    pm.get_plugin_instance.side_effect = get_instance_side_effect

    tm = ToolManager(plugin_manager=pm)
    # Configure the tool to be loaded
    await tm.initialize_tools(tool_configurations={tool_accept_pm_plugin_id: {}})

    tool_instance = await tm.get_tool(tool_accept_pm_plugin_id)
    assert tool_instance is not None, f"Tool '{tool_accept_pm_plugin_id}' was not loaded."
    assert isinstance(tool_instance, ToolAcceptingPM)
    assert tool_instance.injected_plugin_manager is pm
