### tests/unit/tools/test_tool_manager.py
"""Unit tests for the ToolManager."""
import logging  # <<<< ADDED IMPORT
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import Plugin
from genie_tooling.tools.abc import Tool as ToolPlugin
from genie_tooling.tools.formatters.abc import (
    DefinitionFormatter as DefinitionFormatterPlugin,
)
from genie_tooling.tools.manager import ToolManager


class MockTool(ToolPlugin):
    identifier: str
    plugin_id: str

    def __init__(self, identifier: str, metadata: Dict[str, Any], execute_result: Any = "tool_executed", plugin_manager=None):
        self.identifier = identifier
        self.plugin_id = identifier
        self._metadata = metadata
        self._execute_result = execute_result
        self.setup_called_with_config: Optional[Dict[str, Any]] = None
        self.injected_plugin_manager = plugin_manager
        self.teardown_called = False
    async def get_metadata(self) -> Dict[str, Any]:
        if "raise_in_get_metadata" in self._metadata:
            raise RuntimeError("Metadata retrieval failed")
        return self._metadata
    async def execute(self, params, key_provider, context=None) -> Any: return self._execute_result
    async def setup(self, config=None): self.setup_called_with_config = config
    async def teardown(self): self.teardown_called = True


class MockToolNoPMInInit(ToolPlugin):
    identifier: str
    plugin_id: str
    def __init__(self, identifier: str, metadata: Dict[str,Any]):
        self.identifier = identifier
        self.plugin_id = identifier
        self._metadata = metadata
    async def get_metadata(self) -> Dict[str, Any]: return self._metadata
    async def execute(self, params, key_provider, context=None) -> Any: return "executed_no_pm"
    async def setup(self, config=None): pass
    async def teardown(self): pass

class MockNonToolPlugin(Plugin): # Not a Tool
    plugin_id: str = "non_tool_plugin_v1"
    description: str = "Not a tool."
    async def setup(self, config=None): pass
    async def teardown(self): pass


class MockFormatter(DefinitionFormatterPlugin):
    plugin_id: str = "mock_formatter_plugin_v1"
    formatter_id: str = "mock_format_v1"
    description: str = "A mock definition formatter."
    raise_in_format: bool = False
    def format(self, tool_metadata: Dict[str, Any]) -> Any:
        if self.raise_in_format:
            raise RuntimeError("Formatter format failed")
        return {"formatted": True, "original_id": tool_metadata.get("identifier")}
    async def setup(self,config: Optional[Dict[str, Any]]=None): pass
    async def teardown(self): pass


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
async def initialized_tool_manager_fixture(mock_plugin_manager_fixture: PluginManager) -> ToolManager:
    actual_mock_pm: PluginManager = mock_plugin_manager_fixture

    tool1_meta = {"identifier": "tool1", "name": "Tool One", "description_llm": "Desc1"}
    tool2_meta = {"identifier": "tool2", "name": "Tool Two", "description_llm": "Desc2"}

    actual_mock_pm.list_discovered_plugin_classes.return_value = {
        "tool1_plugin_id_from_discovery": MockTool,
        "tool2_plugin_id_from_discovery": MockTool,
        MockFormatter.plugin_id: MockFormatter
    }
    actual_mock_pm._discovered_plugin_classes = actual_mock_pm.list_discovered_plugin_classes.return_value

    async def get_instance_side_effect(plugin_id: str, config: Optional[Dict[str, Any]] = None, **kwargs):
        init_kwargs = kwargs
        if plugin_id == "tool1_plugin_id_from_discovery":
            tool = MockTool(identifier="tool1", metadata=tool1_meta, **init_kwargs)
            await tool.setup(config=config)
            return tool
        if plugin_id == "tool2_plugin_id_from_discovery":
            tool = MockTool(identifier="tool2", metadata=tool2_meta, **init_kwargs)
            await tool.setup(config=config)
            return tool
        if plugin_id == MockFormatter.plugin_id:
            formatter = MockFormatter()
            await formatter.setup(config=config)
            return formatter
        return None
    actual_mock_pm.get_plugin_instance.side_effect = get_instance_side_effect

    tm = ToolManager(plugin_manager=actual_mock_pm)
    await tm.initialize_tools(tool_configurations={
        "tool1_plugin_id_from_discovery": {"setup_key": "setup_val1"},
        "tool2_plugin_id_from_discovery": {"setup_key": "setup_val2"}
    })
    return tm

@pytest.mark.asyncio
async def test_tool_manager_initialization_no_tools(mock_plugin_manager_fixture: PluginManager):
    actual_mock_pm = mock_plugin_manager_fixture
    actual_mock_pm._discovered_plugin_classes = {}
    actual_mock_pm.list_discovered_plugin_classes.return_value = {}

    tm = ToolManager(plugin_manager=actual_mock_pm)
    await tm.initialize_tools()

    if not actual_mock_pm.list_discovered_plugin_classes.called:
         actual_mock_pm.discover_plugins.assert_awaited_once()
    actual_mock_pm.list_discovered_plugin_classes.assert_called()
    assert len(await tm.list_tools()) == 0


@pytest.mark.asyncio
async def test_tool_manager_initializes_discovered_tools(initialized_tool_manager_fixture: ToolManager):
    tm = await initialized_tool_manager_fixture
    tools = await tm.list_tools()
    assert len(tools) == 2, f"Expected 2 tools, got {len(tools)}. Tools: {[t.identifier for t in tools]}"
    tool1_instance = await tm.get_tool("tool1")
    assert tool1_instance is not None
    assert tool1_instance.setup_called_with_config == {"setup_key": "setup_val1"}
    if isinstance(tool1_instance, MockTool):
        assert tool1_instance.injected_plugin_manager is not None


@pytest.mark.asyncio
async def test_tool_manager_init_calls_discover_if_needed(mock_plugin_manager_fixture: PluginManager):
    pm = mock_plugin_manager_fixture
    pm._discovered_plugin_classes = {}
    pm.list_discovered_plugin_classes.side_effect = [{}, {"some_tool_plugin": MockTool}]

    tm = ToolManager(plugin_manager=pm)
    await tm.initialize_tools()

    pm.discover_plugins.assert_awaited_once()


@pytest.mark.asyncio
async def test_tool_manager_handles_tool_init_with_and_without_pm_arg(mock_plugin_manager_fixture: PluginManager):
    pm = mock_plugin_manager_fixture
    tool_with_pm_meta = {"identifier": "tool_pm", "name": "Tool With PM"}
    tool_no_pm_meta = {"identifier": "tool_no_pm", "name": "Tool No PM"}

    pm.list_discovered_plugin_classes.return_value = {
        "tool_with_pm_plugin": MockTool,
        "tool_no_pm_plugin": MockToolNoPMInInit
    }
    pm._discovered_plugin_classes = pm.list_discovered_plugin_classes.return_value

    async def get_instance_side_effect_init_test(plugin_id: str, config: Optional[Dict[str, Any]] = None, **kwargs):
        if plugin_id == "tool_with_pm_plugin":
            instance = MockTool(identifier="tool_pm", metadata=tool_with_pm_meta, **kwargs)
            await instance.setup(config)
            return instance
        if plugin_id == "tool_no_pm_plugin":
            assert "plugin_manager" not in kwargs, "plugin_manager should not be passed to MockToolNoPMInInit"
            instance = MockToolNoPMInInit(identifier="tool_no_pm", metadata=tool_no_pm_meta)
            await instance.setup(config)
            return instance
        return None
    pm.get_plugin_instance.side_effect = get_instance_side_effect_init_test

    tm = ToolManager(plugin_manager=pm)
    await tm.initialize_tools()

    tool_pm_instance = await tm.get_tool("tool_pm")
    assert isinstance(tool_pm_instance, MockTool)
    assert tool_pm_instance.injected_plugin_manager is pm

    tool_no_pm_instance = await tm.get_tool("tool_no_pm")
    assert isinstance(tool_no_pm_instance, MockToolNoPMInInit)

@pytest.mark.asyncio
async def test_tool_manager_init_skips_non_tool_plugins(mock_plugin_manager_fixture: PluginManager):
    pm = mock_plugin_manager_fixture
    pm.list_discovered_plugin_classes.return_value = {
        "tool_plugin": MockTool,
        "non_tool_plugin": MockNonToolPlugin
    }
    pm._discovered_plugin_classes = pm.list_discovered_plugin_classes.return_value

    async def get_instance_side_effect_skip(plugin_id: str, config=None, **kwargs):
        if plugin_id == "tool_plugin":
            return MockTool("tool_id", {"identifier": "tool_id"}, plugin_manager=pm)
        if plugin_id == "non_tool_plugin":
            return MockNonToolPlugin()
        return None
    pm.get_plugin_instance.side_effect = get_instance_side_effect_skip

    tm = ToolManager(plugin_manager=pm)
    await tm.initialize_tools()
    tools = await tm.list_tools()
    assert len(tools) == 1
    assert tools[0].identifier == "tool_id"

@pytest.mark.asyncio
async def test_tool_manager_init_handles_get_instance_failure(mock_plugin_manager_fixture: PluginManager, caplog):
    caplog.set_level(logging.ERROR)
    pm = mock_plugin_manager_fixture
    pm.list_discovered_plugin_classes.return_value = {"failing_tool_plugin": MockTool}
    pm._discovered_plugin_classes = {"failing_tool_plugin": MockTool}
    pm.get_plugin_instance.side_effect = RuntimeError("Failed to get instance")

    tm = ToolManager(plugin_manager=pm)
    await tm.initialize_tools()
    tools = await tm.list_tools()
    assert len(tools) == 0
    assert "Error initializing tool plugin 'failing_tool_plugin'" in caplog.text
    assert "Failed to get instance" in caplog.text

@pytest.mark.asyncio
async def test_tool_manager_init_handles_duplicate_identifier(mock_plugin_manager_fixture: PluginManager, caplog):
    caplog.set_level(logging.WARNING)
    pm = mock_plugin_manager_fixture
    tool_meta = {"identifier": "duplicate_id", "name": "Duplicate Tool"}
    pm.list_discovered_plugin_classes.return_value = {
        "plugin_a": MockTool,
        "plugin_b": MockTool
    }
    pm._discovered_plugin_classes = pm.list_discovered_plugin_classes.return_value

    async def get_instance_side_effect_dup(plugin_id: str, config=None, **kwargs):
        # Both plugins will resolve to the same identifier "duplicate_id"
        return MockTool(identifier="duplicate_id", metadata=tool_meta, plugin_manager=pm)
    pm.get_plugin_instance.side_effect = get_instance_side_effect_dup

    tm = ToolManager(plugin_manager=pm)
    await tm.initialize_tools()
    tools = await tm.list_tools()
    assert len(tools) == 1 # Only one instance for "duplicate_id" should be stored
    assert tools[0].identifier == "duplicate_id"
    assert "Duplicate tool identifier 'duplicate_id' encountered" in caplog.text
    assert "Overwriting previous tool with same identifier." in caplog.text

@pytest.mark.asyncio
async def test_tool_manager_init_inspect_constructor_failure(mock_plugin_manager_fixture: PluginManager, caplog):
    caplog.set_level(logging.DEBUG)
    pm = mock_plugin_manager_fixture

    class UninspectableToolPlugin(ToolPlugin): # Not really uninspectable, but to test the log path
        identifier: str = "uninspectable"
        plugin_id: str = "uninspectable_plugin"
        async def get_metadata(self) -> Dict[str, Any]: return {}
        async def execute(self, params, key_provider, context=None) -> Any: return None
        async def setup(self, config=None): pass
        async def teardown(self): pass

    pm.list_discovered_plugin_classes.return_value = {"uninspectable_plugin": UninspectableToolPlugin}
    pm._discovered_plugin_classes = {"uninspectable_plugin": UninspectableToolPlugin}

    async def get_inst_side_effect(plugin_id, config=None, **kwargs):
        if plugin_id == "uninspectable_plugin":
            return UninspectableToolPlugin()
        return None
    pm.get_plugin_instance.side_effect = get_inst_side_effect

    with patch("inspect.signature", side_effect=ValueError("Cannot inspect")):
        tm = ToolManager(plugin_manager=pm)
        await tm.initialize_tools()

    assert "Could not inspect __init__ for UninspectableToolPlugin" in caplog.text
    tools = await tm.list_tools()
    assert len(tools) == 1
    assert tools[0].identifier == "uninspectable"


@pytest.mark.asyncio
async def test_get_tool_found(initialized_tool_manager_fixture: ToolManager):
    tm = await initialized_tool_manager_fixture
    tool = await tm.get_tool("tool1")
    assert tool is not None
    assert tool.identifier == "tool1"

@pytest.mark.asyncio
async def test_get_tool_not_found(initialized_tool_manager_fixture: ToolManager):
    tm = await initialized_tool_manager_fixture
    tool = await tm.get_tool("non_existent_tool")
    assert tool is None

@pytest.mark.asyncio
async def test_list_tool_summaries(initialized_tool_manager_fixture: ToolManager):
    tm = await initialized_tool_manager_fixture
    summaries, pagination = await tm.list_tool_summaries()
    assert len(summaries) == 2
    assert pagination["total_items"] == 2

@pytest.mark.asyncio
async def test_list_tool_summaries_pagination(mock_plugin_manager_fixture: PluginManager):
    actual_mock_pm = mock_plugin_manager_fixture
    num_tools = 25
    discovered_classes_map = {}
    tool_setup_configs = {}

    for i in range(num_tools):
        plugin_reg_id = f"pag_tool_plugin_{i}"
        discovered_classes_map[plugin_reg_id] = MockTool
        tool_setup_configs[plugin_reg_id] = {"index": i}

    actual_mock_pm.list_discovered_plugin_classes.return_value = discovered_classes_map
    actual_mock_pm._discovered_plugin_classes = discovered_classes_map

    async def pag_get_instance_side_effect(plugin_id: str, config=None, **kwargs):
        if plugin_id.startswith("pag_tool_plugin_"):
            idx = int(plugin_id.split("_")[-1])
            tool_ident = f"pag_tool_{idx}"
            instance = MockTool(identifier=tool_ident, metadata={"identifier": tool_ident, "name": f"Pag Tool {idx}"}, **kwargs)
            await instance.setup(config=config)
            return instance
        return None
    actual_mock_pm.get_plugin_instance.side_effect = pag_get_instance_side_effect

    tm_pag = ToolManager(plugin_manager=actual_mock_pm)
    await tm_pag.initialize_tools(tool_configurations=tool_setup_configs)

    tools_list = await tm_pag.list_tools()
    assert len(tools_list) == num_tools, f"Expected {num_tools} tools, got {len(tools_list)}"

    summaries_p1, pag_p1 = await tm_pag.list_tool_summaries(pagination_params={"page": 1, "page_size": 10})
    assert len(summaries_p1) == 10
    assert pag_p1["total_pages"] == 3
    assert pag_p1["total_items"] == num_tools

    summaries_p3, pag_p3 = await tm_pag.list_tool_summaries(pagination_params={"page": 3, "page_size": 10})
    assert len(summaries_p3) == 5
    assert pag_p3["current_page"] == 3

    # Test edge cases for pagination
    summaries_edge1, pag_edge1 = await tm_pag.list_tool_summaries(pagination_params={"page": 0, "page_size": 5}) # page 0 -> page 1
    assert len(summaries_edge1) == 5
    assert pag_edge1["current_page"] == 1

    summaries_edge2, pag_edge2 = await tm_pag.list_tool_summaries(pagination_params={"page": 100, "page_size": 5}) # page > total_pages -> last page
    assert len(summaries_edge2) == 0
    assert pag_edge2["current_page"] == 100
    assert pag_edge2["total_pages"] == 5

    summaries_edge3, pag_edge3 = await tm_pag.list_tool_summaries(pagination_params={"page": 1, "page_size": 0}) # page_size 0 -> default page_size (20)
    assert len(summaries_edge3) == 20
    assert pag_edge3["page_size"] == 20
    assert pag_edge3["total_pages"] == 2

@pytest.mark.asyncio
async def test_list_tool_summaries_get_metadata_error(mock_plugin_manager_fixture: PluginManager, caplog):
    caplog.set_level(logging.ERROR)
    pm = mock_plugin_manager_fixture
    pm.list_discovered_plugin_classes.return_value = {"error_tool_plugin": MockTool}
    pm._discovered_plugin_classes = {"error_tool_plugin": MockTool}

    async def get_inst_side_effect_meta_error(plugin_id, config=None, **kwargs):
        if plugin_id == "error_tool_plugin":
            return MockTool("error_tool", {"identifier": "error_tool", "raise_in_get_metadata": True}, plugin_manager=pm)
        return None
    pm.get_plugin_instance.side_effect = get_inst_side_effect_meta_error

    tm = ToolManager(plugin_manager=pm)
    await tm.initialize_tools()

    summaries, _ = await tm.list_tool_summaries()
    assert len(summaries) == 0
    assert "Error getting metadata for tool 'error_tool'" in caplog.text
    assert "Metadata retrieval failed" in caplog.text


@pytest.mark.asyncio
async def test_get_formatted_tool_definition_success(initialized_tool_manager_fixture: ToolManager):
    tm = await initialized_tool_manager_fixture
    formatted_def = await tm.get_formatted_tool_definition("tool1", MockFormatter.plugin_id)
    assert formatted_def is not None
    assert formatted_def.get("formatted") is True

@pytest.mark.asyncio
async def test_get_formatted_tool_definition_tool_not_found(initialized_tool_manager_fixture: ToolManager):
    tm = await initialized_tool_manager_fixture
    formatted_def = await tm.get_formatted_tool_definition("unknown_tool", MockFormatter.plugin_id)
    assert formatted_def is None

@pytest.mark.asyncio
async def test_get_formatted_tool_definition_formatter_not_found(mock_plugin_manager_fixture: PluginManager):
    actual_mock_pm = mock_plugin_manager_fixture

    actual_mock_pm.list_discovered_plugin_classes.return_value = {"tool1_plugin_id": MockTool}
    actual_mock_pm._discovered_plugin_classes = {"tool1_plugin_id": MockTool}

    async def side_effect_formatter_nf(plugin_id: str, config=None, **kwargs):
        if plugin_id == "tool1_plugin_id":
            tool_kwargs = {**kwargs}
            instance = MockTool("tool1", {"identifier":"tool1"}, **tool_kwargs)
            await instance.setup(config)
            return instance
        if plugin_id == "unknown_formatter": return None
        return MagicMock()
    actual_mock_pm.get_plugin_instance.side_effect = side_effect_formatter_nf

    tm = ToolManager(plugin_manager=actual_mock_pm)
    await tm.initialize_tools({"tool1_plugin_id": {}})

    formatted_def = await tm.get_formatted_tool_definition("tool1", "unknown_formatter")
    assert formatted_def is None

@pytest.mark.asyncio
async def test_get_formatted_tool_definition_tool_get_metadata_error(mock_plugin_manager_fixture: PluginManager, caplog):
    caplog.set_level(logging.ERROR)
    pm = mock_plugin_manager_fixture
    pm.list_discovered_plugin_classes.return_value = {
        "error_meta_tool_plugin": MockTool,
        MockFormatter.plugin_id: MockFormatter
    }
    pm._discovered_plugin_classes = pm.list_discovered_plugin_classes.return_value

    async def get_inst_side_effect_format_meta_error(plugin_id, config=None, **kwargs):
        if plugin_id == "error_meta_tool_plugin":
            return MockTool("error_meta_tool", {"identifier": "error_meta_tool", "raise_in_get_metadata": True}, plugin_manager=pm)
        if plugin_id == MockFormatter.plugin_id:
            return MockFormatter()
        return None
    pm.get_plugin_instance.side_effect = get_inst_side_effect_format_meta_error

    tm = ToolManager(plugin_manager=pm)
    await tm.initialize_tools()

    formatted_def = await tm.get_formatted_tool_definition("error_meta_tool", MockFormatter.plugin_id)
    assert formatted_def is None
    assert "Error formatting tool 'error_meta_tool'" in caplog.text
    assert "Metadata retrieval failed" in caplog.text

@pytest.mark.asyncio
async def test_get_formatted_tool_definition_formatter_format_error(mock_plugin_manager_fixture: PluginManager, caplog):
    caplog.set_level(logging.ERROR)
    pm = mock_plugin_manager_fixture
    pm.list_discovered_plugin_classes.return_value = {
        "good_tool_plugin": MockTool,
        MockFormatter.plugin_id: MockFormatter
    }
    pm._discovered_plugin_classes = pm.list_discovered_plugin_classes.return_value

    async def get_inst_side_effect_format_error(plugin_id, config=None, **kwargs):
        if plugin_id == "good_tool_plugin":
            return MockTool("good_tool", {"identifier": "good_tool"}, plugin_manager=pm)
        if plugin_id == MockFormatter.plugin_id:
            formatter = MockFormatter()
            formatter.raise_in_format = True
            return formatter
        return None
    pm.get_plugin_instance.side_effect = get_inst_side_effect_format_error

    tm = ToolManager(plugin_manager=pm)
    await tm.initialize_tools()

    formatted_def = await tm.get_formatted_tool_definition("good_tool", MockFormatter.plugin_id)
    assert formatted_def is None
    assert "Error formatting tool 'good_tool'" in caplog.text
    assert "Formatter format failed" in caplog.text


@pytest.mark.asyncio
async def test_list_available_formatters_no_formatters(mock_plugin_manager_fixture: PluginManager):
    pm = mock_plugin_manager_fixture
    pm.list_discovered_plugin_classes.return_value = {
        "tool_plugin_id": MockTool,
    }
    async def get_instance_side_effect_no_fmt(plugin_id: str, config=None, **kwargs):
        if plugin_id == "tool_plugin_id": return MockTool("tool_id", {}, plugin_manager=pm)
        return None
    pm.get_plugin_instance.side_effect = get_instance_side_effect_no_fmt

    tm = ToolManager(plugin_manager=pm)
    formatters = await tm.list_available_formatters()
    assert len(formatters) == 0

@pytest.mark.asyncio
async def test_list_available_formatters_mixed_plugins(mock_plugin_manager_fixture: PluginManager):
    pm = mock_plugin_manager_fixture
    mock_formatter_instance = MockFormatter()

    pm.list_discovered_plugin_classes.return_value = {
        MockFormatter.plugin_id: MockFormatter,
        "tool_plugin_id": MockTool
    }
    async def get_instance_side_effect_mixed(plugin_id: str, config=None, **kwargs):
        if plugin_id == MockFormatter.plugin_id: return mock_formatter_instance
        if plugin_id == "tool_plugin_id": return MockTool("tool_id", {}, plugin_manager=pm)
        return None
    pm.get_plugin_instance.side_effect = get_instance_side_effect_mixed

    tm = ToolManager(plugin_manager=pm)
    formatters = await tm.list_available_formatters()
    assert len(formatters) == 1
    assert formatters[0]["id"] == MockFormatter.formatter_id

@pytest.mark.asyncio
async def test_list_available_formatters_discover_needed(mock_plugin_manager_fixture: PluginManager):
    pm = mock_plugin_manager_fixture
    pm._discovered_plugin_classes = {}

    pm.list_discovered_plugin_classes.side_effect = lambda: {MockFormatter.plugin_id: MockFormatter}

    async def get_inst_side_effect_discover(plugin_id: str, config=None, **kwargs):
        if plugin_id == MockFormatter.plugin_id:
            return MockFormatter()
        return None
    pm.get_plugin_instance.side_effect = get_inst_side_effect_discover

    tm = ToolManager(plugin_manager=pm)
    await tm.list_available_formatters()
    pm.discover_plugins.assert_awaited_once()


@pytest.mark.asyncio
async def test_list_available_formatters(mock_plugin_manager_fixture: PluginManager, mocker):
    actual_mock_pm = mock_plugin_manager_fixture

    actual_mock_pm.list_discovered_plugin_classes.return_value = {
        MockFormatter.plugin_id: MockFormatter,
        "some_other_plugin": MockTool
    }

    async def get_instance_side_effect_formatters(plugin_id, config=None, **kwargs):
        if plugin_id == MockFormatter.plugin_id:
            return MockFormatter()
        if plugin_id == "some_other_plugin":
            tool_kwargs = {**kwargs}
            return MockTool(identifier="other", metadata={}, **tool_kwargs)
        return None
    actual_mock_pm.get_plugin_instance.side_effect = get_instance_side_effect_formatters

    tm = ToolManager(plugin_manager=actual_mock_pm)

    formatters = await tm.list_available_formatters()
    assert len(formatters) == 1
    assert formatters[0]["id"] == MockFormatter.formatter_id
    assert formatters[0]["plugin_id"] == MockFormatter.plugin_id

@pytest.mark.asyncio
async def test_list_available_formatters_get_instance_fails(mock_plugin_manager_fixture: PluginManager, caplog):
    caplog.set_level(logging.DEBUG)
    pm = mock_plugin_manager_fixture
    pm.list_discovered_plugin_classes.return_value = {
        MockFormatter.plugin_id: MockFormatter,
        "failing_formatter_plugin": MockFormatter # Simulate another formatter that fails instantiation
    }
    async def get_inst_side_effect_fail(plugin_id, config=None, **kwargs):
        if plugin_id == MockFormatter.plugin_id:
            return MockFormatter()
        if plugin_id == "failing_formatter_plugin":
            raise RuntimeError("Formatter instantiation failed")
        return None
    pm.get_plugin_instance.side_effect = get_inst_side_effect_fail

    tm = ToolManager(plugin_manager=pm)
    formatters = await tm.list_available_formatters()
    assert len(formatters) == 1 # Only the successful one
    assert formatters[0]["id"] == MockFormatter.formatter_id
    assert "Could not instantiate or check plugin 'failing_formatter_plugin' as DefinitionFormatter" in caplog.text
    assert "Formatter instantiation failed" in caplog.text

# Note: The NameError fixes for OpenAI formatter tests were implicitly handled
# by ensuring this file, tests/unit/tools/test_tool_manager.py, has `import logging`.
# If those specific tests (test_openai_formatter_...) were intended for test_formatters.py,
# they should be moved there, and test_formatters.py should also have `import logging`.
