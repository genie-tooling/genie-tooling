### tests/unit/tools/test_tool_manager.py
import inspect
import logging
from typing import Any, Dict, List, Optional
from unittest.mock import ANY, AsyncMock, MagicMock, PropertyMock, patch

import pytest
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.definition_formatters.abc import DefinitionFormatter
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
        self.teardown_called: bool = False
    
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

class MockFormatter(DefinitionFormatter):
    plugin_id_val: str = "mock_formatter_v1" # Renamed to avoid conflict with property
    formatter_id: str = "mock_format_output_v1"
    description: str = "Mock Formatter"

    @property
    def plugin_id(self) -> str: # Added property
        return self.plugin_id_val

    def format(self, tool_metadata: Dict[str, Any]) -> Any:
        return f"Formatted: {tool_metadata.get('name', 'Unknown')}"
    async def setup(self, config=None): pass
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
def tool_manager_fixture(mock_plugin_manager_fixture: PluginManager) -> ToolManager:
    return ToolManager(plugin_manager=mock_plugin_manager_fixture)


@pytest.mark.asyncio
class TestToolManagerInitializeTools:
    async def test_initialize_tools_no_discovered_plugins(self, mock_plugin_manager_fixture: PluginManager, caplog: pytest.LogCaptureFixture):
        caplog.set_level(logging.INFO, logger=TOOL_MANAGER_LOGGER_NAME)
        pm = mock_plugin_manager_fixture
        pm.list_discovered_plugin_classes.return_value = {}
        pm._discovered_plugin_classes = {}
        tm = ToolManager(plugin_manager=pm)
        await tm.initialize_tools(tool_configurations={})
        assert len(await tm.list_tools()) == 0
        assert "ToolManager initialized. Loaded 0 explicitly configured class-based tools." in caplog.text

    async def test_initialize_tools_success(self, tool_manager_fixture: ToolManager, mock_plugin_manager_fixture: PluginManager):
        tm = tool_manager_fixture
        tool_id = "tool_alpha"
        mock_tool_alpha_class = MockTool
        mock_tool_alpha_instance = MockTool(tool_id, {"name": "Alpha"})
        tool_config_for_setup = {"config_key": "val"}

        mock_plugin_manager_fixture.list_discovered_plugin_classes.return_value = {tool_id: mock_tool_alpha_class}
        
        async def get_instance_side_effect(pid, config, **kwargs):
            if pid == tool_id:
                # Simulate PM calling setup on the instance
                # The instance should be created by PM, then setup called.
                # For this test, we assume PM correctly instantiates and calls setup.
                # The important part is that ToolManager passes the right config to PM.
                # So, we'll have PM return our pre-configured mock instance.
                await mock_tool_alpha_instance.setup(config) # Simulate PM calling setup
                return mock_tool_alpha_instance
            return None
        mock_plugin_manager_fixture.get_plugin_instance.side_effect = get_instance_side_effect

        await tm.initialize_tools(tool_configurations={tool_id: tool_config_for_setup})
        
        loaded_tools = await tm.list_tools()
        assert len(loaded_tools) == 1
        assert loaded_tools[0].identifier == tool_id
        assert mock_tool_alpha_instance.setup_called_with_config == tool_config_for_setup
        # ToolManager passes plugin_manager in kwargs to get_plugin_instance
        mock_plugin_manager_fixture.get_plugin_instance.assert_awaited_once_with(tool_id, config=tool_config_for_setup, plugin_manager=mock_plugin_manager_fixture)


    async def test_initialize_tools_tool_not_discovered(self, tool_manager_fixture: ToolManager, mock_plugin_manager_fixture: PluginManager, caplog):
        caplog.set_level(logging.DEBUG, logger=TOOL_MANAGER_LOGGER_NAME)
        tm = tool_manager_fixture
        mock_plugin_manager_fixture.list_discovered_plugin_classes.return_value = {}
        await tm.initialize_tools(tool_configurations={"non_existent_tool": {}})
        assert len(await tm.list_tools()) == 0
        assert "Tool ID/alias 'non_existent_tool' not found as a discovered plugin class." in caplog.text

    async def test_initialize_tools_alias_resolution(self, tool_manager_fixture: ToolManager, mock_plugin_manager_fixture: PluginManager):
        tm = tool_manager_fixture
        tool_alias = "calc" 
        
        mock_calc_class = MockTool
        mock_calc_instance = MockTool("calc", {"name": "Calculator"})

        mock_plugin_manager_fixture.list_discovered_plugin_classes.return_value = {"calc": mock_calc_class}
        async def get_instance_side_effect_alias(pid, config, **kwargs):
            if pid == "calc":
                await mock_calc_instance.setup(config)
                if "plugin_manager" in kwargs:
                    mock_calc_instance.injected_plugin_manager = kwargs["plugin_manager"]
                return mock_calc_instance
            return None
        mock_plugin_manager_fixture.get_plugin_instance.side_effect = get_instance_side_effect_alias

        await tm.initialize_tools(tool_configurations={tool_alias: {}})
        
        loaded_tools = await tm.list_tools()
        assert len(loaded_tools) == 1
        assert loaded_tools[0].identifier == "calc"
        mock_plugin_manager_fixture.get_plugin_instance.assert_awaited_once_with(tool_alias, config={}, plugin_manager=mock_plugin_manager_fixture)


    async def test_initialize_tools_duplicate_identifier_warning(self, tool_manager_fixture: ToolManager, mock_plugin_manager_fixture: PluginManager, caplog):
        caplog.set_level(logging.WARNING)
        tm = tool_manager_fixture
        common_id = "common_tool_id"
        
        mock_tool_v1_class = type("MockToolV1", (MockTool,), {})
        mock_tool_v2_class = type("MockToolV2", (MockTool,), {})

        mock_tool_v1_instance = MockTool(common_id, {"name": "V1"})
        mock_tool_v2_instance = MockTool(common_id, {"name": "V2"})

        mock_plugin_manager_fixture.list_discovered_plugin_classes.return_value = {
            "plugin_v1_id": mock_tool_v1_class,
            "plugin_v2_id": mock_tool_v2_class
        }
        
        async def get_instance_side_effect_dup(pid, config, **kwargs):
            if pid == "plugin_v1_id":
                await mock_tool_v1_instance.setup(config)
                return mock_tool_v1_instance
            if pid == "plugin_v2_id":
                await mock_tool_v2_instance.setup(config)
                return mock_tool_v2_instance
            return None
        mock_plugin_manager_fixture.get_plugin_instance.side_effect = get_instance_side_effect_dup

        await tm.initialize_tools(tool_configurations={"plugin_v1_id": {}, "plugin_v2_id": {}})
        
        loaded_tools = await tm.list_tools()
        assert len(loaded_tools) == 1
        assert loaded_tools[0].identifier == common_id
        assert "Duplicate tool identifier 'common_tool_id' encountered" in caplog.text


    async def test_initialize_tools_plugin_init_needs_plugin_manager(self, tool_manager_fixture: ToolManager, mock_plugin_manager_fixture: PluginManager):
        tm = tool_manager_fixture
        tool_id = "pm_aware_tool"
        mock_pm_aware_tool_class = MockTool
        
        async def get_instance_side_effect_pm_aware(pid, config, **kwargs):
            if pid == tool_id:
                # Simulate PM creating the instance and passing plugin_manager
                instance = mock_pm_aware_tool_class(identifier_val=pid, metadata={"name":"PMAware"}, plugin_manager=kwargs.get("plugin_manager"))
                await instance.setup(config)
                return instance
            return None
        mock_plugin_manager_fixture.get_plugin_instance.side_effect = get_instance_side_effect_pm_aware
        mock_plugin_manager_fixture.list_discovered_plugin_classes.return_value = {tool_id: mock_pm_aware_tool_class}


        await tm.initialize_tools(tool_configurations={tool_id: {}})
        
        loaded_tool = await tm.get_tool(tool_id)
        assert loaded_tool is not None
        assert loaded_tool.injected_plugin_manager is mock_plugin_manager_fixture # type: ignore
        mock_plugin_manager_fixture.get_plugin_instance.assert_awaited_once_with(tool_id, config={}, plugin_manager=mock_plugin_manager_fixture)


    async def test_initialize_tools_plugin_instantiation_fails(self, tool_manager_fixture: ToolManager, mock_plugin_manager_fixture: PluginManager, caplog):
        caplog.set_level(logging.ERROR)
        tm = tool_manager_fixture
        tool_id = "fail_init_tool"
        
        mock_plugin_manager_fixture.list_discovered_plugin_classes.return_value = {tool_id: MagicMock()}
        mock_plugin_manager_fixture.get_plugin_instance.side_effect = TypeError("Cannot instantiate from PM")

        await tm.initialize_tools(tool_configurations={tool_id: {}})
        assert len(await tm.list_tools()) == 0
        assert f"Error initializing tool plugin from ID/alias '{tool_id}'" in caplog.text
        assert "Cannot instantiate from PM" in caplog.text


    async def test_initialize_tools_plugin_setup_fails(self, tool_manager_fixture: ToolManager, mock_plugin_manager_fixture: PluginManager, caplog):
        caplog.set_level(logging.ERROR)
        tm = tool_manager_fixture
        tool_id = "fail_setup_tool"
        mock_tool_class = MockTool
        
        mock_plugin_manager_fixture.list_discovered_plugin_classes.return_value = {tool_id: mock_tool_class}
        # PluginManager's get_plugin_instance is responsible for calling setup.
        # If setup fails there, get_plugin_instance should raise or return None.
        mock_plugin_manager_fixture.get_plugin_instance.side_effect = RuntimeError("Setup failed from PM during get_instance")

        await tm.initialize_tools(tool_configurations={tool_id: {}})
        assert len(await tm.list_tools()) == 0
        assert f"Error initializing tool plugin from ID/alias '{tool_id}'" in caplog.text
        assert "Setup failed from PM during get_instance" in caplog.text


@pytest.mark.asyncio
class TestToolManagerListFormatters:
    async def test_list_available_formatters_none_discovered(self, tool_manager_fixture: ToolManager, mock_plugin_manager_fixture: PluginManager):
        tm = tool_manager_fixture
        mock_plugin_manager_fixture.list_discovered_plugin_classes.return_value = {}
        formatters = await tm.list_available_formatters()
        assert formatters == []

    async def test_list_available_formatters_success(self, tool_manager_fixture: ToolManager, mock_plugin_manager_fixture: PluginManager):
        tm = tool_manager_fixture
        mock_formatter_instance = MockFormatter()
        
        mock_plugin_manager_fixture.list_discovered_plugin_classes.return_value = {mock_formatter_instance.plugin_id: MockFormatter}
        mock_plugin_manager_fixture.get_plugin_instance.return_value = mock_formatter_instance
        
        formatters = await tm.list_available_formatters()
        assert len(formatters) == 1
        assert formatters[0]["id"] == mock_formatter_instance.formatter_id
        assert formatters[0]["description"] == mock_formatter_instance.description
        assert formatters[0]["plugin_id"] == mock_formatter_instance.plugin_id

    async def test_list_available_formatters_not_a_formatter(self, tool_manager_fixture: ToolManager, mock_plugin_manager_fixture: PluginManager, caplog):
        caplog.set_level(logging.DEBUG)
        tm = tool_manager_fixture
        mock_not_formatter_instance = MockTool("not_fmt", {}, "")
        
        mock_plugin_manager_fixture.list_discovered_plugin_classes.return_value = {"not_fmt_plugin_id": type(mock_not_formatter_instance)}
        mock_plugin_manager_fixture.get_plugin_instance.return_value = mock_not_formatter_instance
        
        formatters = await tm.list_available_formatters()
        assert len(formatters) == 0

    async def test_list_available_formatters_instantiation_fails(self, tool_manager_fixture: ToolManager, mock_plugin_manager_fixture: PluginManager, caplog):
        caplog.set_level(logging.DEBUG, logger=TOOL_MANAGER_LOGGER_NAME) # Target specific logger
        tm = tool_manager_fixture
        mock_plugin_manager_fixture.list_discovered_plugin_classes.return_value = {"fail_fmt_id": MockFormatter}
        mock_plugin_manager_fixture.get_plugin_instance.side_effect = RuntimeError("Formatter init failed")

        formatters = await tm.list_available_formatters()
        assert len(formatters) == 0
        assert "Could not instantiate or check plugin 'fail_fmt_id' as DefinitionFormatter: Formatter init failed" in caplog.text


@pytest.mark.asyncio
class TestToolManagerGetTool:
    async def test_get_tool_exists(self, tool_manager_fixture: ToolManager):
        tm = tool_manager_fixture
        mock_tool = MockTool("tool_exists", {}, "")
        tm._tools = {"tool_exists": mock_tool}
        tool = await tm.get_tool("tool_exists")
        assert tool is mock_tool

    async def test_get_tool_not_exists(self, tool_manager_fixture: ToolManager, caplog):
        caplog.set_level(logging.DEBUG, logger=TOOL_MANAGER_LOGGER_NAME)
        tm = tool_manager_fixture
        tm._tools = {}
        tool = await tm.get_tool("tool_not_found")
        assert tool is None
        assert "Tool with identifier 'tool_not_found' not found" in caplog.text


@pytest.mark.asyncio
class TestToolManagerListTools:
    async def test_list_tools_empty(self, tool_manager_fixture: ToolManager):
        tm = tool_manager_fixture
        tm._tools = {}
        tools = await tm.list_tools()
        assert tools == []

    async def test_list_tools_multiple(self, tool_manager_fixture: ToolManager):
        tm = tool_manager_fixture
        tool1 = MockTool("t1", {}, "")
        tool2 = MockTool("t2", {}, "")
        tm._tools = {"t1": tool1, "t2": tool2}
        tools = await tm.list_tools()
        assert len(tools) == 2
        assert tool1 in tools
        assert tool2 in tools


@pytest.mark.asyncio
class TestToolManagerListToolSummaries:
    async def test_list_tool_summaries_empty(self, tool_manager_fixture: ToolManager):
        tm = tool_manager_fixture
        tm._tools = {}
        summaries, meta = await tm.list_tool_summaries()
        assert summaries == []
        assert meta["total_items"] == 0

    async def test_list_tool_summaries_success(self, tool_manager_fixture: ToolManager):
        tm = tool_manager_fixture
        tool1_meta = {"name": "Tool One", "description_llm": "Desc One LLM", "tags": ["tagA"]}
        tool2_meta = {"name": "Tool Two", "description_human": "Desc Two Human", "tags": ["tagB"]}
        tool1 = MockTool("id1", tool1_meta, "")
        tool2 = MockTool("id2", tool2_meta, "")
        tm._tools = {"id1": tool1, "id2": tool2}

        summaries, meta = await tm.list_tool_summaries()
        assert len(summaries) == 2
        assert meta["total_items"] == 2
        assert any(s["identifier"] == "id1" and s["name"] == "Tool One" for s in summaries)
        assert any(s["identifier"] == "id2" and s["short_description"].startswith("Desc Two Human") for s in summaries)

    async def test_list_tool_summaries_pagination(self, tool_manager_fixture: ToolManager):
        tm = tool_manager_fixture
        tools_list = [MockTool(f"t{i}", {"name": f"Tool {i}"}, "") for i in range(5)]
        tm._tools = {t.identifier: t for t in tools_list}

        summaries_pg1, meta_pg1 = await tm.list_tool_summaries({"page": 1, "page_size": 2})
        assert len(summaries_pg1) == 2
        assert meta_pg1["current_page"] == 1
        assert meta_pg1["total_pages"] == 3
        assert meta_pg1["has_next"] is True

        summaries_pg3, meta_pg3 = await tm.list_tool_summaries({"page": 3, "page_size": "2"})
        assert len(summaries_pg3) == 1
        assert meta_pg3["current_page"] == 3
        assert meta_pg3["has_next"] is False

    async def test_list_tool_summaries_metadata_error(self, tool_manager_fixture: ToolManager, caplog):
        caplog.set_level(logging.ERROR)
        tm = tool_manager_fixture
        tool_error = MockTool("err_tool", {"raise_in_get_metadata": True}, "")
        tm._tools = {"err_tool": tool_error}
        
        summaries, _ = await tm.list_tool_summaries()
        assert len(summaries) == 0
        assert "Error getting metadata for tool 'err_tool': Metadata retrieval failed" in caplog.text


@pytest.mark.asyncio
class TestToolManagerGetFormattedToolDefinition:
    async def test_get_formatted_tool_definition_success(self, tool_manager_fixture: ToolManager, mock_plugin_manager_fixture: PluginManager):
        tm = tool_manager_fixture
        tool_meta = {"name": "My Tool"}
        mock_tool = MockTool("my_tool_id", tool_meta, "")
        tm._tools = {"my_tool_id": mock_tool}

        mock_formatter_instance = MockFormatter()
        mock_plugin_manager_fixture.get_plugin_instance.return_value = mock_formatter_instance

        formatted = await tm.get_formatted_tool_definition("my_tool_id", "mock_formatter_v1")
        assert formatted == "Formatted: My Tool"
        mock_plugin_manager_fixture.get_plugin_instance.assert_awaited_once_with("mock_formatter_v1")

    async def test_get_formatted_tool_definition_tool_not_found(self, tool_manager_fixture: ToolManager):
        tm = tool_manager_fixture
        formatted = await tm.get_formatted_tool_definition("no_such_tool", "any_formatter")
        assert formatted is None

    async def test_get_formatted_tool_definition_formatter_not_found(self, tool_manager_fixture: ToolManager, mock_plugin_manager_fixture: PluginManager, caplog):
        caplog.set_level(logging.WARNING)
        tm = tool_manager_fixture
        tm._tools = {"tool_exists": MockTool("tool_exists", {}, "")}
        mock_plugin_manager_fixture.get_plugin_instance.return_value = None

        formatted = await tm.get_formatted_tool_definition("tool_exists", "bad_formatter_id")
        assert formatted is None
        assert "DefinitionFormatter plugin 'bad_formatter_id' not found or invalid." in caplog.text

    async def test_get_formatted_tool_definition_formatter_fails(self, tool_manager_fixture: ToolManager, mock_plugin_manager_fixture: PluginManager, caplog):
        caplog.set_level(logging.ERROR)
        tm = tool_manager_fixture
        tm._tools = {"tool_for_fmt_fail": MockTool("tool_for_fmt_fail", {}, "")}
        
        mock_failing_formatter = MockFormatter()
        mock_failing_formatter.format = MagicMock(side_effect=RuntimeError("Format crashed"))
        mock_plugin_manager_fixture.get_plugin_instance.return_value = mock_failing_formatter

        formatted = await tm.get_formatted_tool_definition("tool_for_fmt_fail", "mock_formatter_v1")
        assert formatted is None
        assert "Error formatting tool 'tool_for_fmt_fail' with formatter 'mock_formatter_v1': Format crashed" in caplog.text