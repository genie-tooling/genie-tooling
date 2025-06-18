### tests/unit/tools/test_tool_manager.py
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.definition_formatters.abc import DefinitionFormatter
from genie_tooling.observability.manager import InteractionTracingManager
from genie_tooling.tools.abc import Tool as ToolPlugin
from genie_tooling.tools.manager import ToolManager

TOOL_MANAGER_LOGGER_NAME = "genie_tooling.tools.manager"


class MockTool(ToolPlugin):
    _identifier_value: str
    _plugin_id_value: str

    def __init__(
        self,
        identifier_val: str,
        metadata: Dict[str, Any],
        execute_result: Any = "tool_executed",
        plugin_manager=None,
    ):
        self._identifier_value = identifier_val
        self._plugin_id_value = identifier_val
        self._metadata = metadata
        self._execute_result = execute_result
        self.setup_called_with_config: Optional[Dict[str, Any]] = None
        self.injected_plugin_manager = plugin_manager
        self.teardown_called: bool = False

    @property
    def identifier(self) -> str:
        return self._identifier_value

    @property
    def plugin_id(self) -> str:
        return self._plugin_id_value

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
    plugin_id_val: str = "mock_formatter_v1"  # Renamed to avoid conflict with property
    formatter_id: str = "mock_format_output_v1"
    description: str = "Mock Formatter"

    @property
    def plugin_id(self) -> str:
        return self.plugin_id_val

    def format(self, tool_metadata: Dict[str, Any]) -> Any:
        return f"Formatted: {tool_metadata.get('name', 'Unknown')}"

    async def setup(self, config=None):
        pass

    async def teardown(self):
        pass


@pytest.fixture()
def mock_plugin_manager_fixture(mocker) -> PluginManager:
    pm = mocker.MagicMock(spec=PluginManager)
    pm.list_discovered_plugin_classes = MagicMock(return_value={})
    pm.get_plugin_instance = AsyncMock(return_value=None)
    pm.get_plugin_source = MagicMock(return_value="mock_source")
    pm._discovered_plugin_classes = {}
    pm.discover_plugins = AsyncMock()
    return pm


@pytest.fixture()
def mock_tracing_manager_fixture(mocker) -> InteractionTracingManager:
    tm = mocker.MagicMock(spec=InteractionTracingManager)
    tm.trace_event = AsyncMock()
    return tm


@pytest.fixture()
def tool_manager_fixture(
    mock_plugin_manager_fixture: PluginManager,
    mock_tracing_manager_fixture: InteractionTracingManager,
) -> ToolManager:
    return ToolManager(
        plugin_manager=mock_plugin_manager_fixture,
        tracing_manager=mock_tracing_manager_fixture,
    )


@pytest.mark.asyncio()
class TestToolManagerInitializeTools:
    async def test_initialize_tools_no_discovered_plugins(
        self,
        mock_plugin_manager_fixture: PluginManager,
        mock_tracing_manager_fixture: InteractionTracingManager,
    ):
        pm = mock_plugin_manager_fixture
        pm.list_discovered_plugin_classes.return_value = {}
        pm._discovered_plugin_classes = {}
        tm = ToolManager(
            plugin_manager=pm, tracing_manager=mock_tracing_manager_fixture
        )
        await tm.initialize_tools(tool_configurations={})
        assert len(await tm.list_tools()) == 0
        mock_tracing_manager_fixture.trace_event.assert_any_call(
            event_name="log.info",
            data={
                "message": "ToolManager initialized. Loaded 0 explicitly configured class-based tools."
            },
            component="ToolManager",
            correlation_id=None,
        )

    async def test_initialize_tools_success(
        self,
        tool_manager_fixture: ToolManager,
        mock_plugin_manager_fixture: PluginManager,
    ):
        tm = tool_manager_fixture
        tool_id = "tool_alpha"
        mock_tool_alpha_class = MockTool
        mock_tool_alpha_instance = MockTool(tool_id, {"name": "Alpha"})
        tool_config_for_setup = {"config_key": "val"}

        mock_plugin_manager_fixture.list_discovered_plugin_classes.return_value = {
            tool_id: mock_tool_alpha_class
        }

        async def get_instance_side_effect(pid, config, **kwargs):
            if pid == tool_id:
                # The real PM would call setup, so we simulate it.
                await mock_tool_alpha_instance.setup(config)
                return mock_tool_alpha_instance
            return None

        mock_plugin_manager_fixture.get_plugin_instance.side_effect = (
            get_instance_side_effect
        )

        await tm.initialize_tools(tool_configurations={tool_id: tool_config_for_setup})

        loaded_tools = await tm.list_tools()
        assert len(loaded_tools) == 1
        assert loaded_tools[0].identifier == tool_id

        # FIX: The ToolManager no longer injects the plugin_manager into the config.
        # The assertion should check that the tool's setup received the config from
        # tool_configurations, without the extra key.
        assert mock_tool_alpha_instance.setup_called_with_config == tool_config_for_setup

    async def test_initialize_tools_tool_not_discovered(
        self,
        tool_manager_fixture: ToolManager,
        mock_plugin_manager_fixture: PluginManager,
        mock_tracing_manager_fixture: InteractionTracingManager,
    ):
        tm = tool_manager_fixture
        mock_plugin_manager_fixture.list_discovered_plugin_classes.return_value = {}
        await tm.initialize_tools(tool_configurations={"non_existent_tool": {}})
        assert len(await tm.list_tools()) == 0
        mock_tracing_manager_fixture.trace_event.assert_any_call(
            event_name="log.debug",
            data={
                "message": "Tool ID/alias 'non_existent_tool' not found as a discovered plugin class. It may be a function-based tool to be registered later."
            },
            component="ToolManager",
            correlation_id=None,
        )

    async def test_initialize_tools_alias_resolution(
        self,
        tool_manager_fixture: ToolManager,
        mock_plugin_manager_fixture: PluginManager,
    ):
        tm = tool_manager_fixture
        tool_alias = "calc"

        mock_calc_class = MockTool
        mock_calc_instance = MockTool("calc", {"name": "Calculator"})

        mock_plugin_manager_fixture.list_discovered_plugin_classes.return_value = {
            "calc": mock_calc_class
        }

        async def get_instance_side_effect_alias(pid, config):
            if pid == "calc":
                await mock_calc_instance.setup(config)
                return mock_calc_instance
            return None

        mock_plugin_manager_fixture.get_plugin_instance.side_effect = (
            get_instance_side_effect_alias
        )

        await tm.initialize_tools(tool_configurations={tool_alias: {}})

        loaded_tools = await tm.list_tools()
        assert len(loaded_tools) == 1
        assert loaded_tools[0].identifier == "calc"
        # FIX: The call to get_plugin_instance will now have the correct config dict (`{}`)
        # because ToolManager no longer adds the plugin_manager to it.
        mock_plugin_manager_fixture.get_plugin_instance.assert_awaited_once_with(
            tool_alias, config={}
        )

    async def test_initialize_tools_duplicate_identifier_warning(
        self,
        tool_manager_fixture: ToolManager,
        mock_plugin_manager_fixture: PluginManager,
        mock_tracing_manager_fixture: InteractionTracingManager,
    ):
        tm = tool_manager_fixture
        common_id = "common_tool_id"

        mock_tool_v1_class = type("MockToolV1", (MockTool,), {})
        mock_tool_v2_class = type("MockToolV2", (MockTool,), {})

        mock_tool_v1_instance = MockTool(common_id, {"name": "V1"})
        mock_tool_v2_instance = MockTool(common_id, {"name": "V2"})

        mock_plugin_manager_fixture.list_discovered_plugin_classes.return_value = {
            "plugin_v1_id": mock_tool_v1_class,
            "plugin_v2_id": mock_tool_v2_class,
        }

        async def get_instance_side_effect_dup(pid, config, **kwargs):
            if pid == "plugin_v1_id":
                await mock_tool_v1_instance.setup(config)
                return mock_tool_v1_instance
            if pid == "plugin_v2_id":
                await mock_tool_v2_instance.setup(config)
                return mock_tool_v2_instance
            return None

        mock_plugin_manager_fixture.get_plugin_instance.side_effect = (
            get_instance_side_effect_dup
        )

        await tm.initialize_tools(
            tool_configurations={"plugin_v1_id": {}, "plugin_v2_id": {}}
        )

        loaded_tools = await tm.list_tools()
        assert len(loaded_tools) == 1
        assert loaded_tools[0].identifier == common_id
        mock_tracing_manager_fixture.trace_event.assert_any_call(
            event_name="log.warning",
            data={
                "message": "Duplicate tool identifier 'common_tool_id' encountered from plugin ID/alias 'plugin_v2_id'. Source: 'mock_source'. Overwriting previous tool with same identifier."
            },
            component="ToolManager",
            correlation_id=None,
        )

    async def test_initialize_tools_plugin_init_needs_plugin_manager(
        self,
        tool_manager_fixture: ToolManager,
        mock_plugin_manager_fixture: PluginManager,
    ):
        tm = tool_manager_fixture
        tool_id = "pm_aware_tool"
        mock_pm_aware_tool_class = MockTool

        # The PluginManager will inject itself if the tool's __init__ requests it.
        # We simulate this by having the PM return an instance that has been "injected".
        mock_tool_instance = MockTool(
            identifier_val=tool_id,
            metadata={"name": "PMAware"},
            plugin_manager=mock_plugin_manager_fixture,
        )

        async def get_instance_side_effect_pm_aware(pid, config):
            if pid == tool_id:
                await mock_tool_instance.setup(config)
                return mock_tool_instance
            return None

        mock_plugin_manager_fixture.get_plugin_instance.side_effect = (
            get_instance_side_effect_pm_aware
        )
        mock_plugin_manager_fixture.list_discovered_plugin_classes.return_value = {
            tool_id: mock_pm_aware_tool_class
        }

        await tm.initialize_tools(tool_configurations={tool_id: {}})

        loaded_tool = await tm.get_tool(tool_id)
        assert loaded_tool is not None
        assert loaded_tool is mock_tool_instance
        assert loaded_tool.injected_plugin_manager is mock_plugin_manager_fixture  # type: ignore

    async def test_initialize_tools_plugin_instantiation_fails(
        self,
        tool_manager_fixture: ToolManager,
        mock_plugin_manager_fixture: PluginManager,
        mock_tracing_manager_fixture: InteractionTracingManager,
    ):
        tm = tool_manager_fixture
        tool_id = "fail_init_tool"

        mock_plugin_manager_fixture.list_discovered_plugin_classes.return_value = {
            tool_id: MagicMock
        }
        # The error now comes from get_plugin_instance, not ToolManager
        mock_plugin_manager_fixture.get_plugin_instance.side_effect = TypeError(
            "Cannot instantiate from PM"
        )

        await tm.initialize_tools(tool_configurations={tool_id: {}})
        assert len(await tm.list_tools()) == 0
        mock_tracing_manager_fixture.trace_event.assert_any_call(
            event_name="log.error",
            data={
                "message": "Error initializing tool plugin from ID/alias 'fail_init_tool' (class <class 'unittest.mock.MagicMock'>): Cannot instantiate from PM",
                "exc_info": True,
            },
            component="ToolManager",
            correlation_id=None,
        )

    async def test_initialize_tools_plugin_setup_fails(
        self,
        tool_manager_fixture: ToolManager,
        mock_plugin_manager_fixture: PluginManager,
        mock_tracing_manager_fixture: InteractionTracingManager,
    ):
        tm = tool_manager_fixture
        tool_id = "fail_setup_tool"
        mock_tool_class = MockTool

        mock_plugin_manager_fixture.list_discovered_plugin_classes.return_value = {
            tool_id: mock_tool_class
        }
        # PluginManager's get_plugin_instance is responsible for calling setup.
        # If setup fails there, get_plugin_instance should raise or return None.
        mock_plugin_manager_fixture.get_plugin_instance.side_effect = RuntimeError(
            "Setup failed from PM during get_instance"
        )

        await tm.initialize_tools(tool_configurations={tool_id: {}})
        assert len(await tm.list_tools()) == 0
        mock_tracing_manager_fixture.trace_event.assert_any_call(
            event_name="log.error",
            data={
                "message": "Error initializing tool plugin from ID/alias 'fail_setup_tool' (class <class 'tests.unit.tools.test_tool_manager.MockTool'>): Setup failed from PM during get_instance",
                "exc_info": True,
            },
            component="ToolManager",
            correlation_id=None,
        )


@pytest.mark.asyncio()
class TestToolManagerListFormatters:
    async def test_list_available_formatters_none_discovered(
        self,
        tool_manager_fixture: ToolManager,
        mock_plugin_manager_fixture: PluginManager,
    ):
        tm = tool_manager_fixture
        mock_plugin_manager_fixture.list_discovered_plugin_classes.return_value = {}
        formatters = await tm.list_available_formatters()
        assert formatters == []

    async def test_list_available_formatters_success(
        self,
        tool_manager_fixture: ToolManager,
        mock_plugin_manager_fixture: PluginManager,
    ):
        tm = tool_manager_fixture
        mock_formatter_instance = MockFormatter()

        mock_plugin_manager_fixture.list_discovered_plugin_classes.return_value = {
            mock_formatter_instance.plugin_id: MockFormatter
        }
        mock_plugin_manager_fixture.get_plugin_instance.return_value = (
            mock_formatter_instance
        )

        formatters = await tm.list_available_formatters()
        assert len(formatters) == 1
        assert formatters[0]["id"] == mock_formatter_instance.formatter_id
        assert formatters[0]["description"] == mock_formatter_instance.description
        assert formatters[0]["plugin_id"] == mock_formatter_instance.plugin_id

    async def test_list_available_formatters_not_a_formatter(
        self,
        tool_manager_fixture: ToolManager,
        mock_plugin_manager_fixture: PluginManager,
    ):
        tm = tool_manager_fixture
        mock_not_formatter_instance = MockTool("not_fmt", {}, "")

        mock_plugin_manager_fixture.list_discovered_plugin_classes.return_value = {
            "not_fmt_plugin_id": type(mock_not_formatter_instance)
        }
        mock_plugin_manager_fixture.get_plugin_instance.return_value = (
            mock_not_formatter_instance
        )

        formatters = await tm.list_available_formatters()
        assert len(formatters) == 0

    async def test_list_available_formatters_instantiation_fails(
        self,
        tool_manager_fixture: ToolManager,
        mock_plugin_manager_fixture: PluginManager,
        mock_tracing_manager_fixture: InteractionTracingManager,
    ):
        tm = tool_manager_fixture
        mock_plugin_manager_fixture.list_discovered_plugin_classes.return_value = {
            "fail_fmt_id": MockFormatter
        }
        mock_plugin_manager_fixture.get_plugin_instance.side_effect = RuntimeError(
            "Formatter init failed"
        )

        formatters = await tm.list_available_formatters()
        assert len(formatters) == 0
        mock_tracing_manager_fixture.trace_event.assert_awaited_with(
            event_name="log.debug",
            data={
                "message": "Could not instantiate or check plugin 'fail_fmt_id' as DefinitionFormatter: Formatter init failed"
            },
            component="ToolManager",
            correlation_id=None,
        )


@pytest.mark.asyncio()
class TestToolManagerGetTool:
    async def test_get_tool_exists(self, tool_manager_fixture: ToolManager):
        tm = tool_manager_fixture
        mock_tool = MockTool("tool_exists", {}, "")
        tm._tools = {"tool_exists": mock_tool}
        tool = await tm.get_tool("tool_exists")
        assert tool is mock_tool

    async def test_get_tool_not_exists(
        self,
        tool_manager_fixture: ToolManager,
        mock_tracing_manager_fixture: InteractionTracingManager,
    ):
        tm = tool_manager_fixture
        tm._tools = {}
        tool = await tm.get_tool("tool_not_found")
        assert tool is None
        mock_tracing_manager_fixture.trace_event.assert_awaited_with(
            event_name="log.debug",
            data={
                "message": "Tool with identifier 'tool_not_found' not found in ToolManager (not explicitly configured or loaded)."
            },
            component="ToolManager",
            correlation_id=None,
        )


@pytest.mark.asyncio()
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


@pytest.mark.asyncio()
class TestToolManagerListToolSummaries:
    async def test_list_tool_summaries_empty(self, tool_manager_fixture: ToolManager):
        tm = tool_manager_fixture
        tm._tools = {}
        summaries, meta = await tm.list_tool_summaries()
        assert summaries == []
        assert meta["total_items"] == 0

    async def test_list_tool_summaries_success(self, tool_manager_fixture: ToolManager):
        tm = tool_manager_fixture
        tool1_meta = {
            "name": "Tool One",
            "description_llm": "Desc One LLM",
            "tags": ["tagA"],
        }
        tool2_meta = {
            "name": "Tool Two",
            "description_human": "Desc Two Human",
            "tags": ["tagB"],
        }
        tool1 = MockTool("id1", tool1_meta, "")
        tool2 = MockTool("id2", tool2_meta, "")
        tm._tools = {"id1": tool1, "id2": tool2}

        summaries, meta = await tm.list_tool_summaries()
        assert len(summaries) == 2
        assert meta["total_items"] == 2
        assert any(s["identifier"] == "id1" and s["name"] == "Tool One" for s in summaries)
        assert any(
            s["identifier"] == "id2"
            and s["short_description"].startswith("Desc Two Human")
            for s in summaries
        )

    async def test_list_tool_summaries_pagination(
        self, tool_manager_fixture: ToolManager
    ):
        tm = tool_manager_fixture
        tools_list = [MockTool(f"t{i}", {"name": f"Tool {i}"}, "") for i in range(5)]
        tm._tools = {t.identifier: t for t in tools_list}

        summaries_pg1, meta_pg1 = await tm.list_tool_summaries(
            {"page": 1, "page_size": 2}
        )
        assert len(summaries_pg1) == 2
        assert meta_pg1["current_page"] == 1
        assert meta_pg1["total_pages"] == 3
        assert meta_pg1["has_next"] is True

        summaries_pg3, meta_pg3 = await tm.list_tool_summaries(
            {"page": 3, "page_size": "2"}
        )
        assert len(summaries_pg3) == 1
        assert meta_pg3["current_page"] == 3
        assert meta_pg3["has_next"] is False

    async def test_list_tool_summaries_metadata_error(
        self,
        tool_manager_fixture: ToolManager,
        mock_tracing_manager_fixture: InteractionTracingManager,
    ):
        tm = tool_manager_fixture
        tool_error = MockTool("err_tool", {"raise_in_get_metadata": True}, "")
        tm._tools = {"err_tool": tool_error}

        summaries, _ = await tm.list_tool_summaries()
        assert len(summaries) == 0
        mock_tracing_manager_fixture.trace_event.assert_awaited_with(
            event_name="log.error",
            data={
                "message": "Error getting metadata for tool 'err_tool': Metadata retrieval failed",
                "exc_info": True,
            },
            component="ToolManager",
            correlation_id=None,
        )


@pytest.mark.asyncio()
class TestToolManagerGetFormattedToolDefinition:
    async def test_get_formatted_tool_definition_success(
        self,
        tool_manager_fixture: ToolManager,
        mock_plugin_manager_fixture: PluginManager,
    ):
        tm = tool_manager_fixture
        tool_meta = {"name": "My Tool"}
        mock_tool = MockTool("my_tool_id", tool_meta, "")
        tm._tools = {"my_tool_id": mock_tool}

        mock_formatter_instance = MockFormatter()
        mock_plugin_manager_fixture.get_plugin_instance.return_value = (
            mock_formatter_instance
        )

        formatted = await tm.get_formatted_tool_definition(
            "my_tool_id", "mock_formatter_v1"
        )
        assert formatted == "Formatted: My Tool"
        mock_plugin_manager_fixture.get_plugin_instance.assert_awaited_once_with(
            "mock_formatter_v1"
        )

    async def test_get_formatted_tool_definition_tool_not_found(
        self, tool_manager_fixture: ToolManager
    ):
        tm = tool_manager_fixture
        formatted = await tm.get_formatted_tool_definition(
            "no_such_tool", "any_formatter"
        )
        assert formatted is None

    async def test_get_formatted_tool_definition_formatter_not_found(
        self,
        tool_manager_fixture: ToolManager,
        mock_plugin_manager_fixture: PluginManager,
        mock_tracing_manager_fixture: InteractionTracingManager,
    ):
        tm = tool_manager_fixture
        tm._tools = {"tool_exists": MockTool("tool_exists", {}, "")}
        mock_plugin_manager_fixture.get_plugin_instance.return_value = None

        formatted = await tm.get_formatted_tool_definition(
            "tool_exists", "bad_formatter_id"
        )
        assert formatted is None
        mock_tracing_manager_fixture.trace_event.assert_awaited_with(
            event_name="log.warning",
            data={
                "message": "DefinitionFormatter plugin 'bad_formatter_id' not found or invalid."
            },
            component="ToolManager",
            correlation_id=None,
        )

    async def test_get_formatted_tool_definition_formatter_fails(
        self,
        tool_manager_fixture: ToolManager,
        mock_plugin_manager_fixture: PluginManager,
        mock_tracing_manager_fixture: InteractionTracingManager,
    ):
        tm = tool_manager_fixture
        tm._tools = {"tool_for_fmt_fail": MockTool("tool_for_fmt_fail", {}, "")}

        mock_failing_formatter = MockFormatter()
        mock_failing_formatter.format = MagicMock(
            side_effect=RuntimeError("Format crashed")
        )
        mock_plugin_manager_fixture.get_plugin_instance.return_value = (
            mock_failing_formatter
        )

        formatted = await tm.get_formatted_tool_definition(
            "tool_for_fmt_fail", "mock_formatter_v1"
        )
        assert formatted is None
        mock_tracing_manager_fixture.trace_event.assert_awaited_with(
            event_name="log.error",
            data={
                "message": "Error formatting tool 'tool_for_fmt_fail' with formatter 'mock_formatter_v1': Format crashed",
                "exc_info": True,
            },
            component="ToolManager",
            correlation_id=None,
        )