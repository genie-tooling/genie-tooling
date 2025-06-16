### tests/unit/lookup/test_tool_lookup_service.py
import logging
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.definition_formatters.abc import DefinitionFormatter
from genie_tooling.lookup.service import (
    DEFAULT_INDEXING_FORMATTER_ID,
    DEFAULT_LOOKUP_PROVIDER_ID,
    ToolLookupService,
)
from genie_tooling.lookup.types import RankedToolResult
from genie_tooling.observability.manager import InteractionTracingManager
from genie_tooling.tool_lookup_providers.abc import ToolLookupProvider
from genie_tooling.tools.abc import Tool
from genie_tooling.tools.manager import ToolManager

SERVICE_LOGGER_NAME = "genie_tooling.lookup.service"

# --- Mocks ---


@pytest.fixture()
def mock_tool_manager() -> MagicMock:
    tm = MagicMock(spec=ToolManager)
    tm.get_tool = AsyncMock()
    tm.list_tools = AsyncMock(return_value=[])
    return tm


@pytest.fixture()
def mock_tool_lookup_provider() -> MagicMock:
    provider = AsyncMock(spec=ToolLookupProvider)
    provider.plugin_id = "mock_lookup_provider_v1"
    provider.index_tools = AsyncMock()
    provider.add_tool = AsyncMock(return_value=True)
    provider.update_tool = AsyncMock(return_value=True)
    provider.remove_tool = AsyncMock(return_value=True)
    provider.find_tools = AsyncMock(return_value=[])
    return provider


@pytest.fixture()
def mock_definition_formatter() -> MagicMock:
    formatter = MagicMock(spec=DefinitionFormatter)
    formatter.plugin_id = "mock_formatter_v1"

    formatter.format = MagicMock()
    return formatter


@pytest.fixture()
def mock_tracing_manager_for_lookup(mocker) -> InteractionTracingManager:
    """Fixture to provide a mocked InteractionTracingManager."""
    tm = mocker.MagicMock(spec=InteractionTracingManager)
    tm.trace_event = AsyncMock()
    return tm


@pytest.fixture()
def mock_plugin_manager(
    mock_tool_lookup_provider, mock_definition_formatter
) -> MagicMock:
    pm = MagicMock(spec=PluginManager)

    async def get_instance_side_effect(plugin_id, config=None):
        if plugin_id == DEFAULT_LOOKUP_PROVIDER_ID:
            return mock_tool_lookup_provider
        if plugin_id == DEFAULT_INDEXING_FORMATTER_ID:
            return mock_definition_formatter
        return None

    pm.get_plugin_instance = AsyncMock(side_effect=get_instance_side_effect)
    return pm


@pytest.fixture()
def tool_lookup_service(
    mock_tool_manager, mock_plugin_manager, mock_tracing_manager_for_lookup
) -> ToolLookupService:
    return ToolLookupService(
        tool_manager=mock_tool_manager,
        plugin_manager=mock_plugin_manager,
        tracing_manager=mock_tracing_manager_for_lookup,
    )


# --- Test Cases ---


@pytest.mark.asyncio()
class TestToolLookupService:
    async def test_initialization(self, tool_lookup_service: ToolLookupService):
        assert tool_lookup_service._default_provider_id == DEFAULT_LOOKUP_PROVIDER_ID
        assert (
            tool_lookup_service._default_indexing_formatter_id
            == DEFAULT_INDEXING_FORMATTER_ID
        )
        assert not tool_lookup_service._is_indexed_map

    async def test_find_tools_triggers_full_reindex_on_first_call(
        self,
        tool_lookup_service: ToolLookupService,
        mock_tool_lookup_provider: MagicMock,
    ):
        with patch.object(
            tool_lookup_service, "reindex_all_tools", new_callable=AsyncMock
        ) as mock_reindex:
            await tool_lookup_service.find_tools("test query")
            mock_reindex.assert_awaited_once()

        with patch.object(
            tool_lookup_service, "reindex_all_tools", new_callable=AsyncMock
        ) as mock_reindex_2:
            tool_lookup_service._is_indexed_map[DEFAULT_LOOKUP_PROVIDER_ID] = True
            await tool_lookup_service.find_tools("another query")
            mock_reindex_2.assert_not_awaited()

    async def test_reindex_all_tools_success(
        self,
        tool_lookup_service: ToolLookupService,
        mock_tool_manager: MagicMock,
        mock_tool_lookup_provider: MagicMock,
        mock_definition_formatter: MagicMock,
        mock_tracing_manager_for_lookup: InteractionTracingManager,
    ):
        mock_tool1 = MagicMock(spec=Tool)
        mock_tool1.identifier = "tool1"
        mock_tool1.get_metadata = AsyncMock(
            return_value={"identifier": "tool1", "name": "Tool One"}
        )

        mock_tool2 = MagicMock(spec=Tool)
        mock_tool2.identifier = "tool2"
        mock_tool2.get_metadata = AsyncMock(
            return_value={"identifier": "tool2", "name": "Tool Two"}
        )

        mock_tool_manager.list_tools.return_value = [mock_tool1, mock_tool2]

        async def get_tool_side_effect(tool_id):
            if tool_id == "tool1":
                return mock_tool1
            if tool_id == "tool2":
                return mock_tool2
            return None

        mock_tool_manager.get_tool.side_effect = get_tool_side_effect

        # This allows _get_formatted_tool_data to succeed.
        mock_definition_formatter.format.return_value = {}

        success = await tool_lookup_service.reindex_all_tools(DEFAULT_LOOKUP_PROVIDER_ID)

        assert success is True
        assert tool_lookup_service._is_indexed_map[DEFAULT_LOOKUP_PROVIDER_ID] is True
        mock_tool_lookup_provider.index_tools.assert_awaited_once()
        call_args = mock_tool_lookup_provider.index_tools.call_args.kwargs
        assert len(call_args["tools_data"]) == 2

        assert (
            call_args["tools_data"][0]["lookup_text_representation"]
            == "Tool: Tool One. Description: "
        )
        mock_tracing_manager_for_lookup.trace_event.assert_any_call(
            "log.info", ANY, "ToolLookupService", ANY
        )

    async def test_reindex_all_tools_provider_load_fails(
        self,
        tool_lookup_service: ToolLookupService,
        mock_plugin_manager: MagicMock,
        mock_tracing_manager_for_lookup: InteractionTracingManager,
    ):
        mock_plugin_manager.get_plugin_instance.return_value = None

        success = await tool_lookup_service.reindex_all_tools("bad_provider")

        assert success is False
        assert tool_lookup_service._is_indexed_map.get("bad_provider") is False
        mock_tracing_manager_for_lookup.trace_event.assert_any_call(
            "log.error",
            {
                "message": "Provider 'bad_provider' not found or invalid for reindexing.",
            },
            "ToolLookupService",
            ANY,
        )

    async def test_add_or_update_tools_success(
        self,
        tool_lookup_service: ToolLookupService,
        mock_tool_manager: MagicMock,
        mock_tool_lookup_provider: MagicMock,
        mock_definition_formatter: MagicMock,
        mock_tracing_manager_for_lookup: InteractionTracingManager,
    ):
        mock_tool = MagicMock(spec=Tool)
        mock_tool.identifier = "new_tool"
        mock_tool.get_metadata = AsyncMock(
            return_value={"identifier": "new_tool", "name": "New Tool"}
        )
        mock_tool_manager.get_tool.return_value = mock_tool


        mock_definition_formatter.format.return_value = {}

        tool_lookup_service._is_indexed_map[DEFAULT_LOOKUP_PROVIDER_ID] = True

        await tool_lookup_service.add_or_update_tools(["new_tool"])

        mock_tool_lookup_provider.update_tool.assert_awaited_once()
        call_args = mock_tool_lookup_provider.update_tool.call_args
        assert call_args.args[0] == "new_tool"
        tool_data_arg = call_args.args[1]
        assert (
            tool_data_arg["lookup_text_representation"]
            == "Tool: New Tool. Description: "
        )

    async def test_remove_tools_success(
        self,
        tool_lookup_service: ToolLookupService,
        mock_tool_lookup_provider: MagicMock,
        mock_tracing_manager_for_lookup: InteractionTracingManager,
    ):
        tool_lookup_service._is_indexed_map[DEFAULT_LOOKUP_PROVIDER_ID] = True
        await tool_lookup_service.remove_tools(["tool_to_remove"])
        mock_tool_lookup_provider.remove_tool.assert_awaited_once_with(
            "tool_to_remove", config=ANY
        )

    async def test_find_tools_empty_query(
        self,
        tool_lookup_service: ToolLookupService,
        mock_tool_lookup_provider: MagicMock,
        mock_tracing_manager_for_lookup: InteractionTracingManager,
    ):
        results = await tool_lookup_service.find_tools("   ")
        assert results == []
        mock_tool_lookup_provider.find_tools.assert_not_called()

    async def test_find_tools_provider_search_fails(
        self,
        tool_lookup_service: ToolLookupService,
        mock_tool_lookup_provider: MagicMock,
        mock_tracing_manager_for_lookup: InteractionTracingManager,
    ):
        tool_lookup_service._is_indexed_map[DEFAULT_LOOKUP_PROVIDER_ID] = True
        mock_tool_lookup_provider.find_tools.side_effect = RuntimeError("Search failed")

        results = await tool_lookup_service.find_tools("test query")

        assert results == []
        mock_tracing_manager_for_lookup.trace_event.assert_any_call(
            "log.error",
            {
                "message": f"Error finding tools with provider '{DEFAULT_LOOKUP_PROVIDER_ID}': Search failed",
                "exc_info": True,
            },
            "ToolLookupService",
            ANY,
        )

    async def test_invalidate_all_indices(
        self,
        tool_lookup_service: ToolLookupService,
        mock_tracing_manager_for_lookup: InteractionTracingManager,
    ):
        tool_lookup_service._is_indexed_map["provider1"] = True
        tool_lookup_service._is_indexed_map["provider2"] = True

        await tool_lookup_service.invalidate_all_indices()

        assert not tool_lookup_service._is_indexed_map
        mock_tracing_manager_for_lookup.trace_event.assert_any_call(
            "log.info", ANY, "ToolLookupService", ANY
        )

    async def test_get_formatted_tool_data_formatter_returns_string(
        self,
        tool_lookup_service: ToolLookupService,
        mock_tool_manager: MagicMock,
        mock_definition_formatter: MagicMock,
        mock_tracing_manager_for_lookup: InteractionTracingManager,
    ):
        """Tests that a string returned by a formatter is correctly wrapped in a dict."""
        mock_tool = MagicMock(spec=Tool)
        mock_tool.identifier = "tool_for_string_format"
        mock_tool.get_metadata = AsyncMock(
            return_value={"identifier": "tool_for_string_format", "name": "String Tool"}
        )
        mock_tool_manager.get_tool.return_value = mock_tool
        mock_definition_formatter.format.return_value = "This is a formatted string"

        formatted_data = await tool_lookup_service._get_formatted_tool_data(
            "tool_for_string_format", DEFAULT_INDEXING_FORMATTER_ID
        )

        assert isinstance(formatted_data, dict)
        assert formatted_data["identifier"] == "tool_for_string_format"
        assert (
            formatted_data["lookup_text_representation"]
            == "This is a formatted string"
        )
        assert "_raw_metadata_snapshot" in formatted_data

    async def test_reindex_all_tools_no_tools_to_index(
        self,
        tool_lookup_service: ToolLookupService,
        mock_tool_manager: MagicMock,
        mock_tool_lookup_provider: MagicMock,
        mock_tracing_manager_for_lookup: InteractionTracingManager,
    ):
        """Tests that reindexing with no available tools completes gracefully."""
        mock_tool_manager.list_tools.return_value = []

        success = await tool_lookup_service.reindex_all_tools(DEFAULT_LOOKUP_PROVIDER_ID)

        assert success is True

        mock_tool_lookup_provider.index_tools.assert_awaited_once_with(
            tools_data=[], config=ANY
        )
        mock_tracing_manager_for_lookup.trace_event.assert_any_call(
            "log.warning", ANY, "ToolLookupService", ANY
        )

    async def test_find_tools_no_provider_configured(
        self,
        tool_lookup_service: ToolLookupService,
        mock_tracing_manager_for_lookup: InteractionTracingManager,
    ):
        """Tests that find_tools handles having no provider configured."""
        tool_lookup_service._default_provider_id = None
        results = await tool_lookup_service.find_tools("some query")
        assert results == []
        mock_tracing_manager_for_lookup.trace_event.assert_any_call(
            "log.warning",
            {"message": "No tool lookup provider specified or configured. Cannot find tools."},
            "ToolLookupService",
            ANY,
        )