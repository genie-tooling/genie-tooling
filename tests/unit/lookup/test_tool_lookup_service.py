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
from genie_tooling.observability.manager import InteractionTracingManager
from genie_tooling.tool_lookup_providers.abc import ToolLookupProvider
from genie_tooling.tools.abc import Tool
from genie_tooling.tools.manager import ToolManager

SERVICE_LOGGER_NAME = "genie_tooling.lookup.service"

@pytest.fixture
def mock_tool_manager() -> MagicMock:
    tm = MagicMock(spec=ToolManager)
    tm.get_tool = AsyncMock()
    tm.list_tools = AsyncMock(return_value=[])
    return tm

@pytest.fixture
def mock_tool_lookup_provider() -> MagicMock:
    provider = AsyncMock(spec=ToolLookupProvider)
    provider.plugin_id = "mock_lookup_provider_v1"
    provider.index_tools = AsyncMock()
    provider.add_tool = AsyncMock(return_value=True)
    provider.update_tool = AsyncMock(return_value=True)
    provider.remove_tool = AsyncMock(return_value=True)
    provider.find_tools = AsyncMock(return_value=[])
    return provider

@pytest.fixture
def mock_definition_formatter() -> MagicMock:
    formatter = MagicMock(spec=DefinitionFormatter)
    formatter.plugin_id = "mock_formatter_v1"
    # This side_effect is the default behavior for the fixture
    formatter.format = MagicMock(side_effect=lambda tool_metadata: {"identifier": tool_metadata["identifier"], "name": tool_metadata["name"], "lookup_text_representation": f"Formatted: {tool_metadata['name']}", "_raw_metadata_snapshot": {}})
    return formatter

@pytest.fixture
def mock_tracing_manager_for_lookup() -> MagicMock:
    tm = MagicMock(spec=InteractionTracingManager)
    tm.trace_event = AsyncMock()
    return tm

@pytest.fixture
def mock_plugin_manager(mock_tool_lookup_provider, mock_definition_formatter) -> MagicMock:
    pm = MagicMock(spec=PluginManager)
    async def get_instance_side_effect(plugin_id, config=None):
        if plugin_id == DEFAULT_LOOKUP_PROVIDER_ID:
            return mock_tool_lookup_provider
        if plugin_id == DEFAULT_INDEXING_FORMATTER_ID:
            return mock_definition_formatter
        return None
    pm.get_plugin_instance = AsyncMock(side_effect=get_instance_side_effect)
    return pm

@pytest.fixture
def tool_lookup_service(mock_tool_manager, mock_plugin_manager, mock_tracing_manager_for_lookup) -> ToolLookupService:
    return ToolLookupService(
        tool_manager=mock_tool_manager,
        plugin_manager=mock_plugin_manager,
        tracing_manager=mock_tracing_manager_for_lookup
    )

@pytest.mark.asyncio
class TestToolLookupService:
    async def test_reindex_all_tools_provider_load_fails(self, tool_lookup_service: ToolLookupService, mock_plugin_manager: MagicMock, mock_tracing_manager_for_lookup: MagicMock):
        mock_plugin_manager.get_plugin_instance.return_value = None
        success = await tool_lookup_service.reindex_all_tools("bad_provider")
        assert success is False
        assert tool_lookup_service._is_indexed_map["bad_provider"] is False
        mock_tracing_manager_for_lookup.trace_event.assert_any_call("log.error", {"message": "Provider 'bad_provider' not found or invalid for reindexing."}, "ToolLookupService", None)

    async def test_find_tools_provider_search_fails(self, tool_lookup_service: ToolLookupService, mock_tool_lookup_provider: MagicMock, mock_tracing_manager_for_lookup: MagicMock):
        tool_lookup_service._is_indexed_map[DEFAULT_LOOKUP_PROVIDER_ID] = True
        mock_tool_lookup_provider.find_tools.side_effect = RuntimeError("Search failed")
        results = await tool_lookup_service.find_tools("test query")
        assert results == []
        mock_tracing_manager_for_lookup.trace_event.assert_any_call("log.error", {"message": f"Error finding tools with provider '{DEFAULT_LOOKUP_PROVIDER_ID}': Search failed", "exc_info": True}, "ToolLookupService", None)

    async def test_invalidate_all_indices(self, tool_lookup_service: ToolLookupService, mock_tracing_manager_for_lookup: MagicMock):
        tool_lookup_service._is_indexed_map["provider1"] = True
        tool_lookup_service._is_indexed_map["provider2"] = True
        await tool_lookup_service.invalidate_all_indices()
        assert not tool_lookup_service._is_indexed_map
        mock_tracing_manager_for_lookup.trace_event.assert_any_call("log.info", {"message": "All provider indices invalidated."}, "ToolLookupService", None)

@pytest.mark.asyncio
async def test_trace_without_tracing_manager(mock_tool_manager, mock_plugin_manager, caplog):
    """Test that _trace falls back to standard logger when tracing_manager is None."""
    caplog.set_level(logging.INFO, logger=SERVICE_LOGGER_NAME)
    service_no_trace = ToolLookupService(
        tool_manager=mock_tool_manager,
        plugin_manager=mock_plugin_manager,
        tracing_manager=None
    )
    await service_no_trace._trace("test.event", {"message": "log this"}, level="info")
    assert "test.event | log this" in caplog.text

@pytest.mark.asyncio
async def test_get_formatted_tool_data_formatter_returns_string(tool_lookup_service, mock_tool_manager, mock_definition_formatter):
    """Test handling when a formatter returns a plain string."""
    mock_tool = AsyncMock(spec=Tool)
    mock_tool.identifier = "string_tool"
    mock_tool.get_metadata = AsyncMock(return_value={"identifier": "string_tool", "name": "String Tool"})
    mock_tool_manager.get_tool.return_value = mock_tool

    # FIX: Clear the fixture's side_effect before setting a new return_value
    mock_definition_formatter.format.side_effect = None
    mock_definition_formatter.format.return_value = "Just a simple string representation"

    formatted_data = await tool_lookup_service._get_formatted_tool_data("string_tool", DEFAULT_INDEXING_FORMATTER_ID)
    assert formatted_data is not None
    assert formatted_data["identifier"] == "string_tool"
    assert formatted_data["lookup_text_representation"] == "Just a simple string representation"

@pytest.mark.asyncio
async def test_reindex_all_tools_no_tools_to_index(tool_lookup_service, mock_tool_manager, mock_tool_lookup_provider, mock_tracing_manager_for_lookup):
    """Test reindexing when the tool manager returns no tools."""
    mock_tool_manager.list_tools.return_value = []
    success = await tool_lookup_service.reindex_all_tools(DEFAULT_LOOKUP_PROVIDER_ID)
    assert success is True
    mock_tool_lookup_provider.index_tools.assert_awaited_once_with(tools_data=[], config=ANY)
    mock_tracing_manager_for_lookup.trace_event.assert_any_call(
        "log.warning", {"message": f"No tool data could be formatted for provider '{DEFAULT_LOOKUP_PROVIDER_ID}'. Index will be empty."}, "ToolLookupService", None
    )

@pytest.mark.asyncio
async def test_find_tools_empty_query(tool_lookup_service, mock_tracing_manager_for_lookup):
    """Test that find_tools returns an empty list for an empty query."""
    results = await tool_lookup_service.find_tools("")
    assert results == []
    mock_tracing_manager_for_lookup.trace_event.assert_any_call(
        "log.debug", {"message": "Empty query provided. Returning no results."}, "ToolLookupService", None
    )

@pytest.mark.asyncio
async def test_find_tools_no_provider_configured(mock_tool_manager, mock_plugin_manager, mock_tracing_manager_for_lookup):
    """Test find_tools when no default provider ID is set."""
    service_no_provider = ToolLookupService(
        tool_manager=mock_tool_manager,
        plugin_manager=mock_plugin_manager,
        default_provider_id=None,
        tracing_manager=mock_tracing_manager_for_lookup
    )
    results = await service_no_provider.find_tools("any query")
    assert results == []
    mock_tracing_manager_for_lookup.trace_event.assert_any_call(
        "log.warning", {"message": "No tool lookup provider specified or configured. Cannot find tools."}, "ToolLookupService", None
    )
