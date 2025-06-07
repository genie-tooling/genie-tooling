from typing import Any, Dict, Optional
from unittest.mock import ANY, AsyncMock

import pytest
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.definition_formatters.abc import DefinitionFormatter
from genie_tooling.lookup.service import (
    DEFAULT_INDEXING_FORMATTER_ID,
    DEFAULT_LOOKUP_PROVIDER_ID,
    ToolLookupService,
)
from genie_tooling.lookup.types import RankedToolResult
from genie_tooling.tool_lookup_providers.abc import ToolLookupProvider
from genie_tooling.tools.manager import ToolManager


@pytest.fixture
def mock_tool_manager_for_lookup(mocker) -> ToolManager:
    tm = mocker.MagicMock(spec=ToolManager)
    tm.list_tools = AsyncMock(return_value=[])
    return tm

@pytest.fixture
def mock_plugin_manager_for_lookup(mocker) -> PluginManager:
    pm = mocker.MagicMock(spec=PluginManager)
    pm.get_plugin_instance = AsyncMock()
    return pm

@pytest.fixture
def tool_lookup_service_fixture(
    mock_tool_manager_for_lookup: ToolManager,
    mock_plugin_manager_for_lookup: PluginManager
) -> ToolLookupService:
    return ToolLookupService(
        tool_manager=mock_tool_manager_for_lookup,
        plugin_manager=mock_plugin_manager_for_lookup,
        default_provider_id=DEFAULT_LOOKUP_PROVIDER_ID,
        default_indexing_formatter_id=DEFAULT_INDEXING_FORMATTER_ID
    )

@pytest.mark.asyncio
async def test_tool_lookup_service_find_tools_success_first_run_reindexes(
    tool_lookup_service_fixture: ToolLookupService,
    mock_tool_manager_for_lookup: ToolManager,
    mock_plugin_manager_for_lookup: PluginManager
):
    query = "find a tool"
    mock_tool1 = AsyncMock()
    mock_tool1.identifier = "tool1"
    mock_tool1.get_metadata = AsyncMock(return_value={"identifier": "tool1", "name": "Tool One", "description_llm": "Does one thing"})
    mock_tool_manager_for_lookup.list_tools.return_value = [mock_tool1]

    mock_formatter = AsyncMock(spec=DefinitionFormatter)
    mock_formatter.format.return_value = {"identifier": "tool1", "lookup_text_representation": "Formatted: Tool One"}
    mock_provider = AsyncMock(spec=ToolLookupProvider)
    expected_ranked_results = [RankedToolResult("tool1", 0.9, {"data": "formatted_tool1"})]
    mock_provider.find_tools = AsyncMock(return_value=expected_ranked_results)
    mock_provider.index_tools = AsyncMock()

    async def get_plugin_side_effect(plugin_id_req: str, config: Optional[Dict[str, Any]] = None):
        if plugin_id_req == DEFAULT_INDEXING_FORMATTER_ID:
            return mock_formatter
        if plugin_id_req == DEFAULT_LOOKUP_PROVIDER_ID:
            return mock_provider
        return None
    mock_plugin_manager_for_lookup.get_plugin_instance.side_effect = get_plugin_side_effect

    results = await tool_lookup_service_fixture.find_tools(query)

    assert results == expected_ranked_results
    mock_provider.index_tools.assert_awaited_once()
    mock_plugin_manager_for_lookup.get_plugin_instance.assert_any_call(DEFAULT_INDEXING_FORMATTER_ID, config=ANY)
    mock_plugin_manager_for_lookup.get_plugin_instance.assert_any_call(DEFAULT_LOOKUP_PROVIDER_ID, config=ANY)
