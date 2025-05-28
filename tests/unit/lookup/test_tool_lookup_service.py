### tests/unit/lookup/test_tool_lookup_service.py
"""Unit tests for the ToolLookupService."""
import logging
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.lookup.providers.abc import (
    ToolLookupProvider as ToolLookupProviderPlugin,
)
from genie_tooling.lookup.service import (
    ToolLookupService,
)
from genie_tooling.lookup.types import RankedToolResult
from genie_tooling.tools.abc import Tool as ToolPlugin
from genie_tooling.tools.formatters.abc import (
    DefinitionFormatter as DefinitionFormatterPlugin,
)
from genie_tooling.tools.manager import ToolManager

# --- Mock Components ---

class MockToolForLookup(ToolPlugin):
    identifier: str
    plugin_id: str
    raise_in_get_metadata: bool = False

    def __init__(self, identifier: str, name: str = "Mock Tool", desc_llm: str = "Mock LLM Desc"):
        self.identifier = identifier
        self.plugin_id = identifier
        self._metadata = {
            "identifier": identifier,
            "name": name,
            "description_llm": desc_llm,
            "description_human": "Mock Human Desc",
            "input_schema": {}, "output_schema": {}, "tags": ["mock"]
        }
        self.raise_in_get_metadata = False
    async def get_metadata(self) -> Dict[str, Any]:
        if self.raise_in_get_metadata:
            raise RuntimeError("Metadata retrieval failed for lookup")
        return self._metadata
    async def execute(self, params, key_provider, context=None) -> Any: return "executed"
    async def setup(self, config=None): pass
    async def teardown(self): pass


class MockLookupProvider(ToolLookupProviderPlugin):
    plugin_id: str = "mock_lookup_provider_v1"
    description: str = "Mock Lookup Provider"

    _indexed_data: List[Dict[str, Any]] = []
    _find_results: List[RankedToolResult] = []
    index_tools_should_raise: bool = False
    find_tools_should_raise: bool = False

    async def index_tools(self, tools_data: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None) -> None:
        if self.index_tools_should_raise:
            raise RuntimeError("Provider index_tools failed")
        self._indexed_data = tools_data

    async def find_tools(self, natural_language_query: str, top_k: int = 5, config: Optional[Dict[str, Any]] = None) -> List[RankedToolResult]:
        if self.find_tools_should_raise:
            raise RuntimeError("Provider find_tools failed")
        return self._find_results[:top_k]

    def set_find_results(self, results: List[RankedToolResult]):
        self._find_results = results

    def get_indexed_data(self) -> List[Dict[str, Any]]:
        return self._indexed_data

    def reset_mock_behavior(self):
        self.index_tools_should_raise = False
        self.find_tools_should_raise = False
        self._find_results = []
        self._indexed_data = []


class MockLookupFormatter(DefinitionFormatterPlugin):
    plugin_id: str = "mock_lookup_formatter_v1"
    formatter_id: str = "mock_format_for_lookup_v1"
    description: str = "Mock Lookup Formatter"
    format_should_raise: bool = False
    format_returns_invalid_type: bool = False


    def format(self, tool_metadata: Dict[str, Any]) -> Any:
        if self.format_should_raise:
            raise RuntimeError("Formatter format method failed")
        if self.format_returns_invalid_type:
            return 123 # Invalid type
        return {
            "identifier": tool_metadata["identifier"],
            "lookup_text_representation": f"Formatted: {tool_metadata['name']} - {tool_metadata['description_llm']}",
            "_raw_metadata_snapshot": tool_metadata
        }

    def reset_mock_behavior(self):
        self.format_should_raise = False
        self.format_returns_invalid_type = False

# --- Fixtures ---

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
    service = ToolLookupService(
        tool_manager=mock_tool_manager_for_lookup,
        plugin_manager=mock_plugin_manager_for_lookup,
        default_provider_id="mock_lookup_provider_v1",
        default_indexing_formatter_id="mock_lookup_formatter_v1"
    )
    # Reset mocks on provider/formatter if they are stateful and reused across tests implicitly
    # For this setup, new mocks are created or side_effects redefined per test, so it's cleaner.
    return service

# --- Test Cases ---

@pytest.mark.asyncio
async def test_tool_lookup_service_find_tools_success_first_run_reindexes(
    tool_lookup_service_fixture: ToolLookupService,
    mock_tool_manager_for_lookup: ToolManager,
    mock_plugin_manager_for_lookup: PluginManager
):
    query = "find a tool"
    mock_tool1 = MockToolForLookup("tool1", "Tool One", "Does one thing")
    mock_tool_manager_for_lookup.list_tools.return_value = [mock_tool1]

    mock_formatter = MockLookupFormatter()
    mock_provider = MockLookupProvider()

    expected_ranked_results = [RankedToolResult("tool1", 0.9, {"data": "formatted_tool1"})]
    mock_provider.set_find_results(expected_ranked_results)

    async def get_plugin_side_effect(plugin_id_req: str, config: Optional[Dict[str, Any]] = None, **kwargs):
        if plugin_id_req == "mock_lookup_formatter_v1": return mock_formatter
        if plugin_id_req == "mock_lookup_provider_v1": return mock_provider
        return None
    mock_plugin_manager_for_lookup.get_plugin_instance.side_effect = get_plugin_side_effect

    results = await tool_lookup_service_fixture.find_tools(query)

    assert results == expected_ranked_results
    assert len(mock_provider.get_indexed_data()) == 1
    indexed_item = mock_provider.get_indexed_data()[0]
    assert indexed_item["identifier"] == "tool1"
    assert "Formatted: Tool One - Does one thing" in indexed_item["lookup_text_representation"]

    # Check that provider was called with plugin_manager in its config
    # get_plugin_instance for provider will be called twice: once for reindex, once for find.
    # Both should have plugin_manager.
    expected_config_with_pm = {"plugin_manager": mock_plugin_manager_for_lookup}

    # Check calls to get_plugin_instance
    mock_plugin_manager_for_lookup.get_plugin_instance.assert_any_call("mock_lookup_formatter_v1")

    provider_calls = [
        c for c in mock_plugin_manager_for_lookup.get_plugin_instance.call_args_list
        if c.args[0] == "mock_lookup_provider_v1"
    ]
    assert len(provider_calls) >= 1 # Could be 1 or 2 depending on internal logic if provider instance is reused
    for p_call in provider_calls:
         assert p_call.kwargs.get("config") == expected_config_with_pm


@pytest.mark.asyncio
async def test_tool_lookup_service_find_tools_index_already_valid(
    tool_lookup_service_fixture: ToolLookupService,
    mock_plugin_manager_for_lookup: PluginManager
):
    query = "find a tool again"
    mock_provider = MockLookupProvider()

    expected_ranked_results = [RankedToolResult("tool2", 0.88)]
    mock_provider.set_find_results(expected_ranked_results)

    tool_lookup_service_fixture._index_validity_map["mock_lookup_provider_v1"] = True

    mock_plugin_manager_for_lookup.get_plugin_instance.side_effect = lambda plugin_id_req, config=None, **kwargs: (
        mock_provider if plugin_id_req == "mock_lookup_provider_v1" else None
    )

    reindex_spy = AsyncMock(wraps=tool_lookup_service_fixture.reindex_tools_for_provider)
    tool_lookup_service_fixture.reindex_tools_for_provider = reindex_spy

    results = await tool_lookup_service_fixture.find_tools(query)
    assert results == expected_ranked_results
    reindex_spy.assert_not_awaited()
    expected_provider_config = {"plugin_manager": mock_plugin_manager_for_lookup}
    mock_plugin_manager_for_lookup.get_plugin_instance.assert_called_once_with("mock_lookup_provider_v1", config=expected_provider_config)


@pytest.mark.asyncio
async def test_tool_lookup_service_find_tools_provider_not_found(
    tool_lookup_service_fixture: ToolLookupService,
    mock_plugin_manager_for_lookup: PluginManager,
    caplog
):
    caplog.set_level(logging.ERROR)
    mock_plugin_manager_for_lookup.get_plugin_instance.return_value = None
    results = await tool_lookup_service_fixture.find_tools("any query")
    assert results == []
    assert "Tool lookup provider 'mock_lookup_provider_v1' not found or invalid." in caplog.text


@pytest.mark.asyncio
async def test_tool_lookup_service_find_tools_reindex_fails_due_to_provider_index_error(
    tool_lookup_service_fixture: ToolLookupService,
    mock_tool_manager_for_lookup: ToolManager,
    mock_plugin_manager_for_lookup: PluginManager,
    caplog
):
    caplog.set_level(logging.ERROR)
    mock_tool_manager_for_lookup.list_tools.return_value = [MockToolForLookup("tool1")]
    mock_formatter = MockLookupFormatter()
    mock_provider = MockLookupProvider()
    mock_provider.index_tools_should_raise = True # Simulate provider error

    async def get_plugin_side_effect(plugin_id_req: str, config=None, **kwargs):
        if plugin_id_req == "mock_lookup_formatter_v1": return mock_formatter
        if plugin_id_req == "mock_lookup_provider_v1": return mock_provider
        return None
    mock_plugin_manager_for_lookup.get_plugin_instance.side_effect = get_plugin_side_effect

    results = await tool_lookup_service_fixture.find_tools("any query")
    assert results == []
    assert tool_lookup_service_fixture._index_validity_map.get("mock_lookup_provider_v1") is False
    assert "Error during reindexing with provider 'mock_lookup_provider_v1': Provider index_tools failed" in caplog.text
    assert "Re-indexing failed for 'mock_lookup_provider_v1'. Cannot perform lookup." in caplog.text


@pytest.mark.asyncio
async def test_tool_lookup_service_reindex_formatter_not_found(
    tool_lookup_service_fixture: ToolLookupService,
    mock_tool_manager_for_lookup: ToolManager,
    mock_plugin_manager_for_lookup: PluginManager,
    caplog
):
    caplog.set_level(logging.ERROR)
    mock_tool_manager_for_lookup.list_tools.return_value = [MockToolForLookup("tool1")]
    mock_provider = MockLookupProvider()

    mock_plugin_manager_for_lookup.get_plugin_instance.side_effect = lambda plugin_id_req, config=None, **kwargs: (
        mock_provider if plugin_id_req == "mock_lookup_provider_v1" else
        None if plugin_id_req == tool_lookup_service_fixture._default_indexing_formatter_id else
        None
    )

    success = await tool_lookup_service_fixture.reindex_tools_for_provider("mock_lookup_provider_v1")
    assert success is False
    assert tool_lookup_service_fixture._index_validity_map.get("mock_lookup_provider_v1") is False
    assert f"Indexing formatter '{tool_lookup_service_fixture._default_indexing_formatter_id}' not found or invalid." in caplog.text

    # Ensure find_tools also fails if reindex failed due to formatter
    async def get_plugin_side_effect_find_path(plugin_id_req: str, config=None, **kwargs):
        if plugin_id_req == "mock_lookup_provider_v1": return mock_provider
        if plugin_id_req == tool_lookup_service_fixture._default_indexing_formatter_id: return None
        return None
    mock_plugin_manager_for_lookup.get_plugin_instance.side_effect = get_plugin_side_effect_find_path
    tool_lookup_service_fixture._index_validity_map.clear()

    results = await tool_lookup_service_fixture.find_tools("query after formatter fail")
    assert results == []
    assert tool_lookup_service_fixture._index_validity_map.get("mock_lookup_provider_v1") is False


@pytest.mark.asyncio
async def test_tool_lookup_service_find_tools_no_tools_available(
    tool_lookup_service_fixture: ToolLookupService,
    mock_tool_manager_for_lookup: ToolManager,
    mock_plugin_manager_for_lookup: PluginManager,
    caplog
):
    caplog.set_level(logging.DEBUG)
    mock_tool_manager_for_lookup.list_tools.return_value = []
    mock_formatter = MockLookupFormatter()
    mock_provider = MockLookupProvider()
    mock_provider.set_find_results([])

    async def get_plugin_side_effect(plugin_id_req: str, config=None, **kwargs):
        if plugin_id_req == "mock_lookup_formatter_v1": return mock_formatter
        if plugin_id_req == "mock_lookup_provider_v1": return mock_provider
        return None
    mock_plugin_manager_for_lookup.get_plugin_instance.side_effect = get_plugin_side_effect

    results = await tool_lookup_service_fixture.find_tools("any query")
    assert results == []
    assert len(mock_provider.get_indexed_data()) == 0
    assert tool_lookup_service_fixture._index_validity_map.get("mock_lookup_provider_v1") is True
    assert "No tools available to format for indexing." in caplog.text # From _get_formatted_tool_data

    results_again = await tool_lookup_service_fixture.find_tools("any query again")
    assert results_again == []


@pytest.mark.asyncio
async def test_tool_lookup_service_invalidate_index(tool_lookup_service_fixture: ToolLookupService):
    provider_id = "mock_lookup_provider_v1"
    tool_lookup_service_fixture._index_validity_map[provider_id] = True
    assert tool_lookup_service_fixture._index_validity_map[provider_id] is True

    tool_lookup_service_fixture.invalidate_index(provider_id)
    assert tool_lookup_service_fixture._index_validity_map[provider_id] is False

    tool_lookup_service_fixture._index_validity_map[provider_id] = True
    tool_lookup_service_fixture._index_validity_map["other_provider"] = True
    tool_lookup_service_fixture.invalidate_index()
    assert tool_lookup_service_fixture._index_validity_map[provider_id] is False
    assert tool_lookup_service_fixture._index_validity_map["other_provider"] is False

@pytest.mark.asyncio
async def test_tool_lookup_service_find_tools_empty_query(tool_lookup_service_fixture: ToolLookupService, caplog):
    caplog.set_level(logging.DEBUG)
    results = await tool_lookup_service_fixture.find_tools("")
    assert results == []
    assert "Empty query provided. Returning no results." in caplog.text

    results_whitespace = await tool_lookup_service_fixture.find_tools("   ")
    assert results_whitespace == []


@pytest.mark.asyncio
async def test_get_formatted_tool_data_handles_string_formatter_output(
    tool_lookup_service_fixture: ToolLookupService,
    mock_tool_manager_for_lookup: ToolManager,
    mock_plugin_manager_for_lookup: PluginManager
):
    mock_tool1 = MockToolForLookup("tool_str_fmt", "String Tool")
    mock_tool_manager_for_lookup.list_tools.return_value = [mock_tool1]

    mock_string_formatter = MagicMock(spec=DefinitionFormatterPlugin)
    mock_string_formatter.format = MagicMock(return_value="Just a string for tool_str_fmt")

    mock_provider = MockLookupProvider()

    async def get_plugin_side_effect_str_fmt(plugin_id_req: str, config=None, **kwargs):
        if plugin_id_req == "mock_lookup_formatter_v1": return mock_string_formatter
        if plugin_id_req == "mock_lookup_provider_v1": return mock_provider
        return None
    mock_plugin_manager_for_lookup.get_plugin_instance.side_effect = get_plugin_side_effect_str_fmt

    formatted_data_list = await tool_lookup_service_fixture._get_formatted_tool_data_for_provider(mock_provider)
    assert formatted_data_list is not None
    assert len(formatted_data_list) == 1
    item = formatted_data_list[0]
    assert item["identifier"] == "tool_str_fmt"
    assert item["lookup_text_representation"] == "Just a string for tool_str_fmt"
    assert "_raw_metadata_snapshot" in item
    assert item["_raw_metadata_snapshot"]["name"] == "String Tool"

@pytest.mark.asyncio
async def test_get_formatted_tool_data_handles_tool_get_metadata_error(
    tool_lookup_service_fixture: ToolLookupService,
    mock_tool_manager_for_lookup: ToolManager,
    mock_plugin_manager_for_lookup: PluginManager,
    caplog
):
    caplog.set_level(logging.ERROR)
    mock_error_tool = MockToolForLookup("error_tool")
    mock_error_tool.raise_in_get_metadata = True
    mock_tool_manager_for_lookup.list_tools.return_value = [mock_error_tool]
    mock_formatter = MockLookupFormatter()
    mock_provider = MockLookupProvider()

    mock_plugin_manager_for_lookup.get_plugin_instance.side_effect = lambda pid, config=None, **kwargs: mock_formatter if pid == "mock_lookup_formatter_v1" else (mock_provider if pid == "mock_lookup_provider_v1" else None)

    formatted_data = await tool_lookup_service_fixture._get_formatted_tool_data_for_provider(mock_provider)
    assert formatted_data == [] # Should skip the tool that errored
    assert "Error formatting tool 'error_tool' for indexing" in caplog.text
    assert "Metadata retrieval failed for lookup" in caplog.text


@pytest.mark.asyncio
async def test_get_formatted_tool_data_handles_formatter_format_error(
    tool_lookup_service_fixture: ToolLookupService,
    mock_tool_manager_for_lookup: ToolManager,
    mock_plugin_manager_for_lookup: PluginManager,
    caplog
):
    caplog.set_level(logging.ERROR)
    mock_tool1 = MockToolForLookup("tool1")
    mock_tool_manager_for_lookup.list_tools.return_value = [mock_tool1]
    mock_formatter = MockLookupFormatter()
    mock_formatter.format_should_raise = True
    mock_provider = MockLookupProvider()

    mock_plugin_manager_for_lookup.get_plugin_instance.side_effect = lambda pid, config=None, **kwargs: mock_formatter if pid == "mock_lookup_formatter_v1" else (mock_provider if pid == "mock_lookup_provider_v1" else None)

    formatted_data = await tool_lookup_service_fixture._get_formatted_tool_data_for_provider(mock_provider)
    assert formatted_data == []
    assert "Error formatting tool 'tool1' for indexing" in caplog.text
    assert "Formatter format method failed" in caplog.text

@pytest.mark.asyncio
async def test_get_formatted_tool_data_handles_formatter_invalid_return_type(
    tool_lookup_service_fixture: ToolLookupService,
    mock_tool_manager_for_lookup: ToolManager,
    mock_plugin_manager_for_lookup: PluginManager,
    caplog
):
    caplog.set_level(logging.WARNING)
    mock_tool1 = MockToolForLookup("tool1")
    mock_tool_manager_for_lookup.list_tools.return_value = [mock_tool1]
    mock_formatter = MockLookupFormatter()
    mock_formatter.format_returns_invalid_type = True
    mock_provider = MockLookupProvider()

    mock_plugin_manager_for_lookup.get_plugin_instance.side_effect = lambda pid, config=None, **kwargs: mock_formatter if pid == "mock_lookup_formatter_v1" else (mock_provider if pid == "mock_lookup_provider_v1" else None)

    formatted_data = await tool_lookup_service_fixture._get_formatted_tool_data_for_provider(mock_provider)
    assert formatted_data == []
    assert "produced unexpected data type: <class 'int'>. Skipping." in caplog.text


@pytest.mark.asyncio
async def test_find_tools_uses_override_ids_and_config(
    tool_lookup_service_fixture: ToolLookupService,
    mock_tool_manager_for_lookup: ToolManager,
    mock_plugin_manager_for_lookup: PluginManager
):
    query = "custom find"
    override_provider_id = "override_provider"
    override_formatter_id = "override_formatter"
    override_provider_config = {"custom_key": "custom_value"}

    mock_tool1 = MockToolForLookup("tool_override", "Override Tool", "For override test")
    mock_tool_manager_for_lookup.list_tools.return_value = [mock_tool1]

    mock_override_formatter = MockLookupFormatter()
    mock_override_provider = MockLookupProvider()
    expected_results = [RankedToolResult("tool_override", 0.7)]
    mock_override_provider.set_find_results(expected_results)

    async def get_plugin_override_effect(plugin_id_req: str, config: Optional[Dict[str, Any]] = None, **kwargs):
        if plugin_id_req == override_formatter_id: return mock_override_formatter
        if plugin_id_req == override_provider_id:
            # Assert that the override_provider_config is passed during instantiation
            assert config is not None
            assert config.get("custom_key") == "custom_value"
            assert config.get("plugin_manager") is mock_plugin_manager_for_lookup
            return mock_override_provider
        return None
    mock_plugin_manager_for_lookup.get_plugin_instance.side_effect = get_plugin_override_effect

    results = await tool_lookup_service_fixture.find_tools(
        query,
        provider_id_override=override_provider_id,
        indexing_formatter_id_override=override_formatter_id,
        provider_config_override=override_provider_config
    )

    assert results == expected_results
    # Verify formatter was called during reindex
    assert len(mock_override_provider.get_indexed_data()) == 1
    assert mock_override_provider.get_indexed_data()[0]["identifier"] == "tool_override"

    # Verify get_plugin_instance calls with correct IDs and config for provider
    # Formatter call
    mock_plugin_manager_for_lookup.get_plugin_instance.assert_any_call(override_formatter_id)
    # Provider calls (reindex and find)
    expected_config_for_provider = {"plugin_manager": mock_plugin_manager_for_lookup, **override_provider_config}

    provider_calls = [
        c for c in mock_plugin_manager_for_lookup.get_plugin_instance.call_args_list
        if c.args[0] == override_provider_id
    ]
    assert len(provider_calls) >= 1
    for p_call in provider_calls:
        assert p_call.kwargs.get("config") == expected_config_for_provider

@pytest.mark.asyncio
async def test_find_tools_provider_find_tools_fails(
    tool_lookup_service_fixture: ToolLookupService,
    mock_plugin_manager_for_lookup: PluginManager,
    caplog
):
    caplog.set_level(logging.ERROR)
    mock_provider = MockLookupProvider()
    mock_provider.find_tools_should_raise = True # Simulate provider find error
    tool_lookup_service_fixture._index_validity_map["mock_lookup_provider_v1"] = True # Assume index is valid

    mock_plugin_manager_for_lookup.get_plugin_instance.return_value = mock_provider

    results = await tool_lookup_service_fixture.find_tools("query provider fail")
    assert results == []
    assert "Error finding tools with provider 'mock_lookup_provider_v1': Provider find_tools failed" in caplog.text
