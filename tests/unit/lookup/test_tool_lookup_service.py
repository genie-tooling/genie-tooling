### tests/unit/lookup/test_tool_lookup_service.py
import logging
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.definition_formatters.abc import (
    DefinitionFormatter as DefinitionFormatterPlugin,
)
from genie_tooling.lookup.service import (
    DEFAULT_INDEXING_FORMATTER_ID,
    DEFAULT_LOOKUP_PROVIDER_ID,
    ToolLookupService,
)
from genie_tooling.lookup.types import RankedToolResult
from genie_tooling.tool_lookup_providers.abc import (
    ToolLookupProvider as ToolLookupProviderPlugin,
)
from genie_tooling.tools.abc import Tool as ToolPlugin
from genie_tooling.tools.manager import ToolManager

SERVICE_LOGGER_NAME = "genie_tooling.lookup.service"


# --- Mocks ---
class MockToolForLookup(ToolPlugin):
    _identifier_value: str
    _plugin_id_value: str
    _metadata_value: Dict[str, Any]
    raise_in_get_metadata: bool = False

    def __init__(self, identifier_val: str, name: str = "Mock Tool", desc_llm: str = "Mock LLM Desc", input_schema: Optional[Dict[str, Any]] = None):
        self._identifier_value = identifier_val
        self._plugin_id_value = identifier_val
        self._metadata_value = {
            "identifier": self._identifier_value,
            "name": name,
            "description_llm": desc_llm,
            "description_human": "Human readable description for " + name,
            "input_schema": input_schema or {"type": "object", "properties": {"param1": {"type": "string"}}},
            "output_schema": {"type": "object"},
            "tags": ["mock_tag"]
        }
        self.raise_in_get_metadata = False

    @property
    def identifier(self) -> str: return self._identifier_value
    @property
    def plugin_id(self) -> str: return self._plugin_id_value
    async def get_metadata(self) -> Dict[str, Any]:
        if self.raise_in_get_metadata: raise RuntimeError("Simulated metadata retrieval failed")
        return self._metadata_value
    async def execute(self, params, key_provider, context=None) -> Any: return "executed"
    async def setup(self, config=None): pass
    async def teardown(self): pass


class MockLookupProvider(ToolLookupProviderPlugin):
    _plugin_id_value: str
    description: str = "Mock Lookup Provider"
    _indexed_data: List[Dict[str, Any]]
    _find_results: List[RankedToolResult]
    index_tools_should_raise: bool = False
    find_tools_should_raise: bool = False

    def __init__(self, plugin_id_val: str = "mock_lookup_provider_v1"):
        self._plugin_id_value = plugin_id_val
        self.reset_mock_behavior()

    @property
    def plugin_id(self) -> str: return self._plugin_id_value
    async def index_tools(self, tools_data: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None) -> None:
        if self.index_tools_should_raise: raise RuntimeError("Provider index_tools failed")
        self._indexed_data.extend(tools_data)
    async def find_tools(self, natural_language_query: str, top_k: int = 5, config: Optional[Dict[str, Any]] = None) -> List[RankedToolResult]:
        if self.find_tools_should_raise: raise RuntimeError("Provider find_tools failed")
        return self._find_results[:top_k]
    def set_find_results(self, results: List[RankedToolResult]): self._find_results = results
    def get_indexed_data(self) -> List[Dict[str, Any]]: return self._indexed_data
    def reset_mock_behavior(self):
        self.index_tools_should_raise = False
        self.find_tools_should_raise = False
        self._find_results = []
        self._indexed_data = []
    async def setup(self, config=None): pass
    async def teardown(self): pass


class MockLookupFormatter(DefinitionFormatterPlugin):
    _plugin_id_value: str
    formatter_id: str
    description: str = "Mock Lookup Formatter"
    format_should_raise: bool = False
    format_returns_invalid_type: bool = False
    format_returns_no_lookup_text: bool = False

    def __init__(self, plugin_id_val: str = "mock_lookup_formatter_v1", formatter_id_val: str = "mock_format_for_lookup_v1"):
        self._plugin_id_value = plugin_id_val
        self.formatter_id = formatter_id_val
        self.reset_mock_behavior()

    @property
    def plugin_id(self) -> str: return self._plugin_id_value
    def format(self, tool_metadata: Dict[str, Any]) -> Any:
        if self.format_should_raise: raise RuntimeError("Formatter format method failed")
        if self.format_returns_invalid_type: return 123

        formatted_output = {
            "identifier": tool_metadata["identifier"],
            "_raw_metadata_snapshot": tool_metadata
        }
        if not self.format_returns_no_lookup_text:
            formatted_output["lookup_text_representation"] = f"Formatted: {tool_metadata.get('name', 'N/A')} - {tool_metadata.get('description_llm', 'N/A')}"

        return formatted_output

    def reset_mock_behavior(self):
        self.format_should_raise = False
        self.format_returns_invalid_type = False
        self.format_returns_no_lookup_text = False
    async def setup(self, config=None): pass
    async def teardown(self): pass


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
    return ToolLookupService(
        tool_manager=mock_tool_manager_for_lookup,
        plugin_manager=mock_plugin_manager_for_lookup,
        default_provider_id=DEFAULT_LOOKUP_PROVIDER_ID,
        default_indexing_formatter_id=DEFAULT_INDEXING_FORMATTER_ID
    )

# --- Tests ---

@pytest.mark.asyncio
async def test_tool_lookup_service_find_tools_success_first_run_reindexes(
    tool_lookup_service_fixture: ToolLookupService,
    mock_tool_manager_for_lookup: ToolManager,
    mock_plugin_manager_for_lookup: PluginManager
):
    query = "find a tool"
    mock_tool1 = MockToolForLookup(identifier_val="tool1", name="Tool One", desc_llm="Does one thing")
    mock_tool_manager_for_lookup.list_tools.return_value = [mock_tool1]

    mock_formatter = MockLookupFormatter(plugin_id_val=DEFAULT_INDEXING_FORMATTER_ID)
    mock_provider = MockLookupProvider(plugin_id_val=DEFAULT_LOOKUP_PROVIDER_ID)

    expected_ranked_results = [RankedToolResult("tool1", 0.9, {"data": "formatted_tool1"})]
    mock_provider.set_find_results(expected_ranked_results)

    async def get_plugin_side_effect(plugin_id_req: str, config: Optional[Dict[str, Any]] = None, **kwargs):
        if plugin_id_req == DEFAULT_INDEXING_FORMATTER_ID:
            return mock_formatter
        if plugin_id_req == DEFAULT_LOOKUP_PROVIDER_ID:
            assert config is not None
            assert config.get("plugin_manager") is mock_plugin_manager_for_lookup
            return mock_provider
        return None
    mock_plugin_manager_for_lookup.get_plugin_instance.side_effect = get_plugin_side_effect

    results = await tool_lookup_service_fixture.find_tools(query)

    assert results == expected_ranked_results
    assert len(mock_provider.get_indexed_data()) == 1
    indexed_item = mock_provider.get_indexed_data()[0]
    assert indexed_item["identifier"] == "tool1"
    assert "Formatted: Tool One - Does one thing" in indexed_item["lookup_text_representation"]

    mock_plugin_manager_for_lookup.get_plugin_instance.assert_any_call(DEFAULT_INDEXING_FORMATTER_ID)
    mock_plugin_manager_for_lookup.get_plugin_instance.assert_any_call(
        DEFAULT_LOOKUP_PROVIDER_ID,
        config={"plugin_manager": mock_plugin_manager_for_lookup}
    )

@pytest.mark.asyncio
async def test_reindex_formatter_not_found(
    tool_lookup_service_fixture: ToolLookupService,
    mock_plugin_manager_for_lookup: PluginManager,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR, logger=SERVICE_LOGGER_NAME)
    provider_id_for_test = "any_provider_id_formatter_test"
    formatter_id_for_test = "bad_formatter_id"

    # Mock provider to be found, but formatter to fail
    mock_valid_provider = MockLookupProvider(plugin_id_val=provider_id_for_test)
    async def get_instance_side_effect(plugin_id_req: str, config=None, **kwargs):
        if plugin_id_req == provider_id_for_test:
            return mock_valid_provider
        if plugin_id_req == formatter_id_for_test:
            return None # Formatter load fails
        return MagicMock() # Default for other calls
    mock_plugin_manager_for_lookup.get_plugin_instance.side_effect = get_instance_side_effect

    success = await tool_lookup_service_fixture.reindex_tools_for_provider(
        provider_id=provider_id_for_test,
        formatter_plugin_id_override=formatter_id_for_test
    )
    assert success is False
    assert f"Indexing formatter plugin '{formatter_id_for_test}' not found or invalid." in caplog.text
    # Ensure the provider itself was attempted to be loaded
    mock_plugin_manager_for_lookup.get_plugin_instance.assert_any_call(provider_id_for_test, config={"plugin_manager": mock_plugin_manager_for_lookup})


@pytest.mark.asyncio
async def test_reindex_no_tools_available(
    tool_lookup_service_fixture: ToolLookupService,
    mock_tool_manager_for_lookup: ToolManager,
    mock_plugin_manager_for_lookup: PluginManager,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.DEBUG, logger=SERVICE_LOGGER_NAME)
    mock_tool_manager_for_lookup.list_tools.return_value = []
    mock_formatter = MockLookupFormatter()
    mock_provider = MockLookupProvider()

    async def get_plugin_side_effect(plugin_id_req: str, config=None, **kwargs):
        if plugin_id_req == tool_lookup_service_fixture._default_indexing_formatter_id:
            return mock_formatter
        if plugin_id_req == "test_provider_for_reindex":
            return mock_provider
        return None
    mock_plugin_manager_for_lookup.get_plugin_instance.side_effect = get_plugin_side_effect

    success = await tool_lookup_service_fixture.reindex_tools_for_provider(provider_id="test_provider_for_reindex")
    assert success is True
    assert "No tools available to format for indexing." in caplog.text
    assert len(mock_provider.get_indexed_data()) == 0

@pytest.mark.asyncio
async def test_reindex_tool_get_metadata_fails(
    tool_lookup_service_fixture: ToolLookupService,
    mock_tool_manager_for_lookup: ToolManager,
    mock_plugin_manager_for_lookup: PluginManager,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR, logger=SERVICE_LOGGER_NAME)
    tool_fails_meta = MockToolForLookup("fail_meta_tool")
    tool_fails_meta.raise_in_get_metadata = True
    mock_tool_manager_for_lookup.list_tools.return_value = [tool_fails_meta]

    mock_formatter = MockLookupFormatter()
    mock_provider = MockLookupProvider()
    async def get_plugin_side_effect(plugin_id_req: str, config=None, **kwargs):
        if plugin_id_req == tool_lookup_service_fixture._default_indexing_formatter_id: return mock_formatter
        if plugin_id_req == "test_provider_meta_fail": return mock_provider
        return None
    mock_plugin_manager_for_lookup.get_plugin_instance.side_effect = get_plugin_side_effect

    await tool_lookup_service_fixture.reindex_tools_for_provider(provider_id="test_provider_meta_fail")
    assert "Error formatting tool 'fail_meta_tool' for indexing" in caplog.text
    assert "Simulated metadata retrieval failed" in caplog.text
    assert len(mock_provider.get_indexed_data()) == 0

@pytest.mark.asyncio
async def test_reindex_formatter_format_fails(
    tool_lookup_service_fixture: ToolLookupService,
    mock_tool_manager_for_lookup: ToolManager,
    mock_plugin_manager_for_lookup: PluginManager,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR, logger=SERVICE_LOGGER_NAME)
    mock_tool = MockToolForLookup("tool_format_fail")
    mock_tool_manager_for_lookup.list_tools.return_value = [mock_tool]

    mock_formatter_fails = MockLookupFormatter()
    mock_formatter_fails.format_should_raise = True
    mock_provider = MockLookupProvider()

    async def get_plugin_side_effect(plugin_id_req: str, config=None, **kwargs):
        if plugin_id_req == tool_lookup_service_fixture._default_indexing_formatter_id: return mock_formatter_fails
        if plugin_id_req == "test_provider_format_fail": return mock_provider
        return None
    mock_plugin_manager_for_lookup.get_plugin_instance.side_effect = get_plugin_side_effect

    await tool_lookup_service_fixture.reindex_tools_for_provider(provider_id="test_provider_format_fail")
    assert "Error formatting tool 'tool_format_fail' for indexing" in caplog.text
    assert "Formatter format method failed" in caplog.text
    assert len(mock_provider.get_indexed_data()) == 0

@pytest.mark.asyncio
async def test_reindex_formatter_returns_invalid_type_or_missing_fields(
    tool_lookup_service_fixture: ToolLookupService,
    mock_tool_manager_for_lookup: ToolManager,
    mock_plugin_manager_for_lookup: PluginManager,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.WARNING, logger=SERVICE_LOGGER_NAME)
    mock_tool = MockToolForLookup("tool_bad_format_output", name="Tool Bad Output", desc_llm="Desc for Bad Output")
    mock_tool_manager_for_lookup.list_tools.return_value = [mock_tool]

    mock_formatter_bad_type = MockLookupFormatter()
    mock_formatter_bad_type.format_returns_invalid_type = True
    mock_provider = MockLookupProvider()
    mock_provider.reset_mock_behavior() # Ensure clean state for provider

    async def get_plugin_side_effect_bad_type(plugin_id_req: str, config=None, **kwargs):
        if plugin_id_req == tool_lookup_service_fixture._default_indexing_formatter_id: return mock_formatter_bad_type
        if plugin_id_req == "test_provider_bad_type": return mock_provider
        return None
    mock_plugin_manager_for_lookup.get_plugin_instance.side_effect = get_plugin_side_effect_bad_type

    await tool_lookup_service_fixture.reindex_tools_for_provider(provider_id="test_provider_bad_type")
    assert "produced unexpected type: <class 'int'>. Skipping." in caplog.text
    assert len(mock_provider.get_indexed_data()) == 0
    caplog.clear()
    mock_provider.reset_mock_behavior()

    mock_formatter_no_text = MockLookupFormatter()
    mock_formatter_no_text.format_returns_no_lookup_text = True

    async def get_plugin_side_effect_no_text(plugin_id_req: str, config=None, **kwargs):
        if plugin_id_req == tool_lookup_service_fixture._default_indexing_formatter_id: return mock_formatter_no_text
        if plugin_id_req == "test_provider_no_text": return mock_provider
        return None
    mock_plugin_manager_for_lookup.get_plugin_instance.side_effect = get_plugin_side_effect_no_text

    await tool_lookup_service_fixture.reindex_tools_for_provider(provider_id="test_provider_no_text")
    assert len(mock_provider.get_indexed_data()) == 1
    indexed_data = mock_provider.get_indexed_data()[0]
    assert "lookup_text_representation" in indexed_data
    # Corrected expected string
    assert "Tool: Tool Bad Output. Description: Desc for Bad Output" in indexed_data["lookup_text_representation"]


@pytest.mark.asyncio
async def test_reindex_provider_index_tools_fails(
    tool_lookup_service_fixture: ToolLookupService,
    mock_tool_manager_for_lookup: ToolManager,
    mock_plugin_manager_for_lookup: PluginManager,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR, logger=SERVICE_LOGGER_NAME)
    mock_tool = MockToolForLookup("tool_index_fail")
    mock_tool_manager_for_lookup.list_tools.return_value = [mock_tool]

    mock_formatter = MockLookupFormatter()
    mock_provider_fails_index = MockLookupProvider()
    mock_provider_fails_index.index_tools_should_raise = True

    async def get_plugin_side_effect(plugin_id_req: str, config=None, **kwargs):
        if plugin_id_req == tool_lookup_service_fixture._default_indexing_formatter_id: return mock_formatter
        if plugin_id_req == "test_provider_index_fail": return mock_provider_fails_index
        return None
    mock_plugin_manager_for_lookup.get_plugin_instance.side_effect = get_plugin_side_effect

    success = await tool_lookup_service_fixture.reindex_tools_for_provider(provider_id="test_provider_index_fail")
    assert success is False
    assert "Error during reindexing with provider 'test_provider_index_fail': Provider index_tools failed" in caplog.text
    assert tool_lookup_service_fixture._index_validity_map.get("test_provider_index_fail") is False


@pytest.mark.asyncio
async def test_find_tools_empty_query(tool_lookup_service_fixture: ToolLookupService, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG, logger=SERVICE_LOGGER_NAME)
    results_empty = await tool_lookup_service_fixture.find_tools("")
    assert results_empty == []
    assert "Empty query provided. Returning no results." in caplog.text
    caplog.clear()
    results_space = await tool_lookup_service_fixture.find_tools("   ")
    assert results_space == []
    assert "Empty query provided. Returning no results." in caplog.text


@pytest.mark.asyncio
async def test_find_tools_provider_not_found(
    tool_lookup_service_fixture: ToolLookupService,
    mock_plugin_manager_for_lookup: PluginManager,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR, logger=SERVICE_LOGGER_NAME)
    async def get_instance_side_effect(plugin_id_req: str, config=None, **kwargs):
        if plugin_id_req == "non_existent_provider":
            return None # Provider load fails
        return MagicMock() # Default for other calls (like formatter during reindex attempt)
    mock_plugin_manager_for_lookup.get_plugin_instance.side_effect = get_instance_side_effect

    results = await tool_lookup_service_fixture.find_tools("query", provider_id_override="non_existent_provider")
    assert results == []
    assert "Tool lookup provider 'non_existent_provider' not found or invalid." in caplog.text


@pytest.mark.asyncio
async def test_find_tools_reindex_fails_then_find_fails(
    tool_lookup_service_fixture: ToolLookupService,
    mock_plugin_manager_for_lookup: PluginManager,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR, logger=SERVICE_LOGGER_NAME)
    provider_id_for_test = "provider_needs_reindex_test"
    formatter_id_for_test = "bad_formatter_for_reindex_test"

    # Mock provider to be found, but formatter to fail during reindex
    mock_valid_provider = MockLookupProvider(plugin_id_val=provider_id_for_test)
    async def get_instance_side_effect(plugin_id_req: str, config=None, **kwargs):
        if plugin_id_req == provider_id_for_test:
            return mock_valid_provider
        if plugin_id_req == formatter_id_for_test:
            return None # Formatter load fails
        return MagicMock()
    mock_plugin_manager_for_lookup.get_plugin_instance.side_effect = get_instance_side_effect

    results = await tool_lookup_service_fixture.find_tools(
        "query",
        provider_id_override=provider_id_for_test,
        indexing_formatter_id_override=formatter_id_for_test
    )
    assert results == []
    # Check for the sequence of logs
    assert f"Indexing formatter plugin '{formatter_id_for_test}' not found or invalid." in caplog.text
    assert f"Failed to prepare tool data for provider '{provider_id_for_test}' due to formatter error. Indexing aborted." in caplog.text
    assert f"Re-indexing failed for '{provider_id_for_test}'. Cannot perform lookup." in caplog.text


@pytest.mark.asyncio
async def test_find_tools_provider_find_method_fails(
    tool_lookup_service_fixture: ToolLookupService,
    mock_tool_manager_for_lookup: ToolManager,
    mock_plugin_manager_for_lookup: PluginManager,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR, logger=SERVICE_LOGGER_NAME)
    mock_tool = MockToolForLookup("tool_find_fail")
    mock_tool_manager_for_lookup.list_tools.return_value = [mock_tool]

    mock_formatter = MockLookupFormatter()
    mock_provider_fails_find = MockLookupProvider()
    mock_provider_fails_find.find_tools_should_raise = True

    async def get_plugin_side_effect(plugin_id_req: str, config=None, **kwargs):
        if plugin_id_req == tool_lookup_service_fixture._default_indexing_formatter_id: return mock_formatter
        if plugin_id_req == "test_provider_find_fail": return mock_provider_fails_find
        return None
    mock_plugin_manager_for_lookup.get_plugin_instance.side_effect = get_plugin_side_effect

    results = await tool_lookup_service_fixture.find_tools("query", provider_id_override="test_provider_find_fail")
    assert results == []
    assert "Error finding tools with provider 'test_provider_find_fail': Provider find_tools failed" in caplog.text


@pytest.mark.asyncio
async def test_invalidate_index(tool_lookup_service_fixture: ToolLookupService, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger=SERVICE_LOGGER_NAME)
    provider_id1 = "prov1"
    provider_id2 = "prov2"
    tool_lookup_service_fixture._index_validity_map[provider_id1] = True
    tool_lookup_service_fixture._index_validity_map[provider_id2] = True

    tool_lookup_service_fixture.invalidate_index(provider_id1)
    assert tool_lookup_service_fixture._index_validity_map.get(provider_id1) is False
    assert tool_lookup_service_fixture._index_validity_map.get(provider_id2) is True
    assert f"Index for provider '{provider_id1}' invalidated." in caplog.text
    caplog.clear()

    tool_lookup_service_fixture.invalidate_index()
    assert tool_lookup_service_fixture._index_validity_map.get(provider_id1) is False
    assert tool_lookup_service_fixture._index_validity_map.get(provider_id2) is False
    assert "All provider indices invalidated." in caplog.text
    caplog.clear()

    tool_lookup_service_fixture.invalidate_index("non_existent_provider")
    assert "Index for provider 'non_existent_provider' invalidated." not in caplog.text
