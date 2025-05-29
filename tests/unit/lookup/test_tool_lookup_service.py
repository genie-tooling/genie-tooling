import logging
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.tool_lookup_providers.abc import ToolLookupProvider as ToolLookupProviderPlugin
from genie_tooling.lookup.service import ToolLookupService
from genie_tooling.lookup.types import RankedToolResult
from genie_tooling.tools.abc import Tool as ToolPlugin
from genie_tooling.definition_formatters.abc import DefinitionFormatter as DefinitionFormatterPlugin
from genie_tooling.tools.manager import ToolManager

class MockToolForLookup(ToolPlugin):
    _identifier_value: str
    _plugin_id_value: str

    def __init__(self, identifier_val: str, name: str = "Mock Tool", desc_llm: str = "Mock LLM Desc"):
        self._identifier_value = identifier_val
        self._plugin_id_value = identifier_val
        self._metadata = {"identifier": self._identifier_value, "name": name, "description_llm": desc_llm, "description_human": "Mock Human Desc", "input_schema": {}, "output_schema": {}, "tags": ["mock"]}; self.raise_in_get_metadata = False
    
    @property
    def identifier(self) -> str: return self._identifier_value
    @property
    def plugin_id(self) -> str: return self._plugin_id_value
    
    async def get_metadata(self) -> Dict[str, Any]:
        if self.raise_in_get_metadata: raise RuntimeError("Metadata retrieval failed for lookup")
        return self._metadata
    async def execute(self, params, key_provider, context=None) -> Any: return "executed"
    async def setup(self, config=None): pass; 
    async def teardown(self): pass

class MockLookupProvider(ToolLookupProviderPlugin):
    _plugin_id_value: str

    def __init__(self):
        self._plugin_id_value = "mock_lookup_provider_v1"
        self.description: str = "Mock Lookup Provider"
        self._indexed_data: List[Dict[str, Any]] = []; self._find_results: List[RankedToolResult] = []; self.index_tools_should_raise = False; self.find_tools_should_raise = False
    
    @property
    def plugin_id(self) -> str: return self._plugin_id_value
    
    async def index_tools(self, tools_data: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None) -> None:
        if self.index_tools_should_raise: raise RuntimeError("Provider index_tools failed")
        self._indexed_data = tools_data
    async def find_tools(self, natural_language_query: str, top_k: int = 5, config: Optional[Dict[str, Any]] = None) -> List[RankedToolResult]:
        if self.find_tools_should_raise: raise RuntimeError("Provider find_tools failed")
        return self._find_results[:top_k]
    def set_find_results(self, results: List[RankedToolResult]): self._find_results = results
    def get_indexed_data(self) -> List[Dict[str, Any]]: return self._indexed_data
    def reset_mock_behavior(self): self.index_tools_should_raise = False; self.find_tools_should_raise = False; self._find_results = []; self._indexed_data = []
    async def setup(self, config=None): pass; 
    async def teardown(self): pass

class MockLookupFormatter(DefinitionFormatterPlugin):
    _plugin_id_value: str

    def __init__(self):
        self._plugin_id_value = "mock_lookup_formatter_v1"
        self.formatter_id: str = "mock_format_for_lookup_v1"
        self.description: str = "Mock Lookup Formatter"
        self.format_should_raise = False; self.format_returns_invalid_type = False
    
    @property
    def plugin_id(self) -> str: return self._plugin_id_value

    def format(self, tool_metadata: Dict[str, Any]) -> Any:
        if self.format_should_raise: raise RuntimeError("Formatter format method failed")
        if self.format_returns_invalid_type: return 123
        return {"identifier": tool_metadata["identifier"], "lookup_text_representation": f"Formatted: {tool_metadata['name']} - {tool_metadata['description_llm']}", "_raw_metadata_snapshot": tool_metadata}
    def reset_mock_behavior(self): self.format_should_raise = False; self.format_returns_invalid_type = False
    async def setup(self, config=None): pass; 
    async def teardown(self): pass

@pytest.fixture
def mock_tool_manager_for_lookup(mocker) -> ToolManager: tm = mocker.MagicMock(spec=ToolManager); tm.list_tools = AsyncMock(return_value=[]); return tm
@pytest.fixture
def mock_plugin_manager_for_lookup(mocker) -> PluginManager: pm = mocker.MagicMock(spec=PluginManager); pm.get_plugin_instance = AsyncMock(); return pm
@pytest.fixture
def tool_lookup_service_fixture(mock_tool_manager_for_lookup: ToolManager, mock_plugin_manager_for_lookup: PluginManager) -> ToolLookupService:
    return ToolLookupService(tool_manager=mock_tool_manager_for_lookup, plugin_manager=mock_plugin_manager_for_lookup, default_provider_id="mock_lookup_provider_v1", default_indexing_formatter_id="mock_lookup_formatter_v1")

@pytest.mark.asyncio
async def test_tool_lookup_service_find_tools_success_first_run_reindexes(tool_lookup_service_fixture: ToolLookupService, mock_tool_manager_for_lookup: ToolManager, mock_plugin_manager_for_lookup: PluginManager):
    query = "find a tool"; mock_tool1 = MockToolForLookup(identifier_val="tool1", name="Tool One", desc_llm="Does one thing"); mock_tool_manager_for_lookup.list_tools.return_value = [mock_tool1]
    mock_formatter = MockLookupFormatter(); mock_provider = MockLookupProvider()
    expected_ranked_results = [RankedToolResult("tool1", 0.9, {"data": "formatted_tool1"})]; mock_provider.set_find_results(expected_ranked_results)
    async def get_plugin_side_effect(plugin_id_req: str, config: Optional[Dict[str, Any]] = None, **kwargs):
        if plugin_id_req == "mock_lookup_formatter_v1": return mock_formatter
        if plugin_id_req == "mock_lookup_provider_v1": return mock_provider
        return None
    mock_plugin_manager_for_lookup.get_plugin_instance.side_effect = get_plugin_side_effect
    results = await tool_lookup_service_fixture.find_tools(query)
    assert results == expected_ranked_results; assert len(mock_provider.get_indexed_data()) == 1; indexed_item = mock_provider.get_indexed_data()[0]
    assert indexed_item["identifier"] == "tool1"; assert "Formatted: Tool One - Does one thing" in indexed_item["lookup_text_representation"]
    expected_config_with_pm = {"plugin_manager": mock_plugin_manager_for_lookup}
    mock_plugin_manager_for_lookup.get_plugin_instance.assert_any_call("mock_lookup_formatter_v1")
    provider_calls = [c for c in mock_plugin_manager_for_lookup.get_plugin_instance.call_args_list if c.args[0] == "mock_lookup_provider_v1"]
    assert len(provider_calls) >= 1
    for p_call in provider_calls: assert p_call.kwargs.get("config") == expected_config_with_pm
