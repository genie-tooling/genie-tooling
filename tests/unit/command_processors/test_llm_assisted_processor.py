import json
import logging
from typing import Any, Dict, Optional, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genie_tooling.command_processors.impl.llm_assisted_processor import (
    LLMAssistedToolSelectionProcessorPlugin,
)
from genie_tooling.lookup.types import RankedToolResult
from genie_tooling.tools.abc import Tool as ToolPlugin

class MockToolForLLMAssisted(ToolPlugin):
    _identifier_value: str; _plugin_id_value: str
    def __init__(self, identifier_val: str, name: str = "Mock Tool", llm_desc: str = "Mock LLM Desc"):
        self._identifier_value = identifier_val; self._plugin_id_value = identifier_val
        self._metadata = {"identifier": self._identifier_value, "name": name, "description_llm": llm_desc, "description_human": "Human Desc", "input_schema": {"type": "object", "properties": {"param1": {"type": "string"}}}, "output_schema": {"type": "object"}}
    @property
    def identifier(self) -> str: return self._identifier_value
    @property
    def plugin_id(self) -> str: return self._plugin_id_value
    async def get_metadata(self) -> Dict[str, Any]: return self._metadata
    async def execute(self, params: Dict[str, Any], kp: Any, ctx: Optional[Dict[str, Any]]=None) -> Any: return "executed"
    async def setup(self, config: Optional[Dict[str, Any]]=None) -> None: pass # Corrected arg name
    async def teardown(self) -> None: pass

@pytest.fixture
def mock_genie_facade_for_llm_assisted(mocker) -> MagicMock:
    facade = MagicMock(name="MockGenieFacadeForLLMAssisted")
    facade._tool_manager = AsyncMock(name="MockToolManager")
    facade._tool_manager.list_tools = AsyncMock(return_value=[])
    facade._tool_manager.get_formatted_tool_definition = AsyncMock(return_value="Formatted Tool Definition")
    facade._tool_lookup_service = AsyncMock(name="MockToolLookupService")
    facade._tool_lookup_service.find_tools = AsyncMock(return_value=[])
    facade.llm = AsyncMock(name="MockLLMInterface")
    facade.llm.chat = AsyncMock(name="MockLLMChat")
    facade._config = MagicMock(name="MockMiddlewareConfigOnFacade")
    facade._config.default_tool_indexing_formatter_id = "compact_text_formatter_plugin_v1"
    return facade

@pytest.fixture
def llm_assisted_processor() -> LLMAssistedToolSelectionProcessorPlugin:
    return LLMAssistedToolSelectionProcessorPlugin()

@pytest.mark.asyncio
async def test_setup_default_and_custom_config(llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin, mock_genie_facade_for_llm_assisted: MagicMock):
    await llm_assisted_processor.setup(config={"genie_facade": mock_genie_facade_for_llm_assisted})
    assert llm_assisted_processor._genie is mock_genie_facade_for_llm_assisted
    assert llm_assisted_processor._llm_provider_id is None
    assert llm_assisted_processor._tool_formatter_id == "compact_text_formatter_plugin_v1" # Corrected assertion
    assert llm_assisted_processor._tool_lookup_top_k is None
    assert llm_assisted_processor._max_llm_retries == 1
    
    custom_config_params = {
        "llm_provider_id": "custom_llm", 
        "tool_formatter_id": "custom_formatter_plugin_id",
        "tool_lookup_top_k": 3, 
        "system_prompt_template": "Custom prompt: {tool_definitions_string}", 
        "max_llm_retries": 2
    }
    await llm_assisted_processor.setup(config={"genie_facade": mock_genie_facade_for_llm_assisted, **custom_config_params})
    assert llm_assisted_processor._llm_provider_id == "custom_llm"
    assert llm_assisted_processor._tool_formatter_id == "custom_formatter_plugin_id"
    assert llm_assisted_processor._tool_lookup_top_k == 3
    assert llm_assisted_processor._system_prompt_template == "Custom prompt: {tool_definitions_string}"
    assert llm_assisted_processor._max_llm_retries == 2

@pytest.mark.asyncio
async def test_get_tool_definitions_string_with_lookup(llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin, mock_genie_facade_for_llm_assisted: MagicMock):
    # The processor will use its own _tool_formatter_id for get_formatted_tool_definition
    # and genie._config.default_tool_indexing_formatter_id for find_tools's override
    processor_formatter_plugin_id = "formatter_plugin_for_tool_def"
    config_params = {"tool_lookup_top_k": 1, "tool_formatter_id": processor_formatter_plugin_id}
    await llm_assisted_processor.setup(config={"genie_facade": mock_genie_facade_for_llm_assisted, **config_params})
    
    tool1 = MockToolForLLMAssisted("tool1", "Tool One")
    tool2 = MockToolForLLMAssisted("tool2", "Tool Two") 
    mock_genie_facade_for_llm_assisted._tool_manager.list_tools.return_value = [tool1, tool2]
    mock_genie_facade_for_llm_assisted._tool_lookup_service.find_tools.return_value = [RankedToolResult("tool1", 0.9)]
    mock_genie_facade_for_llm_assisted._tool_manager.get_formatted_tool_definition = AsyncMock(return_value="Formatted: Tool One")
    
    # Get the expected indexing_formatter_id_override from the mocked facade's config
    expected_indexing_formatter_id = mock_genie_facade_for_llm_assisted._config.default_tool_indexing_formatter_id

    defs_str, tool_ids = await llm_assisted_processor._get_tool_definitions_string("find tool one")
    
    assert defs_str == "Formatted: Tool One"
    assert tool_ids == ["tool1"]
    mock_genie_facade_for_llm_assisted._tool_lookup_service.find_tools.assert_awaited_once_with(
        "find tool one", 
        top_k=1,
        indexing_formatter_id_override=expected_indexing_formatter_id # Corrected assertion
    )
    mock_genie_facade_for_llm_assisted._tool_manager.get_formatted_tool_definition.assert_awaited_once_with(
        "tool1", processor_formatter_plugin_id
    )

@pytest.mark.asyncio
async def test_get_tool_definitions_string_lookup_fails_fallback_all(llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin, mock_genie_facade_for_llm_assisted: MagicMock, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING)
    formatter_plugin_id = "compact_text_formatter_plugin_v1"
    config_params = {"tool_lookup_top_k": 2, "tool_formatter_id": formatter_plugin_id}
    await llm_assisted_processor.setup(config={"genie_facade": mock_genie_facade_for_llm_assisted, **config_params})
    
    tool1 = MockToolForLLMAssisted("tool1", "Tool One")
    tool2 = MockToolForLLMAssisted("tool2", "Tool Two")
    mock_genie_facade_for_llm_assisted._tool_manager.list_tools.return_value = [tool1, tool2]
    mock_genie_facade_for_llm_assisted._tool_lookup_service.find_tools.side_effect = RuntimeError("Lookup service error")
    
    async def format_def_side_effect(tool_id_req: str, formatter_id_req: str):
        assert formatter_id_req == formatter_plugin_id
        if tool_id_req == "tool1": return "Formatted: Tool One"
        if tool_id_req == "tool2": return "Formatted: Tool Two"
        return None
    mock_genie_facade_for_llm_assisted._tool_manager.get_formatted_tool_definition.side_effect = format_def_side_effect
    
    defs_str, tool_ids = await llm_assisted_processor._get_tool_definitions_string("any command")
    
    assert "Formatted: Tool One" in defs_str
    assert "Formatted: Tool Two" in defs_str
    assert sorted(tool_ids) == ["tool1", "tool2"]
    assert "Error during tool lookup: Lookup service error. Falling back to all tools." in caplog.text
