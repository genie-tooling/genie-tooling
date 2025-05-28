### tests/unit/command_processors/impl/test_llm_assisted_processor.py
import json
import logging
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genie_tooling.command_processors.impl.llm_assisted_processor import (
    LLMAssistedToolSelectionProcessorPlugin,
)

# Removed direct import of CommandProcessorResponse as it's not used for construction here
# from genie_tooling.command_processors.types import CommandProcessorResponse
# Removed direct import of ChatMessage and LLMChatResponse, will use dicts
# from genie_tooling.llm_providers.types import ChatMessage, LLMChatResponse
from genie_tooling.lookup.types import RankedToolResult
from genie_tooling.tools.abc import Tool as ToolPlugin

# --- Mock Components ---

class MockToolForLLMAssisted(ToolPlugin):
    def __init__(self, identifier: str, name: str = "Mock Tool", llm_desc: str = "Mock LLM Desc"):
        self.identifier = identifier
        self.plugin_id = identifier
        self._metadata = {
            "identifier": identifier,
            "name": name,
            "description_llm": llm_desc,
            "description_human": "Human Desc",
            "input_schema": {"type": "object", "properties": {"param1": {"type": "string"}}},
            "output_schema": {"type": "object"},
        }

    async def get_metadata(self) -> Dict[str, Any]:
        return self._metadata

    async def execute(self, params: Dict[str, Any], key_provider: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        return "executed"

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        pass

    async def teardown(self) -> None:
        pass


@pytest.fixture
def mock_genie_facade_for_llm_assisted(mocker) -> MagicMock:
    facade = MagicMock(name="MockGenieFacadeForLLMAssisted")

    # Mock ToolManager
    facade._tool_manager = AsyncMock(name="MockToolManager")
    facade._tool_manager.list_tools = AsyncMock(return_value=[])
    facade._tool_manager.get_formatted_tool_definition = AsyncMock(return_value="Formatted Tool Definition")

    # Mock ToolLookupService
    facade._tool_lookup_service = AsyncMock(name="MockToolLookupService")
    facade._tool_lookup_service.find_tools = AsyncMock(return_value=[])

    # Mock LLMInterface (which is facade.llm)
    facade.llm = AsyncMock(name="MockLLMInterface")
    facade.llm.chat = AsyncMock(name="MockLLMChat")

    return facade


@pytest.fixture
def llm_assisted_processor() -> LLMAssistedToolSelectionProcessorPlugin:
    return LLMAssistedToolSelectionProcessorPlugin()

# --- Test Cases ---

@pytest.mark.asyncio
async def test_setup_default_and_custom_config(
    llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin,
    mock_genie_facade_for_llm_assisted: MagicMock
):
    # Test with default config
    await llm_assisted_processor.setup({}, mock_genie_facade_for_llm_assisted)
    assert llm_assisted_processor._genie is mock_genie_facade_for_llm_assisted
    assert llm_assisted_processor._llm_provider_id is None
    assert llm_assisted_processor._tool_formatter_id == "llm_compact_text_v1"
    assert llm_assisted_processor._tool_lookup_top_k is None
    assert llm_assisted_processor._max_llm_retries == 1

    # Test with custom config
    custom_config = {
        "llm_provider_id": "custom_llm",
        "tool_formatter_id": "custom_formatter",
        "tool_lookup_top_k": 3,
        "system_prompt_template": "Custom prompt: {tool_definitions_string}",
        "max_llm_retries": 2,
    }
    await llm_assisted_processor.setup(custom_config, mock_genie_facade_for_llm_assisted)
    assert llm_assisted_processor._llm_provider_id == "custom_llm"
    assert llm_assisted_processor._tool_formatter_id == "custom_formatter"
    assert llm_assisted_processor._tool_lookup_top_k == 3
    assert llm_assisted_processor._system_prompt_template == "Custom prompt: {tool_definitions_string}"
    assert llm_assisted_processor._max_llm_retries == 2


@pytest.mark.asyncio
async def test_get_tool_definitions_string_no_tools(
    llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin,
    mock_genie_facade_for_llm_assisted: MagicMock
):
    await llm_assisted_processor.setup({}, mock_genie_facade_for_llm_assisted)
    mock_genie_facade_for_llm_assisted._tool_manager.list_tools.return_value = []

    defs_str, tool_ids = await llm_assisted_processor._get_tool_definitions_string("any command")
    assert defs_str == "No tools available."
    assert tool_ids == []


@pytest.mark.asyncio
async def test_get_tool_definitions_string_with_lookup(
    llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin,
    mock_genie_facade_for_llm_assisted: MagicMock
):
    config = {"tool_lookup_top_k": 1}
    await llm_assisted_processor.setup(config, mock_genie_facade_for_llm_assisted)

    tool1 = MockToolForLLMAssisted("tool1", "Tool One")
    tool2 = MockToolForLLMAssisted("tool2", "Tool Two") # Won't be in lookup result
    mock_genie_facade_for_llm_assisted._tool_manager.list_tools.return_value = [tool1, tool2]
    mock_genie_facade_for_llm_assisted._tool_lookup_service.find_tools.return_value = [
        RankedToolResult("tool1", 0.9)
    ]
    mock_genie_facade_for_llm_assisted._tool_manager.get_formatted_tool_definition.return_value = "Formatted: Tool One"

    defs_str, tool_ids = await llm_assisted_processor._get_tool_definitions_string("find tool one")
    assert defs_str == "Formatted: Tool One"
    assert tool_ids == ["tool1"]
    mock_genie_facade_for_llm_assisted._tool_lookup_service.find_tools.assert_awaited_once_with("find tool one", top_k=1)
    mock_genie_facade_for_llm_assisted._tool_manager.get_formatted_tool_definition.assert_awaited_once_with("tool1", llm_assisted_processor._tool_formatter_id)

@pytest.mark.asyncio
async def test_get_tool_definitions_string_lookup_fails_fallback_all(
    llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin,
    mock_genie_facade_for_llm_assisted: MagicMock,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.WARNING)
    config = {"tool_lookup_top_k": 2}
    await llm_assisted_processor.setup(config, mock_genie_facade_for_llm_assisted)

    tool1 = MockToolForLLMAssisted("tool1", "Tool One")
    tool2 = MockToolForLLMAssisted("tool2", "Tool Two")
    mock_genie_facade_for_llm_assisted._tool_manager.list_tools.return_value = [tool1, tool2]
    mock_genie_facade_for_llm_assisted._tool_lookup_service.find_tools.side_effect = RuntimeError("Lookup service error")

    # Ensure get_formatted_tool_definition returns distinct values for each tool
    async def format_def_side_effect(tool_id, formatter_id):
        if tool_id == "tool1": return "Formatted: Tool One"
        if tool_id == "tool2": return "Formatted: Tool Two"
        return None
    mock_genie_facade_for_llm_assisted._tool_manager.get_formatted_tool_definition.side_effect = format_def_side_effect

    defs_str, tool_ids = await llm_assisted_processor._get_tool_definitions_string("any command")

    assert "Formatted: Tool One" in defs_str
    assert "Formatted: Tool Two" in defs_str
    assert sorted(tool_ids) == ["tool1", "tool2"]
    assert "Error during tool lookup: Lookup service error. Falling back to all tools." in caplog.text


@pytest.mark.asyncio
async def test_process_command_no_genie_facade(llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin):
    llm_assisted_processor._genie = None # Ensure it's None
    response = await llm_assisted_processor.process_command("do something")
    assert response.get("error") == f"{llm_assisted_processor.plugin_id} not properly set up."


@pytest.mark.asyncio
async def test_process_command_no_tool_definitions(
    llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin,
    mock_genie_facade_for_llm_assisted: MagicMock
):
    await llm_assisted_processor.setup({}, mock_genie_facade_for_llm_assisted)
    mock_genie_facade_for_llm_assisted._tool_manager.list_tools.return_value = [] # No tools
    # Simulate get_tool_definitions_string returning the "No tools available." message
    with patch.object(llm_assisted_processor, "_get_tool_definitions_string", new_callable=AsyncMock, return_value=("No tools available.", [])):
        response = await llm_assisted_processor.process_command("command")
    assert response.get("error") == "No tools available."
    assert "No tools are available in the system." in response.get("llm_thought_process", "")


@pytest.mark.asyncio
async def test_process_command_llm_selects_tool(
    llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin,
    mock_genie_facade_for_llm_assisted: MagicMock
):
    await llm_assisted_processor.setup({}, mock_genie_facade_for_llm_assisted)
    # tool1 = MockToolForLLMAssisted("calc_tool") # Not directly used, _get_tool_definitions_string is mocked
    with patch.object(llm_assisted_processor, "_get_tool_definitions_string", new_callable=AsyncMock, return_value=("Formatted: calc_tool", ["calc_tool"])):
        llm_output_dict = {"thought": "User wants to calculate.", "tool_id": "calc_tool", "params": {"num1": 1, "op": "add"}}
        llm_content_str = json.dumps(llm_output_dict)
        mock_genie_facade_for_llm_assisted.llm.chat.return_value = { # Using plain dict
            "message": {"role": "assistant", "content": llm_content_str},
            "finish_reason": "stop", "usage": None, "raw_response": {}
        }
        response = await llm_assisted_processor.process_command("calculate 1 plus something")

    assert response.get("chosen_tool_id") == "calc_tool"
    assert response.get("extracted_params") == {"num1": 1, "op": "add"}
    assert response.get("llm_thought_process") == "User wants to calculate."
    assert response.get("error") is None


@pytest.mark.asyncio
async def test_process_command_llm_selects_no_tool(
    llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin,
    mock_genie_facade_for_llm_assisted: MagicMock
):
    await llm_assisted_processor.setup({}, mock_genie_facade_for_llm_assisted)
    with patch.object(llm_assisted_processor, "_get_tool_definitions_string", new_callable=AsyncMock, return_value=("Formatted: some_tool", ["some_tool"])):
        llm_output_dict = {"thought": "No tool needed.", "tool_id": None, "params": None}
        llm_content_str = json.dumps(llm_output_dict)
        mock_genie_facade_for_llm_assisted.llm.chat.return_value = { # Using plain dict
            "message": {"role": "assistant", "content": llm_content_str},
            "finish_reason": "stop", "usage": None, "raw_response": {}
        }
        response = await llm_assisted_processor.process_command("just chatting")

    assert response.get("chosen_tool_id") is None
    assert response.get("extracted_params") == {} # Ensure it's an empty dict
    assert response.get("llm_thought_process") == "No tool needed."


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock) # Mock asyncio.sleep
async def test_process_command_llm_json_parse_fail_then_succeed(
    mock_sleep: AsyncMock,
    llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin,
    mock_genie_facade_for_llm_assisted: MagicMock
):
    await llm_assisted_processor.setup({"max_llm_retries": 1}, mock_genie_facade_for_llm_assisted)

    with patch.object(llm_assisted_processor, "_get_tool_definitions_string", new_callable=AsyncMock, return_value=("Formatted: tool", ["tool"])):
        malformed_json_response = { # Using plain dict
            "message": {"role": "assistant", "content": "This is not JSON"},
            "finish_reason": "stop", "usage": None, "raw_response": {}
        }
        correct_llm_output_dict = {"thought": "Success on retry.", "tool_id": "tool", "params": {}}
        correct_json_response = { # Using plain dict
            "message": {"role": "assistant", "content": json.dumps(correct_llm_output_dict)},
            "finish_reason": "stop", "usage": None, "raw_response": {}
        }
        mock_genie_facade_for_llm_assisted.llm.chat.side_effect = [malformed_json_response, correct_json_response]

        response = await llm_assisted_processor.process_command("test retry")

    assert response.get("chosen_tool_id") == "tool"
    assert response.get("llm_thought_process") == "Success on retry."
    assert mock_genie_facade_for_llm_assisted.llm.chat.call_count == 2
    mock_sleep.assert_awaited_once_with(0.5 * 1)


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_process_command_llm_call_fails_max_retries(
    mock_sleep: AsyncMock,
    llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin,
    mock_genie_facade_for_llm_assisted: MagicMock,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR)
    await llm_assisted_processor.setup({"max_llm_retries": 0}, mock_genie_facade_for_llm_assisted)

    with patch.object(llm_assisted_processor, "_get_tool_definitions_string", new_callable=AsyncMock, return_value=("Formatted: tool", ["tool"])):
        mock_genie_facade_for_llm_assisted.llm.chat.side_effect = RuntimeError("LLM API Error")
        response = await llm_assisted_processor.process_command("test failure")

    assert "Failed to process command with LLM after multiple retries: LLM API Error" in response.get("error", "")
    assert mock_genie_facade_for_llm_assisted.llm.chat.call_count == 1
    mock_sleep.assert_not_awaited()
    assert f"{llm_assisted_processor.plugin_id}: Error during LLM call for tool selection (attempt 1): LLM API Error" in caplog.text


@pytest.mark.asyncio
async def test_process_command_llm_chooses_unknown_tool(
    llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin,
    mock_genie_facade_for_llm_assisted: MagicMock,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.WARNING)
    await llm_assisted_processor.setup({}, mock_genie_facade_for_llm_assisted)
    with patch.object(llm_assisted_processor, "_get_tool_definitions_string", new_callable=AsyncMock, return_value=("Formatted: real_tool", ["real_tool"])):
        llm_output_dict = {"thought": "Chose a fake tool.", "tool_id": "fake_tool", "params": {}}
        llm_content_str = json.dumps(llm_output_dict)
        mock_genie_facade_for_llm_assisted.llm.chat.return_value = { # Using plain dict
            "message": {"role": "assistant", "content": llm_content_str},
            "finish_reason": "stop", "usage": None, "raw_response": {}
        }
        response = await llm_assisted_processor.process_command("command")

    assert response.get("chosen_tool_id") is None
    assert "LLM hallucinated a tool_id not in the provided list. Corrected to no tool." in response.get("llm_thought_process", "")
    assert f"{llm_assisted_processor.plugin_id}: LLM chose tool 'fake_tool' which was not in the candidate list (['real_tool']). Treating as no tool chosen." in caplog.text


@pytest.mark.asyncio
async def test_process_command_llm_invalid_params_type(
    llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin,
    mock_genie_facade_for_llm_assisted: MagicMock,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.WARNING)
    await llm_assisted_processor.setup({"max_llm_retries": 0}, mock_genie_facade_for_llm_assisted)
    with patch.object(llm_assisted_processor, "_get_tool_definitions_string", new_callable=AsyncMock, return_value=("Formatted: tool", ["tool"])):
        llm_output_dict = {"thought": "Invalid params.", "tool_id": "tool", "params": "not_a_dict_or_null"}
        llm_content_str = json.dumps(llm_output_dict)
        mock_genie_facade_for_llm_assisted.llm.chat.return_value = { # Using plain dict
            "message": {"role": "assistant", "content": llm_content_str},
            "finish_reason": "stop", "usage": None, "raw_response": {}
        }
        response = await llm_assisted_processor.process_command("command")

    assert response.get("chosen_tool_id") == "tool"
    assert response.get("extracted_params") == {} # Fallback to empty dict
    assert "LLM returned invalid parameter format. Parameters ignored." in response.get("llm_thought_process", "")
    assert f"{llm_assisted_processor.plugin_id}: LLM returned invalid 'params' type for tool 'tool'. Expected dict or null, got <class 'str'>." in caplog.text

@pytest.mark.asyncio
async def test_process_command_llm_content_is_none(
    llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin,
    mock_genie_facade_for_llm_assisted: MagicMock
):
    await llm_assisted_processor.setup({"max_llm_retries": 0}, mock_genie_facade_for_llm_assisted)
    with patch.object(llm_assisted_processor, "_get_tool_definitions_string", new_callable=AsyncMock, return_value=("Formatted: tool", ["tool"])):
        mock_genie_facade_for_llm_assisted.llm.chat.return_value = { # Using plain dict
            "message": {"role": "assistant", "content": None}, # Content is None
            "finish_reason": "stop", "usage": None, "raw_response": {}
        }
        response = await llm_assisted_processor.process_command("test content none")

    assert response.get("error") == "LLM returned empty or invalid content for tool selection."


@pytest.mark.asyncio
async def test_process_command_llm_response_not_dict(
    llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin,
    mock_genie_facade_for_llm_assisted: MagicMock
):
    await llm_assisted_processor.setup({"max_llm_retries": 0}, mock_genie_facade_for_llm_assisted)
    with patch.object(llm_assisted_processor, "_get_tool_definitions_string", new_callable=AsyncMock, return_value=("Formatted: tool", ["tool"])):
        # Simulate LLM response where the 'message' field is not a dictionary
        mock_genie_facade_for_llm_assisted.llm.chat.return_value = { # Using plain dict
            "message": "This should be a dict, not a string",
            "finish_reason": "stop", "usage": None, "raw_response": {}
        }
        response = await llm_assisted_processor.process_command("test message not dict")
    assert response.get("error") == "Invalid LLM response structure."

@pytest.mark.asyncio
async def test_process_command_llm_parsed_output_not_dict(
    llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin,
    mock_genie_facade_for_llm_assisted: MagicMock
):
    await llm_assisted_processor.setup({"max_llm_retries": 0}, mock_genie_facade_for_llm_assisted)
    with patch.object(llm_assisted_processor, "_get_tool_definitions_string", new_callable=AsyncMock, return_value=("Formatted: tool", ["tool"])):
        # Simulate LLM content that parses to a list, not a dict
        llm_content_str = json.dumps(["list", "not", "a", "dict"])
        mock_genie_facade_for_llm_assisted.llm.chat.return_value = { # Using plain dict
            "message": {"role": "assistant", "content": llm_content_str},
            "finish_reason": "stop", "usage": None, "raw_response": {}
        }
        response = await llm_assisted_processor.process_command("test parsed not dict")

    assert "LLM output was not valid JSON: Parsed content is not a dictionary." in response.get("error", "")
