### tests/unit/command_processors/impl/test_llm_assisted_processor.py
import logging
from typing import Any, Dict, Optional  # Added List for type hint
from unittest.mock import AsyncMock, MagicMock, patch  # Ensure patch is imported

import pytest
from genie_tooling.command_processors.impl.llm_assisted_processor import (
    DEFAULT_SYSTEM_PROMPT_TEMPLATE,
    LLMAssistedToolSelectionProcessorPlugin,
)
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.llm_providers.types import ChatMessage, LLMChatResponse
from genie_tooling.lookup.types import RankedToolResult
from genie_tooling.tools.abc import Tool as ToolPlugin

# Logger for the module under test
PROCESSOR_LOGGER_NAME = "genie_tooling.command_processors.impl.llm_assisted_processor"
processor_module_logger = logging.getLogger(PROCESSOR_LOGGER_NAME)


class MockToolForLLMAssisted(ToolPlugin):
    _identifier_value: str
    _plugin_id_value: str
    _metadata_value: Dict[str, Any]

    def __init__(self, identifier_val: str, name: str = "Mock Tool", llm_desc: str = "Mock LLM Desc", input_schema: Optional[Dict[str, Any]] = None):
        self._identifier_value = identifier_val
        self._plugin_id_value = identifier_val # Assuming plugin_id matches identifier for simplicity
        self._metadata_value = {
            "identifier": self._identifier_value,
            "name": name,
            "description_llm": llm_desc,
            "description_human": "Human readable description for " + name,
            "input_schema": input_schema or {"type": "object", "properties": {"param1": {"type": "string"}}},
            "output_schema": {"type": "object"}
        }

    @property
    def identifier(self) -> str:
        return self._identifier_value

    @property
    def plugin_id(self) -> str:
        return self._plugin_id_value

    async def get_metadata(self) -> Dict[str, Any]:
        return self._metadata_value

    async def execute(self, params: Dict[str, Any], key_provider: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        return {"status": "executed", "params_received": params}

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        pass

    async def teardown(self) -> None:
        pass


@pytest.fixture
def mock_genie_facade_for_llm_assisted(mocker) -> MagicMock:
    facade = MagicMock(name="MockGenieFacadeForLLMAssisted")

    facade._tool_manager = AsyncMock(name="MockToolManagerOnFacade")
    facade._tool_manager.list_tools = AsyncMock(return_value=[])
    facade._tool_manager.get_formatted_tool_definition = AsyncMock(return_value="Formatted Tool Definition String")

    facade._tool_lookup_service = AsyncMock(name="MockToolLookupServiceOnFacade")
    facade._tool_lookup_service.find_tools = AsyncMock(return_value=[])

    facade.llm = AsyncMock(name="MockLLMInterfaceOnFacade")
    facade.llm.chat = AsyncMock(name="MockLLMChatMethodOnFacade")

    # Mock MiddlewareConfig on the facade
    mock_config = MagicMock(spec=MiddlewareConfig)
    mock_config.default_tool_indexing_formatter_id = "default_indexing_formatter_from_config"
    facade._config = mock_config

    return facade


@pytest.fixture
def llm_assisted_processor() -> LLMAssistedToolSelectionProcessorPlugin:
    processor = LLMAssistedToolSelectionProcessorPlugin()
    # The _extract_json_block method is now part of the plugin itself.
    # No need to mock it here unless specifically testing different extraction logic
    # than what's in the plugin. For these tests, we want to test the plugin's method.
    return processor


# --- Test Setup ---
@pytest.mark.asyncio
async def test_setup_default_and_custom_config(
    llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin,
    mock_genie_facade_for_llm_assisted: MagicMock
):
    # Test with default config values
    await llm_assisted_processor.setup(config={"genie_facade": mock_genie_facade_for_llm_assisted})
    assert llm_assisted_processor._genie is mock_genie_facade_for_llm_assisted
    assert llm_assisted_processor._llm_provider_id is None
    assert llm_assisted_processor._tool_formatter_id == "compact_text_formatter_plugin_v1"
    assert llm_assisted_processor._tool_lookup_top_k is None
    assert llm_assisted_processor._system_prompt_template == DEFAULT_SYSTEM_PROMPT_TEMPLATE
    assert llm_assisted_processor._max_llm_retries == 1

    # Test with custom config values
    custom_config_params = {
        "llm_provider_id": "custom_llm_provider",
        "tool_formatter_id": "custom_tool_formatter",
        "tool_lookup_top_k": 5,
        "system_prompt_template": "Your custom prompt: {tool_definitions_string}",
        "max_llm_retries": 3
    }
    await llm_assisted_processor.setup(config={"genie_facade": mock_genie_facade_for_llm_assisted, **custom_config_params})
    assert llm_assisted_processor._llm_provider_id == "custom_llm_provider"
    assert llm_assisted_processor._tool_formatter_id == "custom_tool_formatter"
    assert llm_assisted_processor._tool_lookup_top_k == 5
    assert llm_assisted_processor._system_prompt_template == "Your custom prompt: {tool_definitions_string}"
    assert llm_assisted_processor._max_llm_retries == 3


@pytest.mark.asyncio
async def test_setup_no_genie_facade(
    llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR, logger=PROCESSOR_LOGGER_NAME)
    await llm_assisted_processor.setup(config={}) # No genie_facade
    assert llm_assisted_processor._genie is None
    assert any(
        f"{llm_assisted_processor.plugin_id}: Genie facade not found in config or is invalid." in rec.message
        for rec in caplog.records
    )


# --- Test _get_tool_definitions_string ---
@pytest.mark.asyncio
async def test_get_tool_definitions_string_with_lookup_success(
    llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin,
    mock_genie_facade_for_llm_assisted: MagicMock
):
    processor_formatter_id = "processor_formatter_for_defs"
    await llm_assisted_processor.setup(config={
        "genie_facade": mock_genie_facade_for_llm_assisted,
        "tool_lookup_top_k": 1,
        "tool_formatter_id": processor_formatter_id
    })

    tool1 = MockToolForLLMAssisted("tool_one_id", "Tool One")
    mock_genie_facade_for_llm_assisted._tool_manager.list_tools.return_value = [tool1]
    mock_genie_facade_for_llm_assisted._tool_lookup_service.find_tools.return_value = [
        RankedToolResult("tool_one_id", 0.95)
    ]
    mock_genie_facade_for_llm_assisted._tool_manager.get_formatted_tool_definition.return_value = "Formatted: Tool One Def"

    defs_str, tool_ids = await llm_assisted_processor._get_tool_definitions_string("find tool one")

    assert defs_str == "Formatted: Tool One Def"
    assert tool_ids == ["tool_one_id"]
    mock_genie_facade_for_llm_assisted._tool_lookup_service.find_tools.assert_awaited_once_with(
        "find tool one",
        top_k=1,
        indexing_formatter_id_override=mock_genie_facade_for_llm_assisted._config.default_tool_indexing_formatter_id
    )
    mock_genie_facade_for_llm_assisted._tool_manager.get_formatted_tool_definition.assert_awaited_once_with(
        "tool_one_id", processor_formatter_id
    )


@pytest.mark.asyncio
async def test_get_tool_definitions_string_lookup_fails_fallback_all(
    llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin,
    mock_genie_facade_for_llm_assisted: MagicMock,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.WARNING, logger=PROCESSOR_LOGGER_NAME)
    processor_formatter_id = "formatter_for_fallback"
    await llm_assisted_processor.setup(config={
        "genie_facade": mock_genie_facade_for_llm_assisted,
        "tool_lookup_top_k": 2,
        "tool_formatter_id": processor_formatter_id
    })

    tool1 = MockToolForLLMAssisted("t1", "Tool Alpha")
    tool2 = MockToolForLLMAssisted("t2", "Tool Beta")
    mock_genie_facade_for_llm_assisted._tool_manager.list_tools.return_value = [tool1, tool2]
    mock_genie_facade_for_llm_assisted._tool_lookup_service.find_tools.side_effect = RuntimeError("Lookup service is down")

    async def mock_format_def(tool_id_req: str, formatter_id_req: str):
        assert formatter_id_req == processor_formatter_id
        if tool_id_req == "t1": return "Formatted: Tool Alpha"
        if tool_id_req == "t2": return "Formatted: Tool Beta"
        return None
    mock_genie_facade_for_llm_assisted._tool_manager.get_formatted_tool_definition.side_effect = mock_format_def

    defs_str, tool_ids = await llm_assisted_processor._get_tool_definitions_string("any query")

    assert "Formatted: Tool Alpha" in defs_str
    assert "Formatted: Tool Beta" in defs_str
    assert sorted(tool_ids) == ["t1", "t2"]
    assert any(
        "Error during tool lookup: Lookup service is down. Falling back to all tools." in rec.message
        for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_get_tool_definitions_string_no_tools_available(
    llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin,
    mock_genie_facade_for_llm_assisted: MagicMock
):
    await llm_assisted_processor.setup(config={"genie_facade": mock_genie_facade_for_llm_assisted})
    mock_genie_facade_for_llm_assisted._tool_manager.list_tools.return_value = [] # No tools

    defs_str, tool_ids = await llm_assisted_processor._get_tool_definitions_string("any query")
    assert defs_str == "No tools available."
    assert tool_ids == []


@pytest.mark.asyncio
async def test_get_tool_definitions_string_formatter_fails_for_a_tool(
    llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin,
    mock_genie_facade_for_llm_assisted: MagicMock,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.WARNING, logger=PROCESSOR_LOGGER_NAME)
    formatter_id = "test_formatter"
    await llm_assisted_processor.setup(config={
        "genie_facade": mock_genie_facade_for_llm_assisted,
        "tool_formatter_id": formatter_id
    })

    tool1 = MockToolForLLMAssisted("good_tool", "Good Tool")
    tool2 = MockToolForLLMAssisted("bad_format_tool", "Bad Format Tool")
    mock_genie_facade_for_llm_assisted._tool_manager.list_tools.return_value = [tool1, tool2]

    async def mock_format_def_selective_fail(tool_id_req: str, formatter_id_req: str):
        assert formatter_id_req == formatter_id
        if tool_id_req == "good_tool": return "Formatted: Good Tool"
        if tool_id_req == "bad_format_tool": return None # Simulate formatter returning None
        return "Should not be called"
    mock_genie_facade_for_llm_assisted._tool_manager.get_formatted_tool_definition.side_effect = mock_format_def_selective_fail

    defs_str, tool_ids = await llm_assisted_processor._get_tool_definitions_string("query")

    assert defs_str == "Formatted: Good Tool" # Only good tool's definition
    assert tool_ids == ["good_tool", "bad_format_tool"] # Candidate IDs still include all initially selected
    assert any(
        f"Failed to get formatted definition for tool 'bad_format_tool' using formatter plugin ID '{formatter_id}'." in rec.message
        for rec in caplog.records
    )

@pytest.mark.asyncio
async def test_get_tool_definitions_string_tool_lookup_top_k_none_or_zero(
    llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin,
    mock_genie_facade_for_llm_assisted: MagicMock
):
    tool1 = MockToolForLLMAssisted("t1", "Tool Alpha")
    tool2 = MockToolForLLMAssisted("t2", "Tool Beta")
    mock_genie_facade_for_llm_assisted._tool_manager.list_tools.return_value = [tool1, tool2]
    mock_genie_facade_for_llm_assisted._tool_manager.get_formatted_tool_definition.side_effect = \
        lambda tid, fid: f"Formatted: {tid}"

    # Test with tool_lookup_top_k = None
    await llm_assisted_processor.setup(config={
        "genie_facade": mock_genie_facade_for_llm_assisted,
        "tool_lookup_top_k": None
    })
    defs_str_none, tool_ids_none = await llm_assisted_processor._get_tool_definitions_string("query")
    assert "Formatted: t1" in defs_str_none
    assert "Formatted: t2" in defs_str_none
    assert sorted(tool_ids_none) == ["t1", "t2"]
    mock_genie_facade_for_llm_assisted._tool_lookup_service.find_tools.assert_not_called()

    mock_genie_facade_for_llm_assisted._tool_lookup_service.find_tools.reset_mock()
    # Test with tool_lookup_top_k = 0
    await llm_assisted_processor.setup(config={
        "genie_facade": mock_genie_facade_for_llm_assisted,
        "tool_lookup_top_k": 0
    })
    defs_str_zero, tool_ids_zero = await llm_assisted_processor._get_tool_definitions_string("query")
    assert "Formatted: t1" in defs_str_zero
    assert "Formatted: t2" in defs_str_zero
    assert sorted(tool_ids_zero) == ["t1", "t2"]
    mock_genie_facade_for_llm_assisted._tool_lookup_service.find_tools.assert_not_called()


# --- Test _extract_json_block ---
@pytest.mark.parametrize("text_input, expected_json_str", [
    ('Some text before {"key": "value", "num": 1} and after.', '{"key": "value", "num": 1}'),
    ('{"only_json": true}', '{"only_json": true}'),
    ('```json\n{"code_block_json": "data"}\n```', '{"code_block_json": "data"}'),
    ('Here is the JSON: {"preamble_json": [1,2]}', '{"preamble_json": [1,2]}'),
    ("No JSON here.", None),
    ('Malformed {json: "block",', None),
    ('Text with { "inner": { "nested": "value" } } block.', '{ "inner": { "nested": "value" } }'),
    ('{"a":1} some text {"b":2}', '{"a":1}'),
    ('Thought: ... \n```json\n{"thought": "User wants to calculate.", "tool_id": "tool_calc", "params": {"num1": 5, "num2": 3}}\n```', '{"thought": "User wants to calculate.", "tool_id": "tool_calc", "params": {"num1": 5, "num2": 3}}'),
    ('```\n{"generic_code_block": true}\n```', '{"generic_code_block": true}'),
    ('Text with multiple JSON blocks: {"first": 1} and then {"second": 2}.', '{"first": 1}'), # Should get first
    ('Text with nested JSON in text: "outer text {\\"inner_json\\": true}"', None), # Not a top-level JSON block
    ('Text with array: [1, 2, {"key": "val"}] trailing.', '[1, 2, {"key": "val"}]'),
    ('```json\n[\n  {"item": 1},\n  {"item": 2}\n]\n```', '[\n  {"item": 1},\n  {"item": 2}\n]'),
])
def test_extract_json_block(
    llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin,
    text_input: str,
    expected_json_str: Optional[str]
):
    # We are testing the plugin's actual method now
    assert llm_assisted_processor._extract_json_block(text_input) == expected_json_str


# --- Test process_command ---
@pytest.mark.asyncio
async def test_process_command_success(
    llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin,
    mock_genie_facade_for_llm_assisted: MagicMock
):
    await llm_assisted_processor.setup(config={"genie_facade": mock_genie_facade_for_llm_assisted})
    tool1 = MockToolForLLMAssisted("tool_calc", "Calculator")
    mock_genie_facade_for_llm_assisted._tool_manager.list_tools.return_value = [tool1]
    mock_genie_facade_for_llm_assisted._tool_manager.get_formatted_tool_definition.return_value = "Tool: Calculator..."

    llm_response_content = 'Thought: User wants to calculate. Tool: tool_calc. Params: {"num1": 5, "num2": 3}. \n```json\n{"thought": "User wants to calculate.", "tool_id": "tool_calc", "params": {"num1": 5, "num2": 3}}\n```'
    mock_genie_facade_for_llm_assisted.llm.chat.return_value = LLMChatResponse(
        message=ChatMessage(role="assistant", content=llm_response_content),
        raw_response={"id": "mock_resp_id"}
    )

    response = await llm_assisted_processor.process_command("calculate 5 + 3")

    assert response.get("error") is None
    assert response.get("chosen_tool_id") == "tool_calc"
    assert response.get("extracted_params") == {"num1": 5, "num2": 3}
    assert "User wants to calculate." in response.get("llm_thought_process", "")


@pytest.mark.asyncio
async def test_process_command_no_tool_chosen_by_llm(
    llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin,
    mock_genie_facade_for_llm_assisted: MagicMock
):
    await llm_assisted_processor.setup(config={"genie_facade": mock_genie_facade_for_llm_assisted})
    mock_genie_facade_for_llm_assisted._tool_manager.list_tools.return_value = [MockToolForLLMAssisted("any_tool")]
    mock_genie_facade_for_llm_assisted._tool_manager.get_formatted_tool_definition.return_value = "Tool: Any Tool..."

    llm_response_content = '{"thought": "No suitable tool.", "tool_id": null, "params": null}'
    mock_genie_facade_for_llm_assisted.llm.chat.return_value = LLMChatResponse(
        message=ChatMessage(role="assistant", content=llm_response_content),
        raw_response={}
    )

    response = await llm_assisted_processor.process_command("just chatting")
    assert response.get("chosen_tool_id") is None
    assert response.get("extracted_params") == {} # Should be empty dict if no tool
    assert "No suitable tool." in response.get("llm_thought_process", "")


@pytest.mark.asyncio
async def test_process_command_llm_hallucinates_tool_id(
    llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin,
    mock_genie_facade_for_llm_assisted: MagicMock,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.WARNING, logger=PROCESSOR_LOGGER_NAME)
    await llm_assisted_processor.setup(config={"genie_facade": mock_genie_facade_for_llm_assisted})
    mock_genie_facade_for_llm_assisted._tool_manager.list_tools.return_value = [MockToolForLLMAssisted("actual_tool")]
    mock_genie_facade_for_llm_assisted._tool_manager.get_formatted_tool_definition.return_value = "Tool: Actual Tool..."

    llm_response_content = '{"thought": "Choosing hallucinated tool.", "tool_id": "hallucinated_tool_id", "params": {}}'
    mock_genie_facade_for_llm_assisted.llm.chat.return_value = LLMChatResponse(
        message=ChatMessage(role="assistant", content=llm_response_content),
        raw_response={}
    )

    response = await llm_assisted_processor.process_command("do something")
    assert response.get("chosen_tool_id") is None
    assert any(
        "LLM chose tool 'hallucinated_tool_id' which was not in the candidate list" in rec.message
        for rec in caplog.records
    )
    assert "(Note: LLM hallucinated a tool_id not in the provided list. Corrected to no tool.)" in response.get("llm_thought_process", "")


@pytest.mark.asyncio
async def test_process_command_llm_returns_invalid_params_type(
    llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin,
    mock_genie_facade_for_llm_assisted: MagicMock,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.WARNING, logger=PROCESSOR_LOGGER_NAME)
    await llm_assisted_processor.setup(config={"genie_facade": mock_genie_facade_for_llm_assisted, "max_llm_retries": 0}) # No retries
    mock_genie_facade_for_llm_assisted._tool_manager.list_tools.return_value = [MockToolForLLMAssisted("tool_abc")]
    mock_genie_facade_for_llm_assisted._tool_manager.get_formatted_tool_definition.return_value = "Tool: Tool ABC..."

    llm_response_content = '{"thought": "Trying invalid params.", "tool_id": "tool_abc", "params": "not_a_dict"}'
    mock_genie_facade_for_llm_assisted.llm.chat.return_value = LLMChatResponse(
        message=ChatMessage(role="assistant", content=llm_response_content),
        raw_response={}
    )

    response = await llm_assisted_processor.process_command("test invalid params")
    assert response.get("chosen_tool_id") == "tool_abc" # Tool ID is valid
    assert response.get("extracted_params") == {} # Params should be empty dict
    assert any(
        "LLM returned invalid 'params' type for tool 'tool_abc'" in rec.message
        for rec in caplog.records
    )
    assert "(Note: LLM returned invalid parameter format. Parameters ignored.)" in response.get("llm_thought_process", "")


@pytest.mark.asyncio
async def test_process_command_llm_response_not_json(
    llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin,
    mock_genie_facade_for_llm_assisted: MagicMock,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.DEBUG, logger=PROCESSOR_LOGGER_NAME)

    await llm_assisted_processor.setup(config={"genie_facade": mock_genie_facade_for_llm_assisted, "max_llm_retries": 0})
    mock_genie_facade_for_llm_assisted._tool_manager.list_tools.return_value = [MockToolForLLMAssisted("any_tool")]
    mock_genie_facade_for_llm_assisted._tool_manager.get_formatted_tool_definition.return_value = "Tool: Any..."

    llm_response_content = "This is just plain text, not JSON."
    mock_genie_facade_for_llm_assisted.llm.chat.return_value = LLMChatResponse(
        message=ChatMessage(role="assistant", content=llm_response_content),
        raw_response={}
    )

    response = await llm_assisted_processor.process_command("test no json")
    assert "LLM response did not contain a recognizable JSON block." in response.get("error", "")

    assert any(
        rec.name == PROCESSOR_LOGGER_NAME and
        rec.levelno == logging.DEBUG and
        "Could not extract any valid JSON block from text" in rec.message and
        llm_response_content in rec.message
        for rec in caplog.records
    ), "Expected DEBUG log from _extract_json_block not found or incorrect."

    assert any(
        rec.name == PROCESSOR_LOGGER_NAME and
        rec.levelno == logging.WARNING and
        "Could not extract a JSON block from LLM response." in rec.message and
        llm_response_content in rec.message
        for rec in caplog.records
    ), "Expected WARNING log from process_command about JSON extraction failure not found or incorrect."


@pytest.mark.asyncio
async def test_process_command_llm_call_fails_with_retry(
    llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin,
    mock_genie_facade_for_llm_assisted: MagicMock,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR, logger=PROCESSOR_LOGGER_NAME)
    await llm_assisted_processor.setup(config={"genie_facade": mock_genie_facade_for_llm_assisted, "max_llm_retries": 1})
    mock_genie_facade_for_llm_assisted._tool_manager.list_tools.return_value = [MockToolForLLMAssisted("any_tool")]
    mock_genie_facade_for_llm_assisted._tool_manager.get_formatted_tool_definition.return_value = "Tool: Any..."

    mock_genie_facade_for_llm_assisted.llm.chat.side_effect = [
        RuntimeError("Simulated LLM API error"),
        LLMChatResponse(
            message=ChatMessage(role="assistant", content='{"thought": "Success on retry.", "tool_id": "any_tool", "params": {}}'),
            raw_response={}
        )
    ]
    with patch("asyncio.sleep", AsyncMock()):
        response = await llm_assisted_processor.process_command("test retry")

    assert response.get("error") is None
    assert response.get("chosen_tool_id") == "any_tool"
    assert any(
        "Error during LLM call for tool selection (attempt 1): Simulated LLM API error" in rec.message
        for rec in caplog.records
    )
    assert mock_genie_facade_for_llm_assisted.llm.chat.call_count == 2


@pytest.mark.asyncio
async def test_process_command_llm_fails_all_retries(
    llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin,
    mock_genie_facade_for_llm_assisted: MagicMock,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR, logger=PROCESSOR_LOGGER_NAME)
    await llm_assisted_processor.setup(config={"genie_facade": mock_genie_facade_for_llm_assisted, "max_llm_retries": 1})
    mock_genie_facade_for_llm_assisted._tool_manager.list_tools.return_value = [MockToolForLLMAssisted("any_tool")]
    mock_genie_facade_for_llm_assisted._tool_manager.get_formatted_tool_definition.return_value = "Tool: Any..."

    mock_genie_facade_for_llm_assisted.llm.chat.side_effect = RuntimeError("Persistent LLM API error")

    with patch("asyncio.sleep", AsyncMock()):
        response = await llm_assisted_processor.process_command("test all retries fail")

    assert "Failed to process command with LLM after multiple retries: Persistent LLM API error" in response.get("error", "")
    assert mock_genie_facade_for_llm_assisted.llm.chat.call_count == 2


@pytest.mark.asyncio
async def test_process_command_no_genie_facade(
    llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin
):
    await llm_assisted_processor.setup(config={})
    response = await llm_assisted_processor.process_command("any command")
    assert f"{llm_assisted_processor.plugin_id} not properly set up (Genie facade missing)." in response.get("error", "")


@pytest.mark.asyncio
async def test_process_command_no_tool_definitions_available(
    llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin,
    mock_genie_facade_for_llm_assisted: MagicMock,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.INFO, logger=PROCESSOR_LOGGER_NAME)
    await llm_assisted_processor.setup(config={"genie_facade": mock_genie_facade_for_llm_assisted})
    mock_genie_facade_for_llm_assisted._tool_manager.list_tools.return_value = []
    mock_genie_facade_for_llm_assisted._tool_manager.get_formatted_tool_definition.return_value = None

    response = await llm_assisted_processor.process_command("any command")

    assert "No tools processable." in response.get("error", "")
    assert any(
        "No candidate tools to present to LLM." in rec.message
        for rec in caplog.records
    )
    mock_genie_facade_for_llm_assisted.llm.chat.assert_not_called()

@pytest.mark.asyncio
async def test_process_command_llm_tool_id_ok_params_missing(
    llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin,
    mock_genie_facade_for_llm_assisted: MagicMock
):
    await llm_assisted_processor.setup(config={"genie_facade": mock_genie_facade_for_llm_assisted})
    mock_genie_facade_for_llm_assisted._tool_manager.list_tools.return_value = [MockToolForLLMAssisted("tool_x")]
    mock_genie_facade_for_llm_assisted._tool_manager.get_formatted_tool_definition.return_value = "Tool: X"
    llm_response_content = '{"thought": "Tool X, no params needed.", "tool_id": "tool_x"}' # Params field missing
    mock_genie_facade_for_llm_assisted.llm.chat.return_value = LLMChatResponse(
        message=ChatMessage(role="assistant", content=llm_response_content), raw_response={}
    )
    response = await llm_assisted_processor.process_command("do X")
    assert response.get("chosen_tool_id") == "tool_x"
    assert response.get("extracted_params") == {} # Should default to empty dict

@pytest.mark.asyncio
async def test_process_command_llm_tool_id_null_params_present(
    llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin,
    mock_genie_facade_for_llm_assisted: MagicMock
):
    await llm_assisted_processor.setup(config={"genie_facade": mock_genie_facade_for_llm_assisted})
    llm_response_content = '{"thought": "No tool, but here are params.", "tool_id": null, "params": {"erroneous": "data"}}'
    mock_genie_facade_for_llm_assisted.llm.chat.return_value = LLMChatResponse(
        message=ChatMessage(role="assistant", content=llm_response_content), raw_response={}
    )
    response = await llm_assisted_processor.process_command("no tool query")
    assert response.get("chosen_tool_id") is None
    assert response.get("extracted_params") == {} # Params should be ignored if no tool_id
