### tests/unit/command_processors/test_llm_assisted_processor.py
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest
from genie_tooling.command_processors.impl.llm_assisted_processor import (
    DEFAULT_SYSTEM_PROMPT_TEMPLATE,
    LLMAssistedToolSelectionProcessorPlugin,
)
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.core.types import Plugin as CorePluginType
from genie_tooling.lookup.types import RankedToolResult
from genie_tooling.tools.abc import Tool as ToolPlugin

PROCESSOR_LOGGER_NAME = "genie_tooling.command_processors.impl.llm_assisted_processor"

# --- Mocks ---
class MockToolForLLMAssisted(ToolPlugin, CorePluginType):
    def __init__(self, identifier: str, name: str, description: str):
        self._identifier_val = identifier
        self._name_val = name
        self._description_val = description

    @property
    def plugin_id(self) -> str:
        return self._identifier_val

    @property
    def identifier(self) -> str:
        return self._identifier_val

    async def get_metadata(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier, "name": self._name_val,
            "description_llm": self._description_val,
            "input_schema": {"type": "object", "properties": {}},
            "output_schema": {"type": "object", "properties": {}}
        }
    async def execute(self, params, key_provider, context=None) -> Any: return "executed"
    async def setup(self, config=None): pass
    async def teardown(self): pass


@pytest.fixture()
def mock_genie_facade_for_llm_proc(mocker) -> MagicMock:
    genie = mocker.MagicMock(name="MockGenieFacadeForLLMProc")

    default_mock_tool = MockToolForLLMAssisted("default_mock_tool_id", "Default Mock Tool", "A default tool for testing.")
    genie._tool_manager = AsyncMock(name="MockToolManager")
    genie._tool_manager.list_tools = AsyncMock(return_value=[default_mock_tool])
    genie._tool_manager.get_formatted_tool_definition = AsyncMock(return_value=f"Formatted {default_mock_tool.identifier}")

    genie._tool_lookup_service = AsyncMock(name="MockToolLookupService")
    genie._tool_lookup_service.find_tools = AsyncMock(return_value=[])

    genie.llm = AsyncMock(name="MockLLMInterface")
    default_llm_json_output = json.dumps({"thought": "No specific tool seems appropriate.", "tool_id": None, "params": None})
    genie.llm.chat = AsyncMock(return_value={"message": {"content": default_llm_json_output}})

    genie.prompts = AsyncMock(name="MockPromptInterface")
    genie.prompts.render_prompt = AsyncMock(return_value=DEFAULT_SYSTEM_PROMPT_TEMPLATE)

    genie.observability = AsyncMock(name="MockObservabilityInterface")
    genie.observability.trace_event = AsyncMock()

    genie._config = MiddlewareConfig()
    genie._config.default_tool_indexing_formatter_id = "compact_text_formatter_plugin_v1"
    return genie

@pytest.fixture()
async def llm_assisted_processor() -> LLMAssistedToolSelectionProcessorPlugin:
    processor = LLMAssistedToolSelectionProcessorPlugin()
    return processor

# --- Tests ---

@pytest.mark.asyncio()
async def test_setup_no_genie_facade(
    llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin,
    caplog: pytest.LogCaptureFixture
):
    processor = await llm_assisted_processor
    caplog.set_level(logging.INFO, logger=PROCESSOR_LOGGER_NAME)
    await processor.setup(config={})
    assert processor._genie is None
    assert "Genie facade not found in config" in caplog.text

@pytest.mark.asyncio()
async def test_setup_with_genie_and_custom_config(
    llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin,
    mock_genie_facade_for_llm_proc: MagicMock
):
    processor = await llm_assisted_processor
    custom_config = {
        "genie_facade": mock_genie_facade_for_llm_proc,
        "llm_provider_id": "custom_llm",
        "tool_formatter_id": "custom_formatter",
        "tool_lookup_top_k": 5,
        "system_prompt_template": "Custom prompt: {tool_definitions_string}",
        "max_llm_retries": 2
    }
    await processor.setup(config=custom_config)
    assert processor._genie is mock_genie_facade_for_llm_proc
    assert processor._llm_provider_id == "custom_llm"
    assert processor._tool_formatter_id == "custom_formatter"
    assert processor._tool_lookup_top_k == 5
    assert processor._system_prompt_template == "Custom prompt: {tool_definitions_string}"
    assert processor._max_llm_retries == 2

@pytest.mark.asyncio()
class TestGetToolDefinitionsString:
    async def test_no_genie_facade(self, llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin):
        processor = await llm_assisted_processor
        processor._genie = None
        # FIX: The first argument to the method is the genie instance, which is now None.
        # The other arguments 'command' and 'correlation_id' must be provided.
        defs, ids = await processor._get_tool_definitions_string(None, "cmd", correlation_id="test-id") # type: ignore
        assert "Error: Genie facade not available." in defs
        assert ids == []

    async def test_no_tools_available(
        self, llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin, mock_genie_facade_for_llm_proc: MagicMock
    ):
        processor = await llm_assisted_processor
        await processor.setup({"genie_facade": mock_genie_facade_for_llm_proc})
        mock_genie_facade_for_llm_proc._tool_manager.list_tools.return_value = []
        # FIX: Provide all required arguments to the method call.
        defs, ids = await processor._get_tool_definitions_string(mock_genie_facade_for_llm_proc, "cmd", correlation_id="test-id")
        assert "No tools available." in defs
        assert ids == []

    async def test_tool_lookup_returns_tools(
        self, llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin, mock_genie_facade_for_llm_proc: MagicMock
    ):
        processor = await llm_assisted_processor
        tool1 = MockToolForLLMAssisted("tool1", "Tool One", "Desc1")
        tool2 = MockToolForLLMAssisted("tool2", "Tool Two", "Desc2")
        mock_genie_facade_for_llm_proc._tool_manager.list_tools.return_value = [tool1, tool2]
        mock_genie_facade_for_llm_proc._tool_lookup_service.find_tools.return_value = [
            RankedToolResult("tool1", 0.9, {}),
        ]
        mock_genie_facade_for_llm_proc._tool_manager.get_formatted_tool_definition.side_effect = lambda tid, fid: f"Formatted {tid}"

        await processor.setup({"genie_facade": mock_genie_facade_for_llm_proc, "tool_lookup_top_k": 1})
        # FIX: Provide all required arguments to the method call.
        defs, ids = await processor._get_tool_definitions_string(mock_genie_facade_for_llm_proc, "find tool1", correlation_id="test-id")

        assert defs == "Formatted tool1"
        assert ids == ["tool1"]
        mock_genie_facade_for_llm_proc._tool_lookup_service.find_tools.assert_awaited_once_with("find tool1", top_k=1, indexing_formatter_id_override=ANY)

    async def test_tool_lookup_no_results_fallback_all(
        self, llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin, mock_genie_facade_for_llm_proc: MagicMock
    ):
        processor = await llm_assisted_processor
        tool1 = MockToolForLLMAssisted("tool1", "Tool One", "Desc1")
        tool2 = MockToolForLLMAssisted("tool2", "Tool Two", "Desc2")
        mock_genie_facade_for_llm_proc._tool_manager.list_tools.return_value = [tool1, tool2]
        mock_genie_facade_for_llm_proc._tool_lookup_service.find_tools.return_value = []
        mock_genie_facade_for_llm_proc._tool_manager.get_formatted_tool_definition.side_effect = lambda tid, fid: f"Formatted {tid}"

        await processor.setup({"genie_facade": mock_genie_facade_for_llm_proc, "tool_lookup_top_k": 3})
        # FIX: Provide all required arguments to the method call.
        defs, ids = await processor._get_tool_definitions_string(mock_genie_facade_for_llm_proc, "unrelated query", correlation_id="test-id")

        assert "Formatted tool1" in defs
        assert "Formatted tool2" in defs
        assert sorted(ids) == ["tool1", "tool2"]

    async def test_formatter_fails_for_one_tool(
        self, llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin, mock_genie_facade_for_llm_proc: MagicMock
    ):
        processor = await llm_assisted_processor
        tool1 = MockToolForLLMAssisted("tool1", "Tool One", "Desc1")
        tool2 = MockToolForLLMAssisted("tool2_fails_format", "Tool Two", "Desc2")
        mock_genie_facade_for_llm_proc._tool_manager.list_tools.return_value = [tool1, tool2]
        def format_side_effect(tool_id, formatter_id):
            if tool_id == "tool2_fails_format":
                return None
            return f"Formatted {tool_id}"
        mock_genie_facade_for_llm_proc._tool_manager.get_formatted_tool_definition.side_effect = format_side_effect

        await processor.setup({"genie_facade": mock_genie_facade_for_llm_proc, "tool_lookup_top_k": 0})
        # FIX: Provide all required arguments to the method call.
        defs, ids = await processor._get_tool_definitions_string(mock_genie_facade_for_llm_proc, "cmd", correlation_id="test-id")

        assert defs == "Formatted tool1"
        assert ids == ["tool1", "tool2_fails_format"]
        mock_genie_facade_for_llm_proc.observability.trace_event.assert_any_call(
            "log.warning",
            {"message": "Failed to get formatted definition for tool 'tool2_fails_format' using formatter plugin ID 'compact_text_formatter_plugin_v1'."},
            "LLMAssistedToolSelectionProcessor",
            "test-id"
        )


@pytest.mark.asyncio()
class TestExtractJsonBlock:
    @pytest.mark.parametrize(
        ("text_input", "expected_json_str"),
        [
            ('Some text before {"key": "value"} and after.', '{"key": "value"}'),
            ('{"only_json": true}', '{"only_json": true}'),
            ('```json\n{"code_block_json": "data"}\n```', '{"code_block_json": "data"}'),
            ('```\n{"generic_code_block": true}\n```', '{"generic_code_block": true}'),
            ('Text with array: [1, 2, {"key": "val"}] trailing.', '[1, 2, {"key": "val"}]'),
            ("No JSON here.", None),
            ('Malformed {json: "block",', None),
            ('Text with { "inner": { "nested": "value" } } block.', '{ "inner": { "nested": "value" } }'),
            ('{"a":1} some text {"b":2}', '{"a":1}'),
            ('Thought: ... \n```json\n{"thought": "User wants to calculate.", "tool_id": "tool_calc", "params": {"num1": 5, "num2": 3}}\n```',
             '{"thought": "User wants to calculate.", "tool_id": "tool_calc", "params": {"num1": 5, "num2": 3}}'),
            ('Text with multiple JSON blocks: {"first": 1} and then {"second": 2}.', '{"first": 1}'),
            ('Text with nested JSON in text: "outer text {\\"inner_json\\": true}"', None),
            ('```json\n[\n  {"item": 1},\n  {"item": 2}\n]\n```', '[\n  {"item": 1},\n  {"item": 2}\n]'),
        ],
    )
    async def test_extract_various_formats(self, llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin, mock_genie_facade_for_llm_proc: MagicMock, text_input: str, expected_json_str: Optional[str]):
        processor = await llm_assisted_processor
        await processor.setup({"genie_facade": mock_genie_facade_for_llm_proc})
        # FIX: Provide all required arguments to the method call.
        assert await processor._extract_json_block(mock_genie_facade_for_llm_proc, text_input, correlation_id="test-id") == expected_json_str


@pytest.mark.asyncio()
class TestProcessCommand:
    async def test_process_command_success(
        self, llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin, mock_genie_facade_for_llm_proc: MagicMock
    ):
        processor = await llm_assisted_processor
        tool1 = MockToolForLLMAssisted("tool1", "Tool One", "Desc1")
        mock_genie_facade_for_llm_proc._tool_manager.list_tools.return_value = [tool1]
        mock_genie_facade_for_llm_proc._tool_manager.get_formatted_tool_definition.return_value = "Formatted tool1"

        llm_output_content = '```json\n{"thought": "Use tool1", "tool_id": "tool1", "params": {"p": "val"}}\n```'
        mock_genie_facade_for_llm_proc.llm.chat.return_value = {"message": {"content": llm_output_content}}

        await processor.setup({"genie_facade": mock_genie_facade_for_llm_proc})
        response = await processor.process_command("do tool1", correlation_id="test-id", genie_instance=mock_genie_facade_for_llm_proc)

        assert response["chosen_tool_id"] == "tool1"
        assert response["extracted_params"] == {"p": "val"}
        assert response["llm_thought_process"] == "Use tool1"

    async def test_process_command_llm_chooses_null_tool(
        self, llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin, mock_genie_facade_for_llm_proc: MagicMock
    ):
        processor = await llm_assisted_processor
        llm_output_content = '{"thought": "No tool needed for this.", "tool_id": null, "params": null}'
        mock_genie_facade_for_llm_proc.llm.chat.return_value = {"message": {"content": llm_output_content}}
        await processor.setup({"genie_facade": mock_genie_facade_for_llm_proc})
        response = await processor.process_command("just chat", correlation_id="test-id", genie_instance=mock_genie_facade_for_llm_proc)

        assert response.get("chosen_tool_id") is None
        assert response.get("extracted_params") == {}
        assert response.get("llm_thought_process") == "No tool needed for this."

    async def test_process_command_llm_chooses_invalid_tool(
        self, llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin, mock_genie_facade_for_llm_proc: MagicMock
    ):
        processor = await llm_assisted_processor
        tool1 = MockToolForLLMAssisted("actual_tool", "Actual Tool", "Desc")
        mock_genie_facade_for_llm_proc._tool_manager.list_tools.return_value = [tool1]
        mock_genie_facade_for_llm_proc._tool_manager.get_formatted_tool_definition.return_value = "Formatted actual_tool"

        llm_output_content = '{"thought": "Chose wrong tool", "tool_id": "hallucinated_tool", "params": {}}'
        mock_genie_facade_for_llm_proc.llm.chat.return_value = {"message": {"content": llm_output_content}}
        await processor.setup({"genie_facade": mock_genie_facade_for_llm_proc})
        response = await processor.process_command("do something", correlation_id="test-id", genie_instance=mock_genie_facade_for_llm_proc)

        assert response.get("chosen_tool_id") is None
        mock_genie_facade_for_llm_proc.observability.trace_event.assert_any_call(
            "log.warning",
            {"message": "LLM chose tool 'hallucinated_tool' which was not in the candidate list (['actual_tool']). Treating as no tool chosen."},
            "LLMAssistedToolSelectionProcessor",
            "test-id"
        )
        assert "(Note: LLM hallucinated a tool_id not in the provided list. Corrected to no tool.)" in response.get("llm_thought_process", "")

    async def test_process_command_llm_response_not_json(
        self, llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin, mock_genie_facade_for_llm_proc: MagicMock
    ):
        processor = await llm_assisted_processor
        mock_genie_facade_for_llm_proc.llm.chat.return_value = {"message": {"content": "This is not JSON."}}
        await processor.setup({"genie_facade": mock_genie_facade_for_llm_proc, "max_llm_retries": 0})
        response = await processor.process_command("cmd", correlation_id="test-id", genie_instance=mock_genie_facade_for_llm_proc)

        assert response.get("error") == "LLM response did not contain a recognizable JSON block."
        mock_genie_facade_for_llm_proc.observability.trace_event.assert_any_call(
            "log.warning",
            {"message": "Could not extract a JSON block from LLM response. Content: 'This is not JSON.'"},
            "LLMAssistedToolSelectionProcessor",
            "test-id"
        )

    async def test_process_command_llm_call_fails_retries(
        self, llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin, mock_genie_facade_for_llm_proc: MagicMock
    ):
        processor = await llm_assisted_processor
        mock_genie_facade_for_llm_proc.llm.chat.side_effect = RuntimeError("LLM API down")
        await processor.setup({"genie_facade": mock_genie_facade_for_llm_proc, "max_llm_retries": 1})
        response = await processor.process_command("cmd", correlation_id="test-id", genie_instance=mock_genie_facade_for_llm_proc)

        assert "Failed to process command with LLM after multiple retries: LLM API down" in response.get("error", "")
        assert mock_genie_facade_for_llm_proc.llm.chat.call_count == 2

    async def test_process_command_no_tools_available(
        self, llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin, mock_genie_facade_for_llm_proc: MagicMock
    ):
        processor = await llm_assisted_processor
        mock_genie_facade_for_llm_proc._tool_manager.list_tools.return_value = []
        await processor.setup({"genie_facade": mock_genie_facade_for_llm_proc})
        response = await processor.process_command("cmd", correlation_id="test-id", genie_instance=mock_genie_facade_for_llm_proc)
        assert response.get("error") == "No tools processable."
        assert "No tools are available" in response.get("llm_thought_process", "")

    async def test_process_command_with_conversation_history(
        self, llm_assisted_processor: LLMAssistedToolSelectionProcessorPlugin, mock_genie_facade_for_llm_proc: MagicMock
    ):
        processor = await llm_assisted_processor
        tool1 = MockToolForLLMAssisted("tool1", "Tool One", "Desc1")
        mock_genie_facade_for_llm_proc._tool_manager.list_tools.return_value = [tool1]
        mock_genie_facade_for_llm_proc._tool_manager.get_formatted_tool_definition.return_value = "Formatted tool1"

        await processor.setup({"genie_facade": mock_genie_facade_for_llm_proc})
        history = [{"role": "user", "content": "Previous turn"}]

        llm_output_content = '{"thought": "Considering history", "tool_id": null, "params": null}'
        mock_genie_facade_for_llm_proc.llm.chat.return_value = {"message": {"content": llm_output_content}}

        await processor.process_command("Next turn", conversation_history=history, correlation_id="test-id", genie_instance=mock_genie_facade_for_llm_proc)

        mock_genie_facade_for_llm_proc.llm.chat.assert_awaited_once()
        call_args_list = mock_genie_facade_for_llm_proc.llm.chat.call_args_list
        assert len(call_args_list) == 1
        messages_sent_to_llm = call_args_list[0].kwargs["messages"]
        assert len(messages_sent_to_llm) == 3
        assert messages_sent_to_llm[1]["content"] == "Previous turn"