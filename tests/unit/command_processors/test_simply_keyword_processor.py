from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genie_tooling.command_processors.impl.simple_keyword_processor import (
    SimpleKeywordToolSelectorProcessorPlugin,
)
from genie_tooling.llm_providers.types import ChatMessage
from genie_tooling.tools.abc import Tool as ToolPlugin


class MockToolForSimpleProcessor(ToolPlugin):
    _identifier_value: str
    _plugin_id_value: str

    def __init__(self, identifier_val: str, input_schema: Dict[str, Any]):
        self._identifier_value = identifier_val
        self._plugin_id_value = identifier_val
        self._input_schema = input_schema
        self._name = f"MockTool_{self._identifier_value}"

    @property
    def identifier(self) -> str:
        return self._identifier_value

    @property
    def plugin_id(self) -> str:
        return self._plugin_id_value

    async def get_metadata(self) -> Dict[str, Any]:
        return {"identifier": self.identifier, "name": self._name, "description_llm": f"Description for {self._name}", "input_schema": self._input_schema, "output_schema": {"type": "object"}}
    async def execute(self, params: Dict[str, Any], key_provider: Any, context: Optional[Dict[str, Any]] = None) -> Any: return {"result": "mock_tool_executed", "params_received": params}
    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass
    async def teardown(self) -> None: pass

@pytest.fixture()
def mock_genie_facade(mocker) -> MagicMock:
    facade = MagicMock(name="MockGenieFacade")
    # Correctly mock the tool manager and its methods as async
    facade._tool_manager = AsyncMock(name="MockToolManagerInGenie")
    facade._tool_manager.get_tool = AsyncMock(name="MockGetToolInGenie")
    # Also mock the observability interface as it's used in the processor
    facade.observability = AsyncMock(name="MockObservabilityInterface")
    facade.observability.trace_event = AsyncMock()
    return facade

@pytest.fixture()
def processor() -> SimpleKeywordToolSelectorProcessorPlugin:
    return SimpleKeywordToolSelectorProcessorPlugin()

@pytest.mark.asyncio()
async def test_setup_with_keyword_map(processor: SimpleKeywordToolSelectorProcessorPlugin, mock_genie_facade: MagicMock):
    config_params = {"keyword_map": {"calc": "calculator", "weather": "weather_tool"}, "keyword_priority": ["calc", "weather"]}
    await processor.setup(config={"genie_facade": mock_genie_facade, **config_params})
    assert processor._keyword_tool_map == config_params["keyword_map"]
    assert processor._keyword_priority == config_params["keyword_priority"]
    assert processor._genie is mock_genie_facade

@pytest.mark.asyncio()
async def test_process_command_tool_get_metadata_fails(processor: SimpleKeywordToolSelectorProcessorPlugin, mock_genie_facade: MagicMock):
    config_params = {"keyword_map": {"calc": "calculator"}}
    await processor.setup(config={"genie_facade": mock_genie_facade, **config_params})
    mock_tool = MockToolForSimpleProcessor("calculator", {})
    mock_tool.get_metadata = AsyncMock(side_effect=RuntimeError("Failed to get metadata"))
    mock_genie_facade._tool_manager.get_tool.return_value = mock_tool
    response = await processor.process_command("calculate 1+1")
    assert response.get("error") == "Error processing tool 'calculator': Failed to get metadata"

@pytest.mark.asyncio()
@patch("builtins.input")
async def test_process_command_success_with_params(mock_builtin_input: MagicMock, processor: SimpleKeywordToolSelectorProcessorPlugin, mock_genie_facade: MagicMock):
    config_params = {"keyword_map": {"add": "adder_tool"}, "keyword_priority": ["add"]}
    await processor.setup(config={"genie_facade": mock_genie_facade, **config_params})
    adder_schema = {"type": "object", "properties": {"num1": {"type": "number", "description": "First number"}, "num2": {"type": "number", "description": "Second number"}, "op_name": {"type": "string", "description": "Operation", "default": "sum"}}, "required": ["num1", "num2"]}
    mock_adder_tool = MockToolForSimpleProcessor("adder_tool", adder_schema)
    mock_genie_facade._tool_manager.get_tool.return_value = mock_adder_tool
    mock_builtin_input.side_effect = ["5.0", "3", "y", ""]
    response = await processor.process_command("add two numbers")
    assert response.get("error") is None, f"Expected no error, got: {response.get('error')}"
    assert response.get("chosen_tool_id") == "adder_tool"
    assert response.get("extracted_params") == {"num1": 5.0, "num2": 3.0, "op_name": "sum"}
    assert "Selected tool 'adder_tool' based on keyword match." in response.get("llm_thought_process", "")
    assert mock_builtin_input.call_count == 4

@pytest.mark.asyncio()
@patch("builtins.input")
async def test_process_command_param_coercion(mock_builtin_input: MagicMock, processor: SimpleKeywordToolSelectorProcessorPlugin, mock_genie_facade: MagicMock):
    config_params = {"keyword_map": {"test": "type_test_tool"}}
    await processor.setup(config={"genie_facade": mock_genie_facade, **config_params})
    type_schema = {"type": "object", "properties": {"p_int": {"type": "integer"}, "p_float": {"type": "number"}, "p_bool_true": {"type": "boolean"}, "p_bool_false": {"type": "boolean"}, "p_str": {"type": "string"}}, "required": ["p_int", "p_float", "p_bool_true", "p_bool_false", "p_str"]}
    mock_type_tool = MockToolForSimpleProcessor("type_test_tool", type_schema)
    mock_genie_facade._tool_manager.get_tool.return_value = mock_type_tool
    mock_builtin_input.side_effect = ["10", "3.14", "yes", "0", "hello"]
    response = await processor.process_command("test types")
    assert response.get("extracted_params") == {"p_int": 10, "p_float": 3.14, "p_bool_true": True, "p_bool_false": False, "p_str": "hello"}

@pytest.mark.asyncio()
@patch("builtins.input")
async def test_process_command_required_param_not_provided(mock_builtin_input: MagicMock, processor: SimpleKeywordToolSelectorProcessorPlugin, mock_genie_facade: MagicMock):
    config_params = {"keyword_map": {"req": "req_tool"}}
    await processor.setup(config={"genie_facade": mock_genie_facade, **config_params})
    req_schema = {"type": "object", "properties": {"needed": {"type": "string"}}, "required": ["needed"]}
    mock_req_tool = MockToolForSimpleProcessor("req_tool", req_schema)
    mock_genie_facade._tool_manager.get_tool.return_value = mock_req_tool
    mock_builtin_input.return_value = ""
    response = await processor.process_command("test required")
    assert response.get("error") == "Required parameter 'needed' was not provided."
    assert response.get("chosen_tool_id") is None

@pytest.mark.asyncio()
@patch("builtins.input")
async def test_process_command_optional_param_skipped(mock_builtin_input: MagicMock, processor: SimpleKeywordToolSelectorProcessorPlugin, mock_genie_facade: MagicMock):
    config_params = {"keyword_map": {"opt": "opt_tool"}}
    await processor.setup(config={"genie_facade": mock_genie_facade, **config_params})
    opt_schema = {"type": "object", "properties": {"optional_param": {"type": "string"}}}
    mock_opt_tool = MockToolForSimpleProcessor("opt_tool", opt_schema)
    mock_genie_facade._tool_manager.get_tool.return_value = mock_opt_tool
    mock_builtin_input.return_value = "n"
    response = await processor.process_command("test optional")
    assert response.get("error") is None
    assert response.get("chosen_tool_id") == "opt_tool"
    assert response.get("extracted_params") == {}
    assert mock_builtin_input.call_count == 1

@pytest.mark.asyncio()
@patch("builtins.input")
async def test_process_command_optional_param_provided(mock_builtin_input: MagicMock, processor: SimpleKeywordToolSelectorProcessorPlugin, mock_genie_facade: MagicMock):
    config_params = {"keyword_map": {"opt": "opt_tool"}}
    await processor.setup(config={"genie_facade": mock_genie_facade, **config_params})
    opt_schema = {"type": "object", "properties": {"optional_p": {"type": "string"}}}
    mock_opt_tool = MockToolForSimpleProcessor("opt_tool", opt_schema)
    mock_genie_facade._tool_manager.get_tool.return_value = mock_opt_tool
    mock_builtin_input.side_effect = ["y", "my_opt_value"]
    response = await processor.process_command("test optional provide")
    assert response.get("error") is None
    assert response.get("chosen_tool_id") == "opt_tool"
    assert response.get("extracted_params") == {"optional_p": "my_opt_value"}
    assert mock_builtin_input.call_count == 2

@pytest.mark.asyncio()
@patch("builtins.input")
async def test_tool_with_no_params(mock_builtin_input: MagicMock, processor: SimpleKeywordToolSelectorProcessorPlugin, mock_genie_facade: MagicMock):
    config_params = {"keyword_map": {"noparam": "no_param_tool"}}
    await processor.setup(config={"genie_facade": mock_genie_facade, **config_params})
    no_param_schema = {"type": "object", "properties": {}}
    mock_no_param_tool = MockToolForSimpleProcessor("no_param_tool", no_param_schema)
    mock_genie_facade._tool_manager.get_tool.return_value = mock_no_param_tool
    response = await processor.process_command("run noparam tool")
    assert response.get("error") is None
    assert response.get("chosen_tool_id") == "no_param_tool"
    assert response.get("extracted_params") == {}
    mock_builtin_input.assert_not_called()

@pytest.mark.asyncio()
@patch("builtins.input")
async def test_param_with_enum_in_prompt(mock_builtin_input: MagicMock, processor: SimpleKeywordToolSelectorProcessorPlugin, mock_genie_facade: MagicMock):
    config_params = {"keyword_map": {"enum_test": "enum_tool"}}
    await processor.setup(config={"genie_facade": mock_genie_facade, **config_params})
    enum_schema = {"type": "object", "properties": {"choice": {"type": "string", "description": "Pick one.", "enum": ["A", "B", "C"]}}, "required": ["choice"]}
    mock_enum_tool = MockToolForSimpleProcessor("enum_tool", enum_schema)
    mock_genie_facade._tool_manager.get_tool.return_value = mock_enum_tool
    mock_builtin_input.return_value = "B"
    await processor.process_command("do enum_test")
    assert mock_builtin_input.call_count == 1
    prompt_message_arg = mock_builtin_input.call_args[0][0]
    assert "(choices: A, B, C)" in prompt_message_arg

@pytest.mark.asyncio()
async def test_process_command_with_conversation_history_ignored(processor: SimpleKeywordToolSelectorProcessorPlugin, mock_genie_facade: MagicMock):
    config_params = {"keyword_map": {"greet": "greeting_tool"}}
    await processor.setup(config={"genie_facade": mock_genie_facade, **config_params})
    mock_greeting_tool = MockToolForSimpleProcessor("greeting_tool", {"type": "object", "properties": {}})
    mock_genie_facade._tool_manager.get_tool.return_value = mock_greeting_tool
    history: List[ChatMessage] = [{"role": "user", "content": "Previous message"}]
    response = await processor.process_command("greet me", conversation_history=history)
    assert response.get("error") is None
    assert response.get("chosen_tool_id") == "greeting_tool"
