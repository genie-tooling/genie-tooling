"""Unit tests for the ToolInvoker."""
import logging
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import Plugin, StructuredError
# Updated import paths for DEFAULT_INVOKER_ERROR_FORMATTER_ID and LLMErrorFormatter
from genie_tooling.error_formatters.impl.llm_formatter import LLMErrorFormatter
from genie_tooling.invocation import DEFAULT_INVOKER_ERROR_FORMATTER_ID
from genie_tooling.invocation.invoker import DEFAULT_STRATEGY_ID, ToolInvoker
# Updated import path for InvocationStrategy
from genie_tooling.invocation_strategies.abc import InvocationStrategy
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.abc import Tool as ToolPlugin
from genie_tooling.tools.manager import ToolManager


class MockToolForInvoker(ToolPlugin, Plugin):
    identifier: str
    plugin_id: str

    def __init__(self, identifier="mock_tool", metadata=None, execute_result="executed"):
        self.identifier = identifier
        self.plugin_id = identifier
        self._metadata = metadata or {"identifier": identifier, "input_schema": {}, "output_schema": {}}
        self._execute_result = execute_result
    async def get_metadata(self): return self._metadata
    async def execute(self, params, key_provider, context=None):
        if isinstance(self._execute_result, Exception):
            raise self._execute_result
        return self._execute_result
    async def setup(self, config: Optional[Dict[str, Any]]=None):pass
    async def teardown(self):pass

class MockInvocationStrategy(InvocationStrategy, Plugin):
    plugin_id = "mock_invocation_strategy_v1"
    description = "Mock strategy for testing."

    def __init__(self):
        self.invoke_mock_call_args_list = []
        self.invoke_should_raise: Optional[Exception] = None

    async def invoke(self, tool, params, key_provider, context, invoker_config):
        self.invoke_mock_call_args_list.append({
            "tool": tool, "params": params, "key_provider": key_provider,
            "context": context, "invoker_config": invoker_config
        })
        if self.invoke_should_raise:
            raise self.invoke_should_raise
        return "strategy_executed_successfully"

    async def setup(self, config: Optional[Dict[str, Any]]=None):pass
    async def teardown(self):pass


@pytest.fixture
def mock_tool_manager_fixture(mocker) -> ToolManager:
    tm = mocker.MagicMock(spec=ToolManager)
    tm.get_tool = AsyncMock()
    return tm

@pytest.fixture
def mock_plugin_manager_for_invoker_fixture(mocker) -> PluginManager:
    pm = mocker.MagicMock(spec=PluginManager)
    pm.get_plugin_instance = AsyncMock()
    return pm

@pytest.fixture
def mock_key_provider_for_invoker_fixture(mocker) -> KeyProvider:
    kp = mocker.AsyncMock(spec=KeyProvider)
    kp.get_key = AsyncMock(return_value="dummy_key")
    return kp

@pytest.fixture
def tool_invoker_fixture(
    mock_tool_manager_fixture: ToolManager,
    mock_plugin_manager_for_invoker_fixture: PluginManager
) -> ToolInvoker:
    return ToolInvoker(
        tool_manager=mock_tool_manager_fixture,
        plugin_manager=mock_plugin_manager_for_invoker_fixture,
        default_strategy_id=DEFAULT_STRATEGY_ID
    )

@pytest.mark.asyncio
async def test_tool_invoker_invoke_success(
    tool_invoker_fixture: ToolInvoker,
    mock_tool_manager_fixture: ToolManager,
    mock_plugin_manager_for_invoker_fixture: PluginManager,
    mock_key_provider_for_invoker_fixture: KeyProvider
):
    tool_id = "test_tool1"
    params = {"p1": "v1"}
    context = {"c1": "cv1"}
    strategy_expected_result = "strategy_executed_successfully"

    mock_tool_instance = MockToolForInvoker(identifier=tool_id)
    mock_tool_manager_fixture.get_tool.return_value = mock_tool_instance

    mock_strategy_instance = MockInvocationStrategy()
    mock_default_formatter = LLMErrorFormatter()

    def side_effect_for_get_plugin(plugin_id_req, config=None, **kwargs):
        if plugin_id_req == "mock_invocation_strategy_v1":
            return mock_strategy_instance
        if plugin_id_req == DEFAULT_STRATEGY_ID:
             return mock_strategy_instance
        if plugin_id_req == DEFAULT_INVOKER_ERROR_FORMATTER_ID:
            return mock_default_formatter
        return MagicMock()

    mock_plugin_manager_for_invoker_fixture.get_plugin_instance.side_effect = side_effect_for_get_plugin

    result = await tool_invoker_fixture.invoke(
        tool_identifier=tool_id,
        params=params,
        key_provider=mock_key_provider_for_invoker_fixture,
        context=context,
        strategy_id="mock_invocation_strategy_v1"
    )

    mock_tool_manager_fixture.get_tool.assert_awaited_once_with(tool_id)
    mock_plugin_manager_for_invoker_fixture.get_plugin_instance.assert_any_call("mock_invocation_strategy_v1")

    assert len(mock_strategy_instance.invoke_mock_call_args_list) == 1
    strategy_call_args = mock_strategy_instance.invoke_mock_call_args_list[0]

    assert strategy_call_args["tool"] is mock_tool_instance
    assert strategy_call_args["params"] == params
    assert strategy_call_args["key_provider"] is mock_key_provider_for_invoker_fixture
    assert strategy_call_args["context"] == context
    assert strategy_call_args["invoker_config"]["plugin_manager"] is mock_plugin_manager_for_invoker_fixture

    assert result == strategy_expected_result


@pytest.mark.asyncio
async def test_tool_invoker_tool_not_found(
    tool_invoker_fixture: ToolInvoker,
    mock_tool_manager_fixture: ToolManager,
    mock_plugin_manager_for_invoker_fixture: PluginManager,
    mock_key_provider_for_invoker_fixture: KeyProvider
):
    tool_id = "unknown_tool"
    mock_tool_manager_fixture.get_tool.return_value = None

    mock_error_formatter = LLMErrorFormatter()
    mock_error_formatter.format = MagicMock(return_value="Formatted: Tool not found by invoker")

    async def get_formatter_side_effect(plugin_id: str, config=None, **kwargs):
        if plugin_id == DEFAULT_INVOKER_ERROR_FORMATTER_ID:
            return mock_error_formatter
        return MagicMock()
    mock_plugin_manager_for_invoker_fixture.get_plugin_instance.side_effect = get_formatter_side_effect

    result = await tool_invoker_fixture.invoke(tool_id, {}, mock_key_provider_for_invoker_fixture)

    mock_tool_manager_fixture.get_tool.assert_awaited_once_with(tool_id)
    mock_plugin_manager_for_invoker_fixture.get_plugin_instance.assert_any_call(DEFAULT_INVOKER_ERROR_FORMATTER_ID)
    assert result == "Formatted: Tool not found by invoker"
    mock_error_formatter.format.assert_called_once()
    structured_error_arg: StructuredError = mock_error_formatter.format.call_args[0][0]
    assert structured_error_arg["type"] == "ToolNotFound"


@pytest.mark.asyncio
async def test_tool_invoker_strategy_not_found(
    tool_invoker_fixture: ToolInvoker,
    mock_tool_manager_fixture: ToolManager,
    mock_plugin_manager_for_invoker_fixture: PluginManager,
    mock_key_provider_for_invoker_fixture: KeyProvider
):
    tool_id = "test_tool1"
    mock_tool_instance = MockToolForInvoker(identifier=tool_id)
    mock_tool_manager_fixture.get_tool.return_value = mock_tool_instance

    mock_error_formatter = LLMErrorFormatter()
    mock_error_formatter.format = MagicMock(return_value="Formatted: Strategy not found")

    async def get_instance_side_effect(plugin_id: str, config=None, **kwargs):
        if plugin_id == "non_existent_strategy":
            return None
        if plugin_id == DEFAULT_INVOKER_ERROR_FORMATTER_ID:
            return mock_error_formatter
        return MagicMock()
    mock_plugin_manager_for_invoker_fixture.get_plugin_instance.side_effect = get_instance_side_effect

    result = await tool_invoker_fixture.invoke(
        tool_identifier=tool_id, params={}, key_provider=mock_key_provider_for_invoker_fixture,
        strategy_id="non_existent_strategy"
    )

    mock_plugin_manager_for_invoker_fixture.get_plugin_instance.assert_any_call("non_existent_strategy")
    mock_plugin_manager_for_invoker_fixture.get_plugin_instance.assert_any_call(DEFAULT_INVOKER_ERROR_FORMATTER_ID)
    assert result == "Formatted: Strategy not found"
    mock_error_formatter.format.assert_called_once()
    structured_error_arg: StructuredError = mock_error_formatter.format.call_args[0][0]
    assert structured_error_arg["type"] == "ConfigurationError"


@pytest.mark.asyncio
async def test_tool_invoker_default_error_formatter_fails_to_load(
    tool_invoker_fixture: ToolInvoker,
    mock_tool_manager_fixture: ToolManager,
    mock_plugin_manager_for_invoker_fixture: PluginManager,
    mock_key_provider_for_invoker_fixture: KeyProvider,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR)
    tool_id = "unknown_tool_formatter_fail"
    mock_tool_manager_fixture.get_tool.return_value = None # Trigger ToolNotFound

    # Simulate get_plugin_instance raising an error when trying to load the formatter
    formatter_load_exception = Exception("Simulated formatter load failure in get_plugin_instance")
    async def get_instance_formatter_fail(plugin_id: str, config=None, **kwargs):
        if plugin_id == DEFAULT_INVOKER_ERROR_FORMATTER_ID:
            raise formatter_load_exception
        # Allow other plugins (like a strategy if it were called on a success path) to be mocked
        return MagicMock()
    mock_plugin_manager_for_invoker_fixture.get_plugin_instance.side_effect = get_instance_formatter_fail

    result = await tool_invoker_fixture.invoke(tool_id, {}, mock_key_provider_for_invoker_fixture)

    # Check that the logger in _get_default_error_formatter was called
    expected_log_message = (
        f"Failed to load default error formatter '{DEFAULT_INVOKER_ERROR_FORMATTER_ID}': "
        f"{str(formatter_load_exception)}"
    )
    assert any(expected_log_message in record.message for record in caplog.records if "Traceback" not in record.message)

    # Since the formatter failed, the result should be the raw StructuredError
    assert isinstance(result, dict)
    assert result["type"] == "ToolNotFound"
    assert result["message"] == f"Tool '{tool_id}' not found."


@pytest.mark.asyncio
async def test_tool_invoker_strategy_invoke_raises_exception(
    tool_invoker_fixture: ToolInvoker,
    mock_tool_manager_fixture: ToolManager,
    mock_plugin_manager_for_invoker_fixture: PluginManager,
    mock_key_provider_for_invoker_fixture: KeyProvider
):
    tool_id = "tool_strat_exception"
    mock_tool_instance = MockToolForInvoker(identifier=tool_id)
    mock_tool_manager_fixture.get_tool.return_value = mock_tool_instance

    mock_strategy_instance = MockInvocationStrategy()
    mock_strategy_instance.invoke_should_raise = RuntimeError("Strategy crashed!")

    mock_error_formatter = LLMErrorFormatter()
    mock_error_formatter.format = MagicMock(return_value="Formatted: Strategy internal error")


    async def get_instance_side_effect(plugin_id: str, config=None, **kwargs):
        if plugin_id == DEFAULT_STRATEGY_ID:
            return mock_strategy_instance
        if plugin_id == DEFAULT_INVOKER_ERROR_FORMATTER_ID:
            return mock_error_formatter
        return MagicMock()
    mock_plugin_manager_for_invoker_fixture.get_plugin_instance.side_effect = get_instance_side_effect

    result = await tool_invoker_fixture.invoke(tool_id, {}, mock_key_provider_for_invoker_fixture)

    assert result == "Formatted: Strategy internal error"
    mock_error_formatter.format.assert_called_once()
    structured_error_arg: StructuredError = mock_error_formatter.format.call_args[0][0]
    assert structured_error_arg["type"] == "InternalExecutionError"
    assert "Critical unhandled error during strategy execution" in structured_error_arg["message"]
    assert "Strategy crashed!" in structured_error_arg["message"]
