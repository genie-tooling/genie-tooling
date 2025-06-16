### tests/unit/invocation_strategies/test_tool_invoker.py
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import StructuredError
from genie_tooling.error_formatters.abc import ErrorFormatter
from genie_tooling.invocation.invoker import (
    DEFAULT_INVOKER_ERROR_FORMATTER_ID,
    DEFAULT_STRATEGY_ID,
    ToolInvoker,
)
from genie_tooling.invocation_strategies.abc import InvocationStrategy
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.abc import Tool
from genie_tooling.tools.manager import ToolManager


@pytest.fixture()
def mock_tool_manager_fixture(mocker) -> ToolManager:
    tm = mocker.MagicMock(spec=ToolManager)
    tm.get_tool = AsyncMock()
    return tm

@pytest.fixture()
def mock_plugin_manager_for_invoker_fixture(mocker) -> PluginManager:
    pm = mocker.MagicMock(spec=PluginManager)
    pm.get_plugin_instance = AsyncMock()
    return pm

@pytest.fixture()
def mock_key_provider_for_invoker_fixture(mocker) -> KeyProvider:
    kp = mocker.AsyncMock(spec=KeyProvider)
    kp.get_key = AsyncMock(return_value="dummy_key")
    return kp

@pytest.fixture()
def tool_invoker_fixture(mock_tool_manager_fixture: ToolManager, mock_plugin_manager_for_invoker_fixture: PluginManager) -> ToolInvoker:
    return ToolInvoker(tool_manager=mock_tool_manager_fixture, plugin_manager=mock_plugin_manager_for_invoker_fixture, default_strategy_id=DEFAULT_STRATEGY_ID)

@pytest.mark.asyncio()
async def test_tool_invoker_invoke_success(tool_invoker_fixture: ToolInvoker, mock_tool_manager_fixture: ToolManager, mock_plugin_manager_for_invoker_fixture: PluginManager, mock_key_provider_for_invoker_fixture: KeyProvider):
    tool_id = "test_tool1"
    params = {"p1": "v1"}
    context = {"c1": "cv1"}
    strategy_id = "mock_invocation_strategy_v1"
    strategy_expected_result = "strategy_executed_successfully"

    mock_tool_instance = AsyncMock(spec=Tool)
    mock_tool_manager_fixture.get_tool.return_value = mock_tool_instance

    mock_strategy_instance = AsyncMock(spec=InvocationStrategy)
    mock_strategy_instance.invoke = AsyncMock(return_value=strategy_expected_result)
    mock_plugin_manager_for_invoker_fixture.get_plugin_instance.return_value = mock_strategy_instance

    result = await tool_invoker_fixture.invoke(
        tool_identifier=tool_id, params=params, key_provider=mock_key_provider_for_invoker_fixture,
        context=context, strategy_id=strategy_id
    )

    mock_tool_manager_fixture.get_tool.assert_awaited_once_with(tool_id)
    mock_plugin_manager_for_invoker_fixture.get_plugin_instance.assert_awaited_once_with(strategy_id, config=ANY)
    mock_strategy_instance.invoke.assert_awaited_once_with(
        tool=mock_tool_instance, params=params, key_provider=mock_key_provider_for_invoker_fixture,
        context=context, invoker_config=ANY
    )
    assert result == strategy_expected_result

@pytest.mark.asyncio()
async def test_tool_invoker_strategy_not_found(tool_invoker_fixture: ToolInvoker, mock_tool_manager_fixture: ToolManager, mock_plugin_manager_for_invoker_fixture: PluginManager, mock_key_provider_for_invoker_fixture: KeyProvider):
    tool_id = "test_tool1"
    mock_tool_instance = AsyncMock(spec=Tool)
    mock_tool_manager_fixture.get_tool.return_value = mock_tool_instance

    mock_error_formatter = MagicMock(spec=ErrorFormatter)
    mock_error_formatter.format = MagicMock(return_value="Formatted: Strategy not found")

    async def get_instance_side_effect(plugin_id: str, config=None):
        if plugin_id == "non_existent_strategy":
            return None
        if plugin_id == DEFAULT_INVOKER_ERROR_FORMATTER_ID:
            return mock_error_formatter
        # For this test, we don't care about other plugins, so a generic mock is fine
        return MagicMock()

    mock_plugin_manager_for_invoker_fixture.get_plugin_instance.side_effect = get_instance_side_effect

    result = await tool_invoker_fixture.invoke(
        tool_identifier=tool_id, params={}, key_provider=mock_key_provider_for_invoker_fixture,
        strategy_id="non_existent_strategy"
    )

    mock_plugin_manager_for_invoker_fixture.get_plugin_instance.assert_any_call("non_existent_strategy", config=ANY)
    mock_plugin_manager_for_invoker_fixture.get_plugin_instance.assert_any_call(DEFAULT_INVOKER_ERROR_FORMATTER_ID)
    assert result == "Formatted: Strategy not found"
    mock_error_formatter.format.assert_called_once()
    structured_error_arg: StructuredError = mock_error_formatter.format.call_args[0][0]
    assert structured_error_arg["type"] == "ConfigurationError"
    assert "InvocationStrategy 'non_existent_strategy' not found or invalid." in structured_error_arg["message"]


@pytest.mark.asyncio()
async def test_tool_invoker_tool_not_found(
    tool_invoker_fixture: ToolInvoker,
    mock_tool_manager_fixture: ToolManager,
    mock_plugin_manager_for_invoker_fixture: PluginManager,
    mock_key_provider_for_invoker_fixture: KeyProvider
):
    """Test that if the tool isn't found, an error is formatted and returned."""
    tool_id = "unknown_tool"
    mock_tool_manager_fixture.get_tool.return_value = None  # Simulate tool not found

    mock_error_formatter = MagicMock(spec=ErrorFormatter)
    mock_error_formatter.format = MagicMock(return_value="Formatted: Tool not found")
    # Setup PM to return the formatter when asked
    mock_plugin_manager_for_invoker_fixture.get_plugin_instance.return_value = mock_error_formatter

    result = await tool_invoker_fixture.invoke(
        tool_identifier=tool_id,
        params={},
        key_provider=mock_key_provider_for_invoker_fixture
    )

    mock_tool_manager_fixture.get_tool.assert_awaited_once_with(tool_id)
    # It should not try to get a strategy if the tool is not found first
    # But it WILL try to get the error formatter
    mock_plugin_manager_for_invoker_fixture.get_plugin_instance.assert_awaited_once_with(DEFAULT_INVOKER_ERROR_FORMATTER_ID)
    mock_error_formatter.format.assert_called_once()
    structured_error_arg: StructuredError = mock_error_formatter.format.call_args[0][0]
    assert structured_error_arg["type"] == "ToolNotFound"
    assert result == "Formatted: Tool not found"


@pytest.mark.asyncio()
async def test_tool_invoker_strategy_invoke_raises_exception(
    tool_invoker_fixture: ToolInvoker,
    mock_tool_manager_fixture: ToolManager,
    mock_plugin_manager_for_invoker_fixture: PluginManager,
    mock_key_provider_for_invoker_fixture: KeyProvider
):
    """Test handling when the strategy's invoke method itself raises an unhandled error."""
    tool_id = "test_tool_strat_fail"
    mock_tool_instance = AsyncMock(spec=Tool)
    mock_tool_manager_fixture.get_tool.return_value = mock_tool_instance

    mock_strategy_instance = AsyncMock(spec=InvocationStrategy)
    mock_strategy_instance.invoke.side_effect = Exception("Critical strategy failure")
    mock_error_formatter = MagicMock(spec=ErrorFormatter)
    mock_error_formatter.format = MagicMock(return_value="Formatted: Critical strategy failure")

    async def get_instance_side_effect(plugin_id: str, config=None):
        if plugin_id == DEFAULT_STRATEGY_ID:
            return mock_strategy_instance
        if plugin_id == DEFAULT_INVOKER_ERROR_FORMATTER_ID:
            return mock_error_formatter
        return MagicMock()

    mock_plugin_manager_for_invoker_fixture.get_plugin_instance.side_effect = get_instance_side_effect

    result = await tool_invoker_fixture.invoke(
        tool_identifier=tool_id,
        params={},
        key_provider=mock_key_provider_for_invoker_fixture
    )

    assert result == "Formatted: Critical strategy failure"
    mock_strategy_instance.invoke.assert_awaited_once()
    mock_error_formatter.format.assert_called_once()
    structured_error_arg: StructuredError = mock_error_formatter.format.call_args[0][0]
    assert structured_error_arg["type"] == "InternalExecutionError"
    assert "Critical unhandled error during strategy execution" in structured_error_arg["message"]
