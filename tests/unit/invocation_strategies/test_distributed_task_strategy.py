import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.invocation_strategies.impl.distributed_task_strategy import (
    GENERIC_TOOL_EXECUTION_TASK_NAME,
    DistributedTaskInvocationStrategy,
)
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.task_queues.abc import DistributedTaskQueuePlugin
from genie_tooling.tools.abc import Tool

STRATEGY_LOGGER_NAME = "genie_tooling.invocation_strategies.impl.distributed_task_strategy"


@pytest.fixture
def mock_task_queue_plugin() -> AsyncMock:
    plugin = AsyncMock(spec=DistributedTaskQueuePlugin)
    plugin.plugin_id = "mock_task_queue_v1"
    plugin.submit_task = AsyncMock(return_value="mock_task_123")
    plugin.get_task_status = AsyncMock(return_value="success")
    plugin.get_task_result = AsyncMock(return_value={"result": "task completed successfully"})
    return plugin


@pytest.fixture
def mock_plugin_manager_for_dist_strat(mock_task_queue_plugin: AsyncMock) -> AsyncMock:
    pm = AsyncMock(spec=PluginManager)
    pm.get_plugin_instance = AsyncMock(return_value=mock_task_queue_plugin)
    return pm


@pytest.fixture
def mock_tool_for_dist_strat() -> MagicMock:
    tool = MagicMock(spec=Tool)
    tool.identifier = "distributed_test_tool"
    return tool


@pytest.fixture
def mock_key_provider_for_dist_strat() -> MagicMock:
    return MagicMock(spec=KeyProvider)


@pytest.mark.asyncio
async def test_setup_success(
    mock_plugin_manager_for_dist_strat: AsyncMock,
    mock_task_queue_plugin: AsyncMock,
):
    strategy = DistributedTaskInvocationStrategy()
    await strategy.setup(
        config={
            "plugin_manager": mock_plugin_manager_for_dist_strat,
            "task_queue_plugin_id": "dummy_queue_id_for_strategy_setup",
        }
    )
    assert strategy._task_queue_plugin is mock_task_queue_plugin
    mock_plugin_manager_for_dist_strat.get_plugin_instance.assert_awaited_once_with(
        "dummy_queue_id_for_strategy_setup",
        config={"plugin_manager": mock_plugin_manager_for_dist_strat},
    )


@pytest.mark.asyncio
async def test_setup_no_plugin_manager(caplog: pytest.LogCaptureFixture):
    strategy = DistributedTaskInvocationStrategy()
    caplog.set_level(logging.ERROR, logger=STRATEGY_LOGGER_NAME)
    await strategy.setup(config={})
    assert strategy._task_queue_plugin is None
    assert "PluginManager not provided in config" in caplog.text


@pytest.mark.asyncio
async def test_setup_no_task_queue_id(caplog: pytest.LogCaptureFixture):
    strategy = DistributedTaskInvocationStrategy()
    caplog.set_level(logging.ERROR, logger=STRATEGY_LOGGER_NAME)
    await strategy.setup(config={"plugin_manager": AsyncMock(spec=PluginManager)})
    assert strategy._task_queue_plugin is None
    assert "'task_queue_plugin_id' not provided" in caplog.text


@pytest.mark.asyncio
async def test_setup_plugin_load_fails(
    mock_plugin_manager_for_dist_strat: AsyncMock, caplog: pytest.LogCaptureFixture
):
    strategy = DistributedTaskInvocationStrategy()
    caplog.set_level(logging.ERROR, logger=STRATEGY_LOGGER_NAME)
    mock_plugin_manager_for_dist_strat.get_plugin_instance.return_value = None
    await strategy.setup(
        config={
            "plugin_manager": mock_plugin_manager_for_dist_strat,
            "task_queue_plugin_id": "failing_queue",
        }
    )
    assert strategy._task_queue_plugin is None
    assert "Task queue plugin 'failing_queue' not found or invalid" in caplog.text


@pytest.mark.asyncio
async def test_invoke_success_with_polling(
    mock_plugin_manager_for_dist_strat: AsyncMock,
    mock_task_queue_plugin: AsyncMock,
    mock_tool_for_dist_strat: MagicMock,
    mock_key_provider_for_dist_strat: MagicMock,
):
    strategy = DistributedTaskInvocationStrategy()
    await strategy.setup(
        config={
            "plugin_manager": mock_plugin_manager_for_dist_strat,
            "task_queue_plugin_id": "mock_task_queue_v1",
        }
    )
    # Simulate polling: pending -> running -> success
    mock_task_queue_plugin.get_task_status.side_effect = [
        "pending",
        "running",
        "success",
    ]

    result = await strategy.invoke(
        tool=mock_tool_for_dist_strat,
        params={"p": 1},
        key_provider=mock_key_provider_for_dist_strat,
        context={},
        invoker_config={},
    )

    assert result == {"result": "task completed successfully"}
    mock_task_queue_plugin.submit_task.assert_awaited_once_with(
        task_name=GENERIC_TOOL_EXECUTION_TASK_NAME,
        kwargs={
            "tool_id": "distributed_test_tool",
            "tool_params": {"p": 1},
            "context_info": {},
        },
    )
    assert mock_task_queue_plugin.get_task_status.call_count == 3
    mock_task_queue_plugin.get_task_result.assert_awaited_once_with("mock_task_123")


@pytest.mark.asyncio
async def test_invoke_task_fails(
    mock_plugin_manager_for_dist_strat: AsyncMock,
    mock_task_queue_plugin: AsyncMock,
    mock_tool_for_dist_strat: MagicMock,
    mock_key_provider_for_dist_strat: MagicMock,
):
    strategy = DistributedTaskInvocationStrategy()
    await strategy.setup(
        config={
            "plugin_manager": mock_plugin_manager_for_dist_strat,
            "task_queue_plugin_id": "mock_task_queue_v1",
        }
    )
    mock_task_queue_plugin.get_task_status.return_value = "failure"
    mock_task_queue_plugin.get_task_result.return_value = "Traceback: Error in worker"

    result = await strategy.invoke(
        tool=mock_tool_for_dist_strat,
        params={},
        key_provider=mock_key_provider_for_dist_strat,
        context=None,
        invoker_config={},
    )

    assert result["error"] == "Task execution failed: Traceback: Error in worker"
    assert result["task_id"] == "mock_task_123"


@pytest.mark.asyncio
async def test_invoke_timeout(
    mock_plugin_manager_for_dist_strat: AsyncMock,
    mock_task_queue_plugin: AsyncMock,
    mock_tool_for_dist_strat: MagicMock,
    mock_key_provider_for_dist_strat: MagicMock,
):
    strategy = DistributedTaskInvocationStrategy()
    await strategy.setup(
        config={
            "plugin_manager": mock_plugin_manager_for_dist_strat,
            "task_queue_plugin_id": "mock_task_queue_v1",
        }
    )
    mock_task_queue_plugin.get_task_status.return_value = "running"  # Always running

    result = await strategy.invoke(
        tool=mock_tool_for_dist_strat,
        params={},
        key_provider=mock_key_provider_for_dist_strat,
        context=None,
        invoker_config={
            "distributed_task_timeout_seconds": 0.02,
            "distributed_task_poll_interval_seconds": 0.01,
        },
    )

    assert result["error"] == "Task polling timed out."
    assert result["task_id"] == "mock_task_123"


@pytest.mark.asyncio
async def test_invoke_no_queue_plugin(
    mock_plugin_manager_for_dist_strat: AsyncMock,
    mock_tool_for_dist_strat: MagicMock,
    mock_key_provider_for_dist_strat: MagicMock,
):
    strategy = DistributedTaskInvocationStrategy()
    # Setup fails to load the plugin
    mock_plugin_manager_for_dist_strat.get_plugin_instance.return_value = None
    await strategy.setup(
        config={
            "plugin_manager": mock_plugin_manager_for_dist_strat,
            "task_queue_plugin_id": "failing_queue",
        }
    )

    result = await strategy.invoke(
        tool=mock_tool_for_dist_strat,
        params={},
        key_provider=mock_key_provider_for_dist_strat,
        context=None,
        invoker_config={},
    )
    assert result["error"] == "Task queue system unavailable for tool execution."


@pytest.mark.asyncio
async def test_invoke_submit_fails(
    mock_plugin_manager_for_dist_strat: AsyncMock,
    mock_task_queue_plugin: AsyncMock,
    mock_tool_for_dist_strat: MagicMock,
    mock_key_provider_for_dist_strat: MagicMock,
):
    strategy = DistributedTaskInvocationStrategy()
    await strategy.setup(
        config={
            "plugin_manager": mock_plugin_manager_for_dist_strat,
            "task_queue_plugin_id": "mock_task_queue_v1",
        }
    )
    mock_task_queue_plugin.submit_task.side_effect = RuntimeError("Broker down")

    result = await strategy.invoke(
        tool=mock_tool_for_dist_strat,
        params={},
        key_provider=mock_key_provider_for_dist_strat,
        context=None,
        invoker_config={},
    )
    assert result["error"] == "Failed to invoke tool via task queue: Broker down"


@pytest.mark.asyncio
async def test_teardown(mock_plugin_manager_for_dist_strat: AsyncMock):
    strategy = DistributedTaskInvocationStrategy()
    await strategy.setup(
        config={
            "plugin_manager": mock_plugin_manager_for_dist_strat,
            "task_queue_plugin_id": "mock_task_queue_v1",
        }
    )
    assert strategy._task_queue_plugin is not None
    await strategy.teardown()
    assert strategy._task_queue_plugin is None