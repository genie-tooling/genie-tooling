### tests/unit/invocation_strategies/test_distributed_task_strategy.py
import asyncio
import logging
from typing import AsyncGenerator # Added for correct fixture type hint
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
    plugin.submit_task = AsyncMock(return_value="test_task_id_from_mock_queue")
    plugin.get_task_status = AsyncMock(return_value="success")
    plugin.get_task_result = AsyncMock(return_value={"tool_output": "result from worker"})
    plugin.revoke_task = AsyncMock(return_value=True)
    plugin.teardown = AsyncMock()
    return plugin

@pytest.fixture
def mock_plugin_manager_for_dist_strat(mock_task_queue_plugin: AsyncMock) -> AsyncMock:
    pm = AsyncMock(spec=PluginManager)
    async def get_instance_side_effect(plugin_id_req, config=None):
        if plugin_id_req == "dummy_queue_id_for_strategy_setup":
            return mock_task_queue_plugin
        return None
    pm.get_plugin_instance.side_effect = get_instance_side_effect
    return pm

@pytest.fixture
async def distributed_strategy(
    mock_plugin_manager_for_dist_strat: AsyncMock,
    mock_task_queue_plugin: AsyncMock
) -> AsyncGenerator[DistributedTaskInvocationStrategy, None]:
    strategy = DistributedTaskInvocationStrategy()
    await strategy.setup(config={
        "plugin_manager": mock_plugin_manager_for_dist_strat,
        "task_queue_plugin_id": "dummy_queue_id_for_strategy_setup"
    })

    # Correctly reset mocks and re-assign return_values
    mock_task_queue_plugin.submit_task.reset_mock()
    mock_task_queue_plugin.submit_task.return_value = "test_task_id_from_mock_queue"

    mock_task_queue_plugin.get_task_status.reset_mock()
    mock_task_queue_plugin.get_task_status.return_value = "success" # Default, can be overridden by side_effect

    mock_task_queue_plugin.get_task_result.reset_mock()
    mock_task_queue_plugin.get_task_result.return_value = {"tool_output": "result from worker"}

    yield strategy

@pytest.fixture
def mock_tool_for_dist_strat() -> AsyncMock:
    tool = AsyncMock(spec=Tool)
    tool.identifier = "test_tool_for_dist_strat"
    return tool

@pytest.fixture
def mock_key_provider_for_dist_strat() -> AsyncMock:
    return AsyncMock(spec=KeyProvider)


@pytest.mark.asyncio
class TestDistributedTaskStrategySetup:
    async def test_setup_success(
        self,
        mock_plugin_manager_for_dist_strat: AsyncMock,
        mock_task_queue_plugin: AsyncMock
    ):
        strategy = DistributedTaskInvocationStrategy()
        await strategy.setup(config={
            "plugin_manager": mock_plugin_manager_for_dist_strat,
            "task_queue_plugin_id": "dummy_queue_id_for_strategy_setup"
        })
        assert strategy._task_queue_plugin is mock_task_queue_plugin
        mock_plugin_manager_for_dist_strat.get_plugin_instance.assert_awaited_with("dummy_queue_id_for_strategy_setup")

    async def test_setup_no_plugin_manager(self, caplog: pytest.LogCaptureFixture):
        caplog.set_level(logging.ERROR, logger=STRATEGY_LOGGER_NAME)
        strategy = DistributedTaskInvocationStrategy()
        await strategy.setup(config={"task_queue_plugin_id": "any_id"})
        assert strategy._task_queue_plugin is None
        assert "PluginManager not provided in config" in caplog.text

    async def test_setup_no_task_queue_id(
        self,
        mock_plugin_manager_for_dist_strat: AsyncMock,
        caplog: pytest.LogCaptureFixture
    ):
        caplog.set_level(logging.ERROR, logger=STRATEGY_LOGGER_NAME)
        strategy = DistributedTaskInvocationStrategy()
        await strategy.setup(config={"plugin_manager": mock_plugin_manager_for_dist_strat})
        assert strategy._task_queue_plugin is None
        assert "'task_queue_plugin_id' not provided" in caplog.text

    async def test_setup_task_queue_plugin_not_found(
        self,
        mock_plugin_manager_for_dist_strat: AsyncMock,
        caplog: pytest.LogCaptureFixture
    ):
        caplog.set_level(logging.ERROR, logger=STRATEGY_LOGGER_NAME)
        mock_plugin_manager_for_dist_strat.get_plugin_instance.return_value = None
        strategy = DistributedTaskInvocationStrategy()
        await strategy.setup(config={
            "plugin_manager": mock_plugin_manager_for_dist_strat,
            "task_queue_plugin_id": "non_existent_queue"
        })
        assert strategy._task_queue_plugin is None
        assert "Task queue plugin 'non_existent_queue' not found or invalid" in caplog.text

@pytest.mark.asyncio
class TestDistributedTaskStrategyInvoke:
    async def test_invoke_success_polling(
        self,
        distributed_strategy: AsyncGenerator[DistributedTaskInvocationStrategy, None],
        mock_task_queue_plugin: AsyncMock,
        mock_tool_for_dist_strat: AsyncMock,
        mock_key_provider_for_dist_strat: AsyncMock
    ):
        strategy = await anext(distributed_strategy)
        mock_task_queue_plugin.get_task_status.side_effect = ["pending", "running", "success"]

        result = await strategy.invoke(
            tool=mock_tool_for_dist_strat,
            params={"p": 1},
            key_provider=mock_key_provider_for_dist_strat,
            context={"user": "test"},
            invoker_config={"distributed_task_poll_interval_seconds": 0.01}
        )
        assert result == {"tool_output": "result from worker"}
        mock_task_queue_plugin.submit_task.assert_awaited_once_with(
            task_name=GENERIC_TOOL_EXECUTION_TASK_NAME,
            kwargs={
                "tool_id": "test_tool_for_dist_strat",
                "tool_params": {"p": 1},
                "context_info": {"user": "test"}
            }
        )
        assert mock_task_queue_plugin.get_task_status.call_count == 3
        mock_task_queue_plugin.get_task_result.assert_awaited_once_with("test_task_id_from_mock_queue")

    async def test_invoke_task_queue_not_available(
        self,
        mock_tool_for_dist_strat: AsyncMock,
        mock_key_provider_for_dist_strat: AsyncMock,
        caplog: pytest.LogCaptureFixture
    ):
        caplog.set_level(logging.ERROR, logger=STRATEGY_LOGGER_NAME)
        strategy_no_queue = DistributedTaskInvocationStrategy() # No setup call
        result = await strategy_no_queue.invoke(
            tool=mock_tool_for_dist_strat, params={}, key_provider=mock_key_provider_for_dist_strat, context=None, invoker_config={}
        )
        assert result == {"error": "Task queue system unavailable for tool execution."}
        assert "Task queue plugin not available. Cannot submit task." in caplog.text

    async def test_invoke_submit_task_fails(
        self,
        distributed_strategy: AsyncGenerator[DistributedTaskInvocationStrategy, None],
        mock_task_queue_plugin: AsyncMock,
        mock_tool_for_dist_strat: AsyncMock,
        mock_key_provider_for_dist_strat: AsyncMock,
        caplog: pytest.LogCaptureFixture
    ):
        caplog.set_level(logging.ERROR, logger=STRATEGY_LOGGER_NAME)
        strategy = await anext(distributed_strategy)
        mock_task_queue_plugin.submit_task.side_effect = RuntimeError("Celery broker down")

        result = await strategy.invoke(
            tool=mock_tool_for_dist_strat, params={}, key_provider=mock_key_provider_for_dist_strat, context=None, invoker_config={}
        )
        assert result == {"error": "Failed to invoke tool via task queue: Celery broker down"}
        assert "Error during distributed task invocation" in caplog.text

    async def test_invoke_task_fails_on_worker(
        self,
        distributed_strategy: AsyncGenerator[DistributedTaskInvocationStrategy, None],
        mock_task_queue_plugin: AsyncMock,
        mock_tool_for_dist_strat: AsyncMock,
        mock_key_provider_for_dist_strat: AsyncMock,
        caplog: pytest.LogCaptureFixture
    ):
        caplog.set_level(logging.ERROR, logger=STRATEGY_LOGGER_NAME)
        strategy = await anext(distributed_strategy)
        mock_task_queue_plugin.get_task_status.return_value = "failure"
        mock_task_queue_plugin.get_task_result.return_value = "Worker error details"

        result = await strategy.invoke(
            tool=mock_tool_for_dist_strat, params={}, key_provider=mock_key_provider_for_dist_strat, context=None, invoker_config={}
        )
        assert result == {"error": "Task execution failed: Worker error details", "task_id": "test_task_id_from_mock_queue"}
        assert "Task execution failed: Worker error details (Task ID: test_task_id_from_mock_queue)" in caplog.text

    async def test_invoke_task_fails_result_retrieval_fails(
        self,
        distributed_strategy: AsyncGenerator[DistributedTaskInvocationStrategy, None],
        mock_task_queue_plugin: AsyncMock,
        mock_tool_for_dist_strat: AsyncMock,
        mock_key_provider_for_dist_strat: AsyncMock,
        caplog: pytest.LogCaptureFixture
    ):
        caplog.set_level(logging.ERROR, logger=STRATEGY_LOGGER_NAME)
        strategy = await anext(distributed_strategy)
        mock_task_queue_plugin.get_task_status.return_value = "failure"
        mock_task_queue_plugin.get_task_result.side_effect = Exception("Cannot get result of failed task")

        result = await strategy.invoke(
            tool=mock_tool_for_dist_strat, params={}, key_provider=mock_key_provider_for_dist_strat, context=None, invoker_config={}
        )
        assert result["error"] == "Task execution failed, and result retrieval also failed: Cannot get result of failed task"
        assert result["task_id"] == "test_task_id_from_mock_queue"

    async def test_invoke_task_revoked_or_unknown(
        self,
        distributed_strategy: AsyncGenerator[DistributedTaskInvocationStrategy, None],
        mock_task_queue_plugin: AsyncMock,
        mock_tool_for_dist_strat: AsyncMock,
        mock_key_provider_for_dist_strat: AsyncMock,
        caplog: pytest.LogCaptureFixture
    ):
        caplog.set_level(logging.ERROR, logger=STRATEGY_LOGGER_NAME)
        strategy = await anext(distributed_strategy)
        mock_task_queue_plugin.get_task_status.return_value = "revoked"

        result = await strategy.invoke(
            tool=mock_tool_for_dist_strat, params={}, key_provider=mock_key_provider_for_dist_strat, context=None, invoker_config={}
        )
        assert result == {"error": "Task 'test_task_id_from_mock_queue' was revoked or status is unknown.", "task_id": "test_task_id_from_mock_queue"}
        assert "Task 'test_task_id_from_mock_queue' was revoked or status is unknown." in caplog.text

    async def test_invoke_polling_timeout(
        self,
        distributed_strategy: AsyncGenerator[DistributedTaskInvocationStrategy, None],
        mock_task_queue_plugin: AsyncMock,
        mock_tool_for_dist_strat: AsyncMock,
        mock_key_provider_for_dist_strat: AsyncMock,
        caplog: pytest.LogCaptureFixture
    ):
        caplog.set_level(logging.WARNING, logger=STRATEGY_LOGGER_NAME)
        strategy = await anext(distributed_strategy)
        mock_task_queue_plugin.get_task_status.return_value = "pending" # Always pending

        result = await strategy.invoke(
            tool=mock_tool_for_dist_strat, params={}, key_provider=mock_key_provider_for_dist_strat, context=None,
            invoker_config={"distributed_task_timeout_seconds": 0.02, "distributed_task_poll_interval_seconds": 0.01}
        )
        assert result == {"error": "Task polling timed out.", "task_id": "test_task_id_from_mock_queue"}
        assert "Polling timeout for task 'test_task_id_from_mock_queue'" in caplog.text

    async def test_teardown(self, distributed_strategy: AsyncGenerator[DistributedTaskInvocationStrategy, None]):
        strategy = await anext(distributed_strategy)
        assert strategy._task_queue_plugin is not None
        await strategy.teardown()
        assert strategy._task_queue_plugin is None