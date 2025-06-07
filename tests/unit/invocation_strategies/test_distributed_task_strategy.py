import logging
from unittest.mock import AsyncMock

import pytest
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.invocation_strategies.impl.distributed_task_strategy import (
    DistributedTaskInvocationStrategy,
)
from genie_tooling.task_queues.abc import DistributedTaskQueuePlugin

STRATEGY_LOGGER_NAME = "genie_tooling.invocation_strategies.impl.distributed_task_strategy"

@pytest.fixture
def mock_task_queue_plugin() -> AsyncMock:
    plugin = AsyncMock(spec=DistributedTaskQueuePlugin)
    plugin.plugin_id = "mock_task_queue_v1"
    return plugin

@pytest.fixture
def mock_plugin_manager_for_dist_strat(mock_task_queue_plugin: AsyncMock) -> AsyncMock:
    pm = AsyncMock(spec=PluginManager)
    pm.get_plugin_instance = AsyncMock(return_value=mock_task_queue_plugin)
    return pm

@pytest.mark.asyncio
async def test_setup_success(
    mock_plugin_manager_for_dist_strat: AsyncMock,
    mock_task_queue_plugin: AsyncMock
):
    strategy = DistributedTaskInvocationStrategy()
    await strategy.setup(config={
        "plugin_manager": mock_plugin_manager_for_dist_strat,
        "task_queue_plugin_id": "dummy_queue_id_for_strategy_setup"
    })
    assert strategy._task_queue_plugin is mock_task_queue_plugin
    mock_plugin_manager_for_dist_strat.get_plugin_instance.assert_awaited_once_with(
        "dummy_queue_id_for_strategy_setup",
        config={"plugin_manager": mock_plugin_manager_for_dist_strat}
    )
