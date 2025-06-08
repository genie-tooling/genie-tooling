### tests/unit/task_queues/test_task_queue_manager.py
import logging
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import Plugin
from genie_tooling.observability.manager import InteractionTracingManager
from genie_tooling.task_queues.abc import (
    DistributedTaskQueuePlugin,
    TaskStatus,
)
from genie_tooling.task_queues.manager import DistributedTaskQueueManager

MANAGER_LOGGER_NAME = "genie_tooling.task_queues.manager"


# --- Mocks ---
class MockTaskQueuePlugin(DistributedTaskQueuePlugin):
    plugin_id: str = "mock_task_queue_v1"
    description: str = "Mock Task Queue Plugin"
    teardown_called: bool = False

    # Flags to control behavior for testing
    submit_task_should_raise: bool
    get_status_should_raise: bool
    get_result_should_raise: bool
    revoke_task_should_raise: bool
    teardown_should_raise: bool

    def __init__(self):
        # Initialize methods as AsyncMocks directly
        self.submit_task = AsyncMock(return_value="mock_task_id_123")
        self.get_task_status = AsyncMock(return_value="success")
        self.get_task_result = AsyncMock(return_value={"data": "task complete"})
        self.revoke_task = AsyncMock(return_value=True)
        self.teardown = AsyncMock() # Make teardown an AsyncMock for assertions

        # Initialize flags
        self.teardown_called = False
        self.submit_task_should_raise = False
        self.get_status_should_raise = False
        self.get_result_should_raise = False
        self.revoke_task_should_raise = False
        self.teardown_should_raise = False


    async def setup(self, config=None):
        self.teardown_called = False # Reset on setup
        # Configure side_effects based on flags
        if self.submit_task_should_raise:
            self.submit_task.side_effect = RuntimeError("Submit task failed")
        else:
            self.submit_task.side_effect = None
            self.submit_task.return_value = "mock_task_id_123"

        if self.get_status_should_raise:
            self.get_task_status.side_effect = RuntimeError("Get status failed")
        else:
            self.get_task_status.side_effect = None
            self.get_task_status.return_value = "success"

        if self.get_result_should_raise:
            self.get_task_result.side_effect = RuntimeError("Get result failed")
        else:
            self.get_task_result.side_effect = None
            self.get_task_result.return_value = {"data": "task complete"}

        if self.revoke_task_should_raise:
            self.revoke_task.side_effect = RuntimeError("Revoke failed")
        else:
            self.revoke_task.side_effect = None
            self.revoke_task.return_value = True

        if self.teardown_should_raise:
            self.teardown.side_effect = RuntimeError("Teardown failed")
        else:
            self.teardown.side_effect = None


class NotATaskQueuePlugin(Plugin):
    plugin_id: str = "not_a_task_queue_v1"
    description: str = "Not a task queue plugin"
    async def setup(self, config=None): pass
    async def teardown(self): pass


@pytest.fixture
def mock_plugin_manager_for_tq_mgr() -> MagicMock:
    pm = MagicMock(spec=PluginManager)
    pm.get_plugin_instance = AsyncMock()
    return pm

@pytest.fixture
def mock_tracing_manager_for_tq_mgr() -> MagicMock:
    tm = MagicMock(spec=InteractionTracingManager)
    tm.trace_event = AsyncMock()
    return tm

@pytest.fixture
def task_queue_manager(
    mock_plugin_manager_for_tq_mgr: MagicMock,
    mock_tracing_manager_for_tq_mgr: MagicMock,
) -> DistributedTaskQueueManager:
    return DistributedTaskQueueManager(
        plugin_manager=mock_plugin_manager_for_tq_mgr,
        default_queue_id="default_queue",
        queue_configurations={"default_queue": {"some_config": "value"}},
        tracing_manager=mock_tracing_manager_for_tq_mgr,
    )


@pytest.mark.asyncio
class TestDistributedTaskQueueManagerGetPlugin:
    async def test_get_queue_plugin_success_new_instance(
        self, task_queue_manager: DistributedTaskQueueManager, mock_plugin_manager_for_tq_mgr: MagicMock
    ):
        mock_queue_instance = MockTaskQueuePlugin()
        await mock_queue_instance.setup() # Setup the instance before PM returns it
        mock_plugin_manager_for_tq_mgr.get_plugin_instance.return_value = mock_queue_instance

        plugin = await task_queue_manager._get_queue_plugin("default_queue")

        assert plugin is mock_queue_instance
        mock_plugin_manager_for_tq_mgr.get_plugin_instance.assert_awaited_once_with(
            "default_queue", config={"some_config": "value"}
        )
        assert "default_queue" in task_queue_manager._active_queues

    async def test_get_queue_plugin_cached_instance(
        self, task_queue_manager: DistributedTaskQueueManager, mock_plugin_manager_for_tq_mgr: MagicMock
    ):
        mock_queue_instance = MockTaskQueuePlugin()
        await mock_queue_instance.setup()
        task_queue_manager._active_queues["cached_queue"] = mock_queue_instance

        plugin = await task_queue_manager._get_queue_plugin("cached_queue")

        assert plugin is mock_queue_instance
        mock_plugin_manager_for_tq_mgr.get_plugin_instance.assert_not_called()

    async def test_get_queue_plugin_no_default_id(
        self, mock_plugin_manager_for_tq_mgr: MagicMock, mock_tracing_manager_for_tq_mgr: MagicMock
    ):
        manager_no_default = DistributedTaskQueueManager(
            plugin_manager=mock_plugin_manager_for_tq_mgr,
            tracing_manager=mock_tracing_manager_for_tq_mgr
        )
        plugin = await manager_no_default._get_queue_plugin() # Request default
        assert plugin is None
        mock_tracing_manager_for_tq_mgr.trace_event.assert_awaited_with(
            event_name="log.error",
            data={"message": "No task queue ID specified and no default is set."},
            component="DistributedTaskQueueManager",
            correlation_id=None
        )

    async def test_get_queue_plugin_not_found(
        self, task_queue_manager: DistributedTaskQueueManager, mock_plugin_manager_for_tq_mgr: MagicMock, mock_tracing_manager_for_tq_mgr: MagicMock
    ):
        mock_plugin_manager_for_tq_mgr.get_plugin_instance.return_value = None
        plugin = await task_queue_manager._get_queue_plugin("non_existent_queue")
        assert plugin is None
        mock_tracing_manager_for_tq_mgr.trace_event.assert_awaited_with(
            event_name="log.warning",
            data={"message": "DistributedTaskQueuePlugin 'non_existent_queue' not found or failed to load."},
            component="DistributedTaskQueueManager",
            correlation_id=None
        )

    async def test_get_queue_plugin_wrong_type(
        self, task_queue_manager: DistributedTaskQueueManager, mock_plugin_manager_for_tq_mgr: MagicMock, mock_tracing_manager_for_tq_mgr: MagicMock
    ):
        mock_plugin_manager_for_tq_mgr.get_plugin_instance.return_value = NotATaskQueuePlugin()
        plugin = await task_queue_manager._get_queue_plugin("wrong_type_queue")
        assert plugin is None
        mock_tracing_manager_for_tq_mgr.trace_event.assert_awaited_with(
            event_name="log.warning",
            data={"message": "Plugin 'wrong_type_queue' loaded but is not a valid DistributedTaskQueuePlugin."},
            component="DistributedTaskQueueManager",
            correlation_id=None
        )

    async def test_get_queue_plugin_load_error(
        self, task_queue_manager: DistributedTaskQueueManager, mock_plugin_manager_for_tq_mgr: MagicMock, mock_tracing_manager_for_tq_mgr: MagicMock
    ):
        mock_plugin_manager_for_tq_mgr.get_plugin_instance.side_effect = RuntimeError("Load failed")
        plugin = await task_queue_manager._get_queue_plugin("error_queue")
        assert plugin is None
        mock_tracing_manager_for_tq_mgr.trace_event.assert_awaited_with(
            event_name="log.error",
            data={"message": "Error loading DistributedTaskQueuePlugin 'error_queue': Load failed", "exc_info": True},
            component="DistributedTaskQueueManager",
            correlation_id=None
        )


@pytest.mark.asyncio
class TestDistributedTaskQueueManagerOperations:
    async def test_submit_task_success(
        self, task_queue_manager: DistributedTaskQueueManager, mock_plugin_manager_for_tq_mgr: MagicMock
    ):
        mock_queue_instance = MockTaskQueuePlugin()
        await mock_queue_instance.setup() # Ensure methods are AsyncMocks
        mock_plugin_manager_for_tq_mgr.get_plugin_instance.return_value = mock_queue_instance

        task_id = await task_queue_manager.submit_task("my_task", args=(1,), kwargs={"a": 2}, queue_id="default_queue")
        assert task_id == "mock_task_id_123"
        # Assert on the AsyncMock attribute of the instance
        mock_queue_instance.submit_task.assert_awaited_once_with("my_task", (1,), {"a": 2}, "default_queue", None)

    async def test_submit_task_plugin_fails(
        self, task_queue_manager: DistributedTaskQueueManager, mock_plugin_manager_for_tq_mgr: MagicMock, mock_tracing_manager_for_tq_mgr: MagicMock
    ):
        mock_queue_instance = MockTaskQueuePlugin()
        mock_queue_instance.submit_task_should_raise = True
        await mock_queue_instance.setup() # This configures the side_effect
        mock_plugin_manager_for_tq_mgr.get_plugin_instance.return_value = mock_queue_instance

        task_id = await task_queue_manager.submit_task("failing_task")
        assert task_id is None
        mock_tracing_manager_for_tq_mgr.trace_event.assert_awaited_with(
            event_name="log.error",
            data={"message": "Error submitting task 'failing_task' via plugin 'mock_task_queue_v1': Submit task failed", "exc_info": True},
            component="DistributedTaskQueueManager",
            correlation_id=None
        )

    async def test_get_task_status_success(
        self, task_queue_manager: DistributedTaskQueueManager, mock_plugin_manager_for_tq_mgr: MagicMock
    ):
        mock_queue_instance = MockTaskQueuePlugin()
        await mock_queue_instance.setup()
        mock_plugin_manager_for_tq_mgr.get_plugin_instance.return_value = mock_queue_instance

        status = await task_queue_manager.get_task_status("task_abc")
        assert status == "success"
        mock_queue_instance.get_task_status.assert_awaited_once_with("task_abc", None)

    async def test_get_task_result_success(
        self, task_queue_manager: DistributedTaskQueueManager, mock_plugin_manager_for_tq_mgr: MagicMock
    ):
        mock_queue_instance = MockTaskQueuePlugin()
        await mock_queue_instance.setup()
        mock_plugin_manager_for_tq_mgr.get_plugin_instance.return_value = mock_queue_instance

        result = await task_queue_manager.get_task_result("task_xyz", timeout_seconds=5.0)
        assert result == {"data": "task complete"}
        mock_queue_instance.get_task_result.assert_awaited_once_with("task_xyz", None, 5.0)

    async def test_get_task_result_plugin_fails(
        self, task_queue_manager: DistributedTaskQueueManager, mock_plugin_manager_for_tq_mgr: MagicMock
    ):
        mock_queue_instance = MockTaskQueuePlugin()
        mock_queue_instance.get_result_should_raise = True
        await mock_queue_instance.setup()
        mock_plugin_manager_for_tq_mgr.get_plugin_instance.return_value = mock_queue_instance

        with pytest.raises(RuntimeError, match="Get result failed"):
            await task_queue_manager.get_task_result("task_res_fail")

    async def test_revoke_task_success(
        self, task_queue_manager: DistributedTaskQueueManager, mock_plugin_manager_for_tq_mgr: MagicMock
    ):
        mock_queue_instance = MockTaskQueuePlugin()
        await mock_queue_instance.setup()
        mock_plugin_manager_for_tq_mgr.get_plugin_instance.return_value = mock_queue_instance

        revoked = await task_queue_manager.revoke_task("task_to_revoke", terminate=True)
        assert revoked is True
        mock_queue_instance.revoke_task.assert_awaited_once_with("task_to_revoke", None, True)

    async def test_operations_fail_if_no_plugin(
        self, task_queue_manager: DistributedTaskQueueManager, mock_plugin_manager_for_tq_mgr: MagicMock
    ):
        mock_plugin_manager_for_tq_mgr.get_plugin_instance.return_value = None # Simulate plugin load failure

        assert await task_queue_manager.submit_task("task") is None
        assert await task_queue_manager.get_task_status("id") == "unknown"
        with pytest.raises(RuntimeError, match="Task queue plugin not available to fetch result."):
            await task_queue_manager.get_task_result("id")
        assert await task_queue_manager.revoke_task("id") is False


@pytest.mark.asyncio
class TestDistributedTaskQueueManagerTeardown:
    async def test_teardown_calls_plugin_teardown(
        self, task_queue_manager: DistributedTaskQueueManager, mock_plugin_manager_for_tq_mgr: MagicMock
    ):
        mock_queue_instance = MockTaskQueuePlugin()
        await mock_queue_instance.setup()
        mock_plugin_manager_for_tq_mgr.get_plugin_instance.return_value = mock_queue_instance
        # Load it into active_queues
        await task_queue_manager._get_queue_plugin("default_queue")
        assert "default_queue" in task_queue_manager._active_queues

        await task_queue_manager.teardown()
        mock_queue_instance.teardown.assert_awaited_once()
        assert not task_queue_manager._active_queues

    async def test_teardown_plugin_teardown_fails(
        self, task_queue_manager: DistributedTaskQueueManager, mock_plugin_manager_for_tq_mgr: MagicMock, mock_tracing_manager_for_tq_mgr: MagicMock
    ):
        mock_queue_instance = MockTaskQueuePlugin()
        mock_queue_instance.teardown_should_raise = True
        await mock_queue_instance.setup() # This will configure the teardown mock
        mock_plugin_manager_for_tq_mgr.get_plugin_instance.return_value = mock_queue_instance
        await task_queue_manager._get_queue_plugin("default_queue")

        await task_queue_manager.teardown()
        mock_queue_instance.teardown.assert_awaited_once() # Teardown was attempted
        assert not task_queue_manager._active_queues # Still cleared
        mock_tracing_manager_for_tq_mgr.trace_event.assert_any_call(
            event_name="log.error",
            data={"message": "Error tearing down task queue plugin 'default_queue': Teardown failed", "exc_info": True},
            component="DistributedTaskQueueManager",
            correlation_id=None
        )
