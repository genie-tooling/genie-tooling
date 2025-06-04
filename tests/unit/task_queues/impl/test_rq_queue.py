### tests/unit/task_queues/impl/test_rq_queue.py
import logging
import re  # For regex matching in error messages
from typing import AsyncGenerator
from unittest.mock import MagicMock, patch

import pytest
from genie_tooling.task_queues.abc import TaskStatus
from genie_tooling.task_queues.impl.rq_queue import (
    RedisQueueTaskPlugin,
)

# Dynamically check RQ_AVAILABLE for the test module itself
try:
    from redis import Redis as ActualRedis
    from rq import Queue as ActualRQQueue
    from rq.exceptions import NoSuchJobError as ActualRQNSError
    from rq.job import Job as ActualRQJob
except ImportError:
    ActualRedis = MagicMock(name="MockActualRedis") # type: ignore
    ActualRQQueue = MagicMock(name="MockActualRQQueue") # type: ignore
    ActualRQNSError = type("MockActualRQNSError", (Exception,), {}) # type: ignore
    ActualRQJob = MagicMock(name="MockActualRQJob") # type: ignore


PLUGIN_LOGGER_NAME = "genie_tooling.task_queues.impl.rq_queue"


@pytest.fixture
def mock_redis_conn_instance() -> MagicMock:
    conn = MagicMock(spec=ActualRedis)
    conn.ping = MagicMock(return_value=True)
    conn.close = MagicMock()
    conn.delete = MagicMock(return_value=1)
    return conn


@pytest.fixture
def mock_rq_queue_instance() -> MagicMock:
    queue = MagicMock(spec=ActualRQQueue)
    mock_job_on_queue = MagicMock(spec=ActualRQJob)
    mock_job_on_queue.id = "mock_rq_job_id_from_enqueue"
    queue.enqueue = MagicMock(return_value=mock_job_on_queue)
    return queue


@pytest.fixture
def mock_rq_job_instance() -> MagicMock:
    job = MagicMock(spec=ActualRQJob)
    job.id = "fetched_rq_job_id_456"
    job.get_status = MagicMock(return_value="finished") # Default status
    job.is_finished = True # Default to True for simplicity in some tests
    job.is_failed = False
    job.is_canceled = False
    job.is_started = False
    job.result = "mock_job_result_data"
    job.exc_info = None
    job.cancel = MagicMock()
    job.worker_name = "mock_worker.1"
    return job


@pytest.fixture
async def rq_queue_plugin_fixt(
    mock_redis_conn_instance: MagicMock,
    mock_rq_queue_instance: MagicMock,
    mock_rq_job_instance: MagicMock,
    request,
) -> AsyncGenerator[RedisQueueTaskPlugin, None]:
    rq_available_for_this_test_run = getattr(request, "param", {}).get("rq_available", True)


    plugin_instance = RedisQueueTaskPlugin()

    PatchedRedis = MagicMock(name="PatchedRedisClassInModule")
    PatchedQueue = MagicMock(name="PatchedQueueClassInModule")
    PatchedJob = MagicMock(name="PatchedJobClassInModule")
    PatchedNoSuchJobError = type("PatchedNoSuchJobErrorForTest", (Exception,), {})
    # Mock for send_stop_job_command (the actual function from rq.command)
    PatchedSendStopJobCommand = MagicMock(name="PatchedSendStopJobCommandInFixture")


    PatchedRedis.from_url = MagicMock(return_value=mock_redis_conn_instance)
    PatchedQueue.return_value = mock_rq_queue_instance
    PatchedJob.fetch = MagicMock(return_value=mock_rq_job_instance)

    plugin_instance._test_mocks = { # type: ignore
        "PatchedRedisClass": PatchedRedis,
        "PatchedQueueClass": PatchedQueue,
        "PatchedJobClass": PatchedJob,
        "PatchedNoSuchJobError": PatchedNoSuchJobError,
        "PatchedSendStopJobCommand": PatchedSendStopJobCommand,
        "mock_redis_conn_instance": mock_redis_conn_instance,
        "mock_rq_queue_instance": mock_rq_queue_instance,
        "mock_rq_job_instance": mock_rq_job_instance,
    }

    # Patch the send_stop_job_command where it's imported in the plugin module
    with patch("genie_tooling.task_queues.impl.rq_queue.Redis", PatchedRedis), \
         patch("genie_tooling.task_queues.impl.rq_queue.Queue", PatchedQueue), \
         patch("genie_tooling.task_queues.impl.rq_queue.Job", PatchedJob), \
         patch("genie_tooling.task_queues.impl.rq_queue.NoSuchJobError", PatchedNoSuchJobError), \
         patch("genie_tooling.task_queues.impl.rq_queue.send_stop_job_command", PatchedSendStopJobCommand), \
         patch("genie_tooling.task_queues.impl.rq_queue.RQ_AVAILABLE", rq_available_for_this_test_run):

        await plugin_instance.setup(config={"redis_url": "redis://dummy"})

        yield plugin_instance

    if hasattr(plugin_instance, "_redis_conn") and plugin_instance._redis_conn is not None:
        await plugin_instance.teardown()
    elif hasattr(plugin_instance, "teardown"):
        await plugin_instance.teardown()


@pytest.mark.asyncio
class TestRQQueuePluginSetup:
    async def test_setup_success(self, rq_queue_plugin_fixt: AsyncGenerator[RedisQueueTaskPlugin, None]):
        plugin = await anext(rq_queue_plugin_fixt)
        if not getattr(plugin, "_test_mocks", {}).get("mock_redis_conn_instance") or not plugin._redis_conn: # type: ignore
            pytest.fail("Test setup failed: RQ components or Redis connection not available for success test.")


        mocks = plugin._test_mocks # type: ignore
        assert plugin._redis_conn is mocks["mock_redis_conn_instance"]
        assert plugin._default_queue_name in plugin._queues
        assert plugin._queues[plugin._default_queue_name] is mocks["mock_rq_queue_instance"]

        assert mocks["PatchedRedisClass"] is not None
        mocks["PatchedRedisClass"].from_url.assert_called_once_with("redis://dummy")
        mocks["mock_redis_conn_instance"].ping.assert_called_once()

    @pytest.mark.parametrize("rq_queue_plugin_fixt", [{"rq_available": False}], indirect=True)
    async def test_setup_rq_not_available(self, rq_queue_plugin_fixt: AsyncGenerator[RedisQueueTaskPlugin, None], caplog: pytest.LogCaptureFixture):
        caplog.set_level(logging.ERROR, logger=PLUGIN_LOGGER_NAME)
        plugin = await anext(rq_queue_plugin_fixt)
        assert plugin._redis_conn is None
        assert not plugin._queues
        assert f"{plugin.plugin_id}: RQ or Redis library not available. Cannot initialize." in caplog.text

    async def test_setup_redis_connection_fails(self, mock_redis_conn_instance: MagicMock, caplog: pytest.LogCaptureFixture):
        caplog.set_level(logging.ERROR, logger=PLUGIN_LOGGER_NAME)
        mock_redis_conn_instance.ping.side_effect = Exception("Redis down")

        plugin_fails_conn = RedisQueueTaskPlugin()

        PatchedRedisForThisTest = MagicMock(name="PatchedRedisForConnFailTest")
        PatchedRedisForThisTest.from_url = MagicMock(return_value=mock_redis_conn_instance)

        with patch("genie_tooling.task_queues.impl.rq_queue.Redis", PatchedRedisForThisTest), \
             patch("genie_tooling.task_queues.impl.rq_queue.Queue", MagicMock()), \
             patch("genie_tooling.task_queues.impl.rq_queue.Job", MagicMock()), \
             patch("genie_tooling.task_queues.impl.rq_queue.NoSuchJobError", type("NSE", (Exception,), {})), \
             patch("genie_tooling.task_queues.impl.rq_queue.RQ_AVAILABLE", True):

            await plugin_fails_conn.setup(config={"redis_url": "redis://badhost"})

        PatchedRedisForThisTest.from_url.assert_called_once_with("redis://badhost")
        assert plugin_fails_conn._redis_conn is None
        assert "Failed to connect to Redis or initialize default RQ queue: Redis down" in caplog.text

    async def test_get_rq_queue_creates_new_if_not_exists(self, rq_queue_plugin_fixt: AsyncGenerator[RedisQueueTaskPlugin, None]):
        plugin = await anext(rq_queue_plugin_fixt)
        if not getattr(plugin, "_test_mocks", {}).get("mock_redis_conn_instance") or not plugin._redis_conn: # type: ignore
             pytest.fail("Test setup failed: RQ components or Redis connection not available for get_rq_queue test.")

        mocks = plugin._test_mocks # type: ignore
        PatchedQueueClass = mocks["PatchedQueueClass"]
        mock_redis_conn = mocks["mock_redis_conn_instance"]

        new_queue_name = "custom_q_for_get_test"
        if new_queue_name in plugin._queues:
            del plugin._queues[new_queue_name]
        assert new_queue_name not in plugin._queues

        new_mock_q_instance_for_custom = MagicMock(name="MockQueueInstanceForCustomQGetTest")

        assert PatchedQueueClass is not None
        PatchedQueueClass.reset_mock()
        PatchedQueueClass.return_value = new_mock_q_instance_for_custom

        q_instance = plugin._get_rq_queue(new_queue_name)

        assert q_instance is new_mock_q_instance_for_custom
        assert new_queue_name in plugin._queues
        PatchedQueueClass.assert_called_once_with(new_queue_name, connection=mock_redis_conn)
        PatchedQueueClass.return_value = mocks["mock_rq_queue_instance"]


    async def test_get_rq_queue_returns_none_if_no_connection(self, rq_queue_plugin_fixt: AsyncGenerator[RedisQueueTaskPlugin, None]):
        plugin = await anext(rq_queue_plugin_fixt)
        if not getattr(plugin, "_test_mocks", {}).get("mock_redis_conn_instance"): # type: ignore
             pytest.fail("Test setup failed: RQ components not available for no_connection test.")
        plugin._redis_conn = None
        assert plugin._get_rq_queue("any") is None


@pytest.mark.asyncio
class TestRQQueuePluginOperations:
    async def test_submit_task_success(self, rq_queue_plugin_fixt: AsyncGenerator[RedisQueueTaskPlugin, None]):
        plugin = await anext(rq_queue_plugin_fixt)
        if not getattr(plugin, "_test_mocks", {}).get("mock_redis_conn_instance") or not plugin._redis_conn: # type: ignore
            pytest.fail("Test setup failed: RQ components or Redis connection not available for submit_task test.")

        mocks = plugin._test_mocks # type: ignore
        PatchedQueueClass = mocks["PatchedQueueClass"]
        mock_redis_conn = mocks["mock_redis_conn_instance"]

        mock_special_q_instance = MagicMock(name="MockQueueInstanceForSpecialQSubmit")
        mock_special_q_job = MagicMock(name="MockJobFromSpecialQSubmit")
        mock_special_q_job.id = "mock_rq_job_id_special_q_submit"
        mock_special_q_instance.enqueue = MagicMock(return_value=mock_special_q_job)

        original_queue_constructor_return_value = PatchedQueueClass.return_value if PatchedQueueClass else None # type: ignore

        if PatchedQueueClass:
            def queue_constructor_side_effect_submit(name, connection):
                if name == "special_q_submit":
                    return mock_special_q_instance
                if name == plugin._default_queue_name:
                    return mocks["mock_rq_queue_instance"]
                raise ValueError(f"Unexpected queue name '{name}' in mock constructor for submit_task_success")
            PatchedQueueClass.side_effect = queue_constructor_side_effect_submit

        task_args_tuple = (1, 2)
        task_kwargs_dict = {"op": "add"}
        enqueue_options_dict = {"job_timeout": 60, "description": "Test Job"}

        task_id = await plugin.submit_task(
            "my_module.my_func", args=task_args_tuple, kwargs=task_kwargs_dict,
            queue_name="special_q_submit", task_options=enqueue_options_dict
        )
        assert task_id == "mock_rq_job_id_special_q_submit"

        if PatchedQueueClass:
            PatchedQueueClass.assert_any_call("special_q_submit", connection=mock_redis_conn)

        mock_special_q_instance.enqueue.assert_called_once_with(
            "my_module.my_func",
            *task_args_tuple,
            op="add",
            job_timeout=60,
            description="Test Job"
        )

        if PatchedQueueClass:
            PatchedQueueClass.side_effect = None
            PatchedQueueClass.return_value = original_queue_constructor_return_value


    async def test_submit_task_no_redis_connection(self, rq_queue_plugin_fixt: AsyncGenerator[RedisQueueTaskPlugin, None]):
        plugin = await anext(rq_queue_plugin_fixt)
        if not getattr(plugin, "_test_mocks", {}).get("mock_redis_conn_instance"): # type: ignore
             pytest.fail("Test setup failed: RQ components not available for no_redis_connection test.")
        plugin._redis_conn = None
        with pytest.raises(RuntimeError, match="Redis connection not available"):
            await plugin.submit_task("task")

    async def test_submit_task_enqueue_fails(self, rq_queue_plugin_fixt: AsyncGenerator[RedisQueueTaskPlugin, None], caplog: pytest.LogCaptureFixture):
        plugin = await anext(rq_queue_plugin_fixt)
        if not getattr(plugin, "_test_mocks", {}).get("mock_redis_conn_instance") or not plugin._redis_conn: # type: ignore
             pytest.fail("Test setup failed: RQ components or Redis connection not available for enqueue_fails test.")
        caplog.set_level(logging.ERROR, logger=PLUGIN_LOGGER_NAME)
        mocks = plugin._test_mocks # type: ignore
        mock_default_queue = mocks["mock_rq_queue_instance"]
        mock_default_queue.enqueue.side_effect = Exception("RQ enqueue error")

        with pytest.raises(Exception, match="RQ enqueue error"):
            await plugin.submit_task("task_fails")
        assert f"{plugin.plugin_id}: Error submitting task 'task_fails' to RQ: RQ enqueue error" in caplog.text

    @pytest.mark.parametrize("rq_job_status, expected_genie_status", [
        ("queued", "pending"), ("started", "running"), ("deferred", "pending"),
        ("finished", "success"), ("failed", "failure"), ("scheduled", "pending"),
        ("canceled", "revoked"), ("non_standard_rq_status", "unknown"),
    ])
    async def test_get_task_status_maps_correctly(
        self, rq_queue_plugin_fixt: AsyncGenerator[RedisQueueTaskPlugin, None], mock_rq_job_instance: MagicMock,
        rq_job_status: str, expected_genie_status: TaskStatus
    ):
        plugin = await anext(rq_queue_plugin_fixt)
        if not getattr(plugin, "_test_mocks", {}).get("mock_redis_conn_instance") or not plugin._redis_conn: # type: ignore
             pytest.fail("Test setup failed: RQ components or Redis connection not available for status_maps test.")

        mocks = plugin._test_mocks # type: ignore
        PatchedJobClass = mocks["PatchedJobClass"]
        mock_redis_conn = mocks["mock_redis_conn_instance"]

        # Configure the mock_rq_job_instance for this specific test case
        mock_rq_job_instance.get_status.return_value = rq_job_status
        if PatchedJobClass:
            PatchedJobClass.fetch.return_value = mock_rq_job_instance

        status = await plugin.get_task_status("task_id_status_test")

        assert status == expected_genie_status
        if PatchedJobClass:
            PatchedJobClass.fetch.assert_called_once_with("task_id_status_test", connection=mock_redis_conn)

        mock_rq_job_instance.get_status.assert_called_once_with(refresh=True)

        # Reset mocks for the next parameterized run
        mock_rq_job_instance.get_status.reset_mock()
        if PatchedJobClass:
            PatchedJobClass.fetch.reset_mock()


    async def test_get_task_status_no_such_job(self, rq_queue_plugin_fixt: AsyncGenerator[RedisQueueTaskPlugin, None], caplog: pytest.LogCaptureFixture):
        plugin = await anext(rq_queue_plugin_fixt)
        if not getattr(plugin, "_test_mocks", {}).get("mock_redis_conn_instance") or not plugin._redis_conn: # type: ignore
             pytest.fail("Test setup failed: RQ components or Redis connection not available for no_such_job test.")

        caplog.set_level(logging.DEBUG, logger=PLUGIN_LOGGER_NAME)
        mocks = plugin._test_mocks # type: ignore
        PatchedJobClass = mocks["PatchedJobClass"]
        PatchedNoSuchJobError = mocks["PatchedNoSuchJobError"]

        if PatchedJobClass and PatchedNoSuchJobError:
            PatchedJobClass.fetch.side_effect = PatchedNoSuchJobError("Job not found")

        status = await plugin.get_task_status("non_existent_task_id")

        assert status == "unknown"
        assert f"{plugin.plugin_id}: Task ID 'non_existent_task_id' not found in RQ." in caplog.text
        if PatchedJobClass:
            PatchedJobClass.fetch.side_effect = None

    async def test_get_task_result_success(self, rq_queue_plugin_fixt: AsyncGenerator[RedisQueueTaskPlugin, None], mock_rq_job_instance: MagicMock):
        plugin = await anext(rq_queue_plugin_fixt)
        if not getattr(plugin, "_test_mocks", {}).get("mock_redis_conn_instance") or not plugin._redis_conn: # type: ignore
             pytest.fail("Test setup failed: RQ components or Redis connection not available for get_result_success test.")

        mocks = plugin._test_mocks # type: ignore
        PatchedJobClass = mocks["PatchedJobClass"]

        mock_rq_job_instance.is_finished = True
        mock_rq_job_instance.is_failed = False
        mock_rq_job_instance.is_canceled = False
        mock_rq_job_instance.result = "final_result_data"
        if PatchedJobClass:
            PatchedJobClass.fetch.return_value = mock_rq_job_instance

        result = await plugin.get_task_result("task_id_res_test")
        assert result == "final_result_data"

    async def test_get_task_result_job_failed(self, rq_queue_plugin_fixt: AsyncGenerator[RedisQueueTaskPlugin, None], mock_rq_job_instance: MagicMock):
        plugin = await anext(rq_queue_plugin_fixt)
        if not getattr(plugin, "_test_mocks", {}).get("mock_redis_conn_instance") or not plugin._redis_conn: # type: ignore
             pytest.fail("Test setup failed: RQ components or Redis connection not available for job_failed test.")

        mocks = plugin._test_mocks # type: ignore
        PatchedJobClass = mocks["PatchedJobClass"]

        mock_rq_job_instance.is_finished = False
        mock_rq_job_instance.is_failed = True
        mock_rq_job_instance.is_canceled = False
        mock_rq_job_instance.exc_info = "Traceback: Error in worker"
        mock_rq_job_instance.id = "failed_job_id_test"
        if PatchedJobClass:
            PatchedJobClass.fetch.return_value = mock_rq_job_instance

        expected_error_message = "RQ task 'failed_job_id_test' failed. Info: Traceback: Error in worker"
        with pytest.raises(RuntimeError, match=re.escape(expected_error_message)):
            await plugin.get_task_result("task_id_fail_test")

    async def test_get_task_result_timeout(self, rq_queue_plugin_fixt: AsyncGenerator[RedisQueueTaskPlugin, None], mock_rq_job_instance: MagicMock):
        plugin = await anext(rq_queue_plugin_fixt)
        if not getattr(plugin, "_test_mocks", {}).get("mock_redis_conn_instance") or not plugin._redis_conn: # type: ignore
             pytest.fail("Test setup failed: RQ components or Redis connection not available for timeout test.")

        mocks = plugin._test_mocks # type: ignore
        PatchedJobClass = mocks["PatchedJobClass"]

        mock_rq_job_instance.is_finished = False
        mock_rq_job_instance.is_failed = False
        mock_rq_job_instance.is_canceled = False
        if PatchedJobClass:
            PatchedJobClass.fetch.return_value = mock_rq_job_instance

        with pytest.raises(TimeoutError, match="Timeout waiting for RQ task 'task_id_timeout' result."):
            await plugin.get_task_result("task_id_timeout", timeout_seconds=0.01)

    async def test_revoke_task_queued_job(self, rq_queue_plugin_fixt: AsyncGenerator[RedisQueueTaskPlugin, None], mock_rq_job_instance: MagicMock):
        plugin = await anext(rq_queue_plugin_fixt)
        if not getattr(plugin, "_test_mocks", {}).get("mock_redis_conn_instance") or not plugin._redis_conn: # type: ignore
             pytest.fail("Test setup failed: RQ components or Redis connection not available for revoke_queued test.")

        mocks = plugin._test_mocks # type: ignore
        PatchedJobClass = mocks["PatchedJobClass"]

        mock_rq_job_instance.is_started = False
        mock_rq_job_instance.is_finished = False
        mock_rq_job_instance.is_failed = False
        mock_rq_job_instance.is_canceled = False
        if PatchedJobClass:
            PatchedJobClass.fetch.return_value = mock_rq_job_instance

        revoked = await plugin.revoke_task("task_id_revoke_queued")
        assert revoked is True
        mock_rq_job_instance.cancel.assert_called_once()

    async def test_revoke_task_running_job_terminate(self, rq_queue_plugin_fixt: AsyncGenerator[RedisQueueTaskPlugin, None], mock_rq_job_instance: MagicMock):
        plugin = await anext(rq_queue_plugin_fixt)
        if not getattr(plugin, "_test_mocks", {}).get("mock_redis_conn_instance") or not plugin._redis_conn: # type: ignore
             pytest.fail("Test setup failed: RQ components or Redis connection not available for revoke_running test.")

        mocks = plugin._test_mocks # type: ignore
        PatchedJobClass = mocks["PatchedJobClass"]
        PatchedSendStopJobCommand = mocks["PatchedSendStopJobCommand"]


        mock_rq_job_instance.is_started = True
        mock_rq_job_instance.id = "running_job_for_revoke"
        if PatchedJobClass:
            PatchedJobClass.fetch.return_value = mock_rq_job_instance

        revoked = await plugin.revoke_task("task_id_revoke_running", terminate=True)
        assert revoked is True
        PatchedSendStopJobCommand.assert_called_once_with(plugin._redis_conn, "running_job_for_revoke")


    async def test_revoke_task_already_finished(self, rq_queue_plugin_fixt: AsyncGenerator[RedisQueueTaskPlugin, None], mock_rq_job_instance: MagicMock):
        plugin = await anext(rq_queue_plugin_fixt)
        if not getattr(plugin, "_test_mocks", {}).get("mock_redis_conn_instance") or not plugin._redis_conn: # type: ignore
             pytest.fail("Test setup failed: RQ components or Redis connection not available for revoke_finished test.")

        mocks = plugin._test_mocks # type: ignore
        PatchedJobClass = mocks["PatchedJobClass"]

        mock_rq_job_instance.is_finished = True
        if PatchedJobClass:
            PatchedJobClass.fetch.return_value = mock_rq_job_instance

        revoked = await plugin.revoke_task("task_id_finished")
        assert revoked is True
        mock_rq_job_instance.cancel.assert_not_called()

    async def test_revoke_task_no_such_job(self, rq_queue_plugin_fixt: AsyncGenerator[RedisQueueTaskPlugin, None], caplog: pytest.LogCaptureFixture):
        plugin = await anext(rq_queue_plugin_fixt)
        if not getattr(plugin, "_test_mocks", {}).get("mock_redis_conn_instance") or not plugin._redis_conn: # type: ignore
             pytest.fail("Test setup failed: RQ components or Redis connection not available for revoke_no_job test.")

        caplog.set_level(logging.WARNING, logger=PLUGIN_LOGGER_NAME)
        mocks = plugin._test_mocks # type: ignore
        PatchedJobClass = mocks["PatchedJobClass"]
        PatchedNoSuchJobError = mocks["PatchedNoSuchJobError"]

        if PatchedJobClass and PatchedNoSuchJobError:
            PatchedJobClass.fetch.side_effect = PatchedNoSuchJobError("Job not found for revoke")

        revoked = await plugin.revoke_task("non_existent_for_revoke")
        assert revoked is True
        assert f"{plugin.plugin_id}: Task ID 'non_existent_for_revoke' not found for revoke. Considered successful." in caplog.text
        if PatchedJobClass:
            PatchedJobClass.fetch.side_effect = None

    async def test_teardown_closes_connection(self, rq_queue_plugin_fixt: AsyncGenerator[RedisQueueTaskPlugin, None]):
        plugin = await anext(rq_queue_plugin_fixt)
        if not getattr(plugin, "_test_mocks", {}).get("mock_redis_conn_instance") or not plugin._redis_conn: # type: ignore
             pytest.fail("Test setup failed: RQ components or Redis connection not available for teardown_closes test.")

        mocks = plugin._test_mocks # type: ignore
        redis_conn_mock_to_check = mocks["mock_redis_conn_instance"]
        assert plugin._redis_conn is not None

        await plugin.teardown()
        redis_conn_mock_to_check.close.assert_called_once()
        assert plugin._redis_conn is None
        assert not plugin._queues

    async def test_teardown_no_connection(self, rq_queue_plugin_fixt: AsyncGenerator[RedisQueueTaskPlugin, None]):
        plugin = await anext(rq_queue_plugin_fixt)
        if not getattr(plugin, "_test_mocks", {}).get("mock_redis_conn_instance"): # type: ignore
             pytest.fail("Test setup failed: RQ components not available for teardown_no_conn test.")
        plugin._redis_conn = None
        await plugin.teardown()
        assert plugin._redis_conn is None

    async def test_teardown_connection_close_fails(self, rq_queue_plugin_fixt: AsyncGenerator[RedisQueueTaskPlugin, None], caplog: pytest.LogCaptureFixture):
        plugin = await anext(rq_queue_plugin_fixt)
        if not getattr(plugin, "_test_mocks", {}).get("mock_redis_conn_instance") or not plugin._redis_conn: # type: ignore
             pytest.fail("Test setup failed: RQ components or Redis connection not available for teardown_close_fails test.")

        caplog.set_level(logging.ERROR, logger=PLUGIN_LOGGER_NAME)
        mocks = plugin._test_mocks # type: ignore
        mocks["mock_redis_conn_instance"].close.side_effect = Exception("Failed to close Redis")

        await plugin.teardown()
        assert plugin._redis_conn is None
        assert f"{plugin.plugin_id}: Error closing Redis connection for RQ: Failed to close Redis" in caplog.text

    @pytest.mark.parametrize("rq_queue_plugin_fixt", [{"rq_available": False}], indirect=True)
    async def test_operations_fail_if_rq_unavailable(self, rq_queue_plugin_fixt: AsyncGenerator[RedisQueueTaskPlugin, None], caplog: pytest.LogCaptureFixture):
        plugin = await anext(rq_queue_plugin_fixt)
        caplog.set_level(logging.DEBUG) # Ensure DEBUG logs are captured

        # This error comes from setup() when rq_available is False
        expected_setup_error_log = f"{plugin.plugin_id}: RQ or Redis library not available. Cannot initialize."

        with pytest.raises(RuntimeError, match="Redis connection not available"):
            await plugin.submit_task("task")
        # Check that the setup error was logged (it should be in caplog.text from the fixture's setup call)
        assert expected_setup_error_log in caplog.text
        # The submit_task itself won't log an additional "not available" if _redis_conn is None, it raises.

        # For get_task_status, it logs a DEBUG message if components are not available
        expected_get_status_debug_log = f"{plugin.plugin_id}: Cannot get task status, Redis/RQ components not available."
        assert await plugin.get_task_status("id") == "unknown"
        assert expected_get_status_debug_log in caplog.text

        # For get_task_result, it raises RuntimeError
        with pytest.raises(RuntimeError, match="Redis connection or RQ Job type not available"):
            await plugin.get_task_result("id")

        # For revoke_task, it logs a DEBUG message
        expected_revoke_debug_log = f"{plugin.plugin_id}: Cannot revoke task, Redis/RQ components not available."
        assert await plugin.revoke_task("id") is False
        assert expected_revoke_debug_log in caplog.text
