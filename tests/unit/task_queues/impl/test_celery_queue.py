### tests/unit/task_queues/impl/test_celery_queue.py
import logging
from typing import AsyncGenerator
from unittest.mock import MagicMock, patch

import pytest

# Conditional import based on CELERY_AVAILABLE
try:
    from celery import Celery
    from celery.exceptions import CeleryError
    from celery.result import AsyncResult
    CELERY_AVAILABLE_FOR_TEST = True
except ImportError:
    CELERY_AVAILABLE_FOR_TEST = False
    Celery = MagicMock # type: ignore
    AsyncResult = MagicMock # type: ignore
    CeleryError = Exception # type: ignore

from genie_tooling.task_queues.abc import TaskStatus
from genie_tooling.task_queues.impl.celery_queue import CeleryTaskQueuePlugin

PLUGIN_LOGGER_NAME = "genie_tooling.task_queues.impl.celery_queue"


@pytest.fixture()
def mock_celery_app_instance() -> MagicMock:
    app = MagicMock(spec=Celery)
    app.AsyncResult = lambda task_id, app_instance=app: AsyncResult(task_id, app=app_instance) # type: ignore

    # Configure send_task's return_value to be a mock with an 'id' attribute
    mock_sent_task_async_result = MagicMock(name="SentTaskAsyncResult")
    mock_sent_task_async_result.id = "mock_task_id_from_send" # This is a string
    app.send_task = MagicMock(return_value=mock_sent_task_async_result)

    app.control = MagicMock()
    app.control.revoke = MagicMock()
    return app

@pytest.fixture()
def mock_async_result_instance() -> MagicMock:
    res = MagicMock(spec=AsyncResult)
    res.id = "mock_task_id_from_async_result"
    res.state = "PENDING"
    res.ready = MagicMock(return_value=False)
    res.result = None
    res.get = MagicMock(return_value="mock_task_result_from_get")
    return res


@pytest.fixture()
async def celery_queue_plugin_fixt(
    mock_celery_app_instance: MagicMock,
    request
) -> AsyncGenerator[CeleryTaskQueuePlugin, None]:
    celery_available_for_this_test = getattr(request, "param", {}).get("celery_available", CELERY_AVAILABLE_FOR_TEST)

    with patch("genie_tooling.task_queues.impl.celery_queue.Celery", return_value=mock_celery_app_instance) as mock_celery_constructor, \
         patch("genie_tooling.task_queues.impl.celery_queue.AsyncResult", return_value=request.getfixturevalue("mock_async_result_instance")) as mock_async_result_constructor, \
         patch("genie_tooling.task_queues.impl.celery_queue.CELERY_AVAILABLE", celery_available_for_this_test):
        plugin = CeleryTaskQueuePlugin()

        # Reset and reconfigure mocks on the app instance that will be used by the plugin
        mock_celery_app_instance.send_task.reset_mock()
        new_mock_async_result_for_send = MagicMock(name="NewAsyncResultForSendTaskAfterReset")
        new_mock_async_result_for_send.id = "mock_task_id_from_send" # Ensure this is a string
        mock_celery_app_instance.send_task.return_value = new_mock_async_result_for_send

        mock_celery_app_instance.control.revoke.reset_mock()

        await plugin.setup(config={"celery_broker_url": "dummy_broker", "celery_backend_url": "dummy_backend"})
        plugin._celery_constructor_mock = mock_celery_constructor
        plugin._async_result_constructor_mock = mock_async_result_constructor
        yield plugin


@pytest.mark.skipif(not CELERY_AVAILABLE_FOR_TEST, reason="Celery library not installed")
@pytest.mark.asyncio()
class TestCeleryTaskQueuePluginSetup:
    async def test_setup_success(self, celery_queue_plugin_fixt: AsyncGenerator[CeleryTaskQueuePlugin, None]):
        plugin = await anext(celery_queue_plugin_fixt)
        assert plugin._celery_app is not None
        plugin._celery_constructor_mock.assert_called_once_with(
            "genie_celery_tasks", broker="dummy_broker", backend="dummy_backend"
        )

    @pytest.mark.parametrize("celery_queue_plugin_fixt", [{"celery_available": False}], indirect=True)
    async def test_setup_celery_not_available(self, celery_queue_plugin_fixt: AsyncGenerator[CeleryTaskQueuePlugin, None], caplog: pytest.LogCaptureFixture):
        caplog.set_level(logging.ERROR, logger=PLUGIN_LOGGER_NAME)
        plugin = await anext(celery_queue_plugin_fixt)
        assert plugin._celery_app is None
        assert f"{plugin.plugin_id}: Celery library not available. Cannot initialize." in caplog.text

    async def test_setup_celery_constructor_fails(self, mock_celery_app_instance: MagicMock, caplog: pytest.LogCaptureFixture):
        caplog.set_level(logging.ERROR, logger=PLUGIN_LOGGER_NAME)
        with patch("genie_tooling.task_queues.impl.celery_queue.Celery", side_effect=CeleryError("Init failed")) as mock_celery_constructor, \
             patch("genie_tooling.task_queues.impl.celery_queue.CELERY_AVAILABLE", True):
            plugin_fails_init = CeleryTaskQueuePlugin()
            await plugin_fails_init.setup(config={"celery_broker_url": "bad", "celery_backend_url": "bad"})
            assert plugin_fails_init._celery_app is None
            assert f"{plugin_fails_init.plugin_id}: Failed to initialize Celery app: Init failed" in caplog.text


@pytest.mark.skipif(not CELERY_AVAILABLE_FOR_TEST, reason="Celery library not installed")
@pytest.mark.asyncio()
class TestCeleryTaskQueuePluginOperations:
    async def test_submit_task_success(self, celery_queue_plugin_fixt: AsyncGenerator[CeleryTaskQueuePlugin, None], mock_celery_app_instance: MagicMock):
        plugin = await anext(celery_queue_plugin_fixt)
        task_id = await plugin.submit_task("test.task", args=(1,), kwargs={"a": 2}, queue_name="q1", task_options={"countdown": 10})
        assert task_id == "mock_task_id_from_send"
        mock_celery_app_instance.send_task.assert_called_once_with(
            "test.task", args=(1,), kwargs={"a": 2}, queue="q1", countdown=10
        )

    async def test_submit_task_app_not_initialized(self, celery_queue_plugin_fixt: AsyncGenerator[CeleryTaskQueuePlugin, None]):
        plugin = await anext(celery_queue_plugin_fixt)
        plugin._celery_app = None
        with pytest.raises(RuntimeError, match="Celery app not initialized"):
            await plugin.submit_task("test.task")

    async def test_submit_task_send_fails(self, celery_queue_plugin_fixt: AsyncGenerator[CeleryTaskQueuePlugin, None], mock_celery_app_instance: MagicMock, caplog: pytest.LogCaptureFixture):
        caplog.set_level(logging.ERROR, logger=PLUGIN_LOGGER_NAME)
        plugin = await anext(celery_queue_plugin_fixt)
        mock_celery_app_instance.send_task.side_effect = CeleryError("Broker connection error")
        with pytest.raises(CeleryError, match="Broker connection error"):
            await plugin.submit_task("test.task")
        assert f"{plugin.plugin_id}: Error submitting task 'test.task' to Celery: Broker connection error" in caplog.text

    @pytest.mark.parametrize("celery_state, expected_genie_status", [
        ("PENDING", "pending"), ("STARTED", "running"), ("SUCCESS", "success"),
        ("FAILURE", "failure"), ("RETRY", "running"), ("REVOKED", "revoked"),
        ("UNKNOWN_CELERY_STATE", "unknown"),
    ])
    async def test_get_task_status_maps_correctly(
        self, celery_queue_plugin_fixt: AsyncGenerator[CeleryTaskQueuePlugin, None], mock_async_result_instance: MagicMock,
        celery_state: str, expected_genie_status: TaskStatus
    ):
        plugin = await anext(celery_queue_plugin_fixt)
        mock_async_result_instance.state = celery_state
        status = await plugin.get_task_status("some_task_id")
        assert status == expected_genie_status
        plugin._async_result_constructor_mock.assert_called_once_with("some_task_id", app=plugin._celery_app)

    async def test_get_task_status_app_not_initialized(self, celery_queue_plugin_fixt: AsyncGenerator[CeleryTaskQueuePlugin, None]):
        plugin = await anext(celery_queue_plugin_fixt)
        plugin._celery_app = None
        status = await plugin.get_task_status("task_id")
        assert status == "unknown"

    async def test_get_task_status_async_result_fails(self, celery_queue_plugin_fixt: AsyncGenerator[CeleryTaskQueuePlugin, None], caplog: pytest.LogCaptureFixture):
        caplog.set_level(logging.ERROR, logger=PLUGIN_LOGGER_NAME)
        plugin = await anext(celery_queue_plugin_fixt)
        plugin._async_result_constructor_mock.side_effect = Exception("AsyncResult init failed")
        status = await plugin.get_task_status("task_id")
        assert status == "unknown"
        assert f"{plugin.plugin_id}: Error getting Celery task status for 'task_id': AsyncResult init failed" in caplog.text

    async def test_get_task_result_success_ready(self, celery_queue_plugin_fixt: AsyncGenerator[CeleryTaskQueuePlugin, None], mock_async_result_instance: MagicMock):
        plugin = await anext(celery_queue_plugin_fixt)
        mock_async_result_instance.ready.return_value = True
        mock_async_result_instance.result = "task_data_direct"
        result = await plugin.get_task_result("task_id_ready")
        assert result == "task_data_direct"
        mock_async_result_instance.get.assert_not_called()

    async def test_get_task_result_success_not_ready_waits(self, celery_queue_plugin_fixt: AsyncGenerator[CeleryTaskQueuePlugin, None], mock_async_result_instance: MagicMock):
        plugin = await anext(celery_queue_plugin_fixt)
        mock_async_result_instance.ready.return_value = False
        mock_async_result_instance.get.return_value = "task_data_from_get"
        result = await plugin.get_task_result("task_id_not_ready", timeout_seconds=0.1)
        assert result == "task_data_from_get"
        mock_async_result_instance.get.assert_called_once_with(timeout=0.1)

    async def test_get_task_result_app_not_initialized(self, celery_queue_plugin_fixt: AsyncGenerator[CeleryTaskQueuePlugin, None]):
        plugin = await anext(celery_queue_plugin_fixt)
        plugin._celery_app = None
        with pytest.raises(RuntimeError, match="Celery app not initialized"):
            await plugin.get_task_result("task_id")

    async def test_get_task_result_get_raises_exception(self, celery_queue_plugin_fixt: AsyncGenerator[CeleryTaskQueuePlugin, None], mock_async_result_instance: MagicMock, caplog: pytest.LogCaptureFixture):
        caplog.set_level(logging.ERROR, logger=PLUGIN_LOGGER_NAME)
        plugin = await anext(celery_queue_plugin_fixt)
        mock_async_result_instance.ready.return_value = False
        mock_async_result_instance.get.side_effect = CeleryError("Task failed in worker")
        with pytest.raises(CeleryError, match="Task failed in worker"):
            await plugin.get_task_result("task_id_get_fail")
        assert f"{plugin.plugin_id}: Error getting Celery task result for 'task_id_get_fail': Task failed in worker" in caplog.text

    async def test_revoke_task_success(self, celery_queue_plugin_fixt: AsyncGenerator[CeleryTaskQueuePlugin, None], mock_celery_app_instance: MagicMock):
        plugin = await anext(celery_queue_plugin_fixt)
        revoked = await plugin.revoke_task("task_to_revoke", terminate=True)
        assert revoked is True
        mock_celery_app_instance.control.revoke.assert_called_once_with("task_to_revoke", terminate=True)

    async def test_revoke_task_app_not_initialized(self, celery_queue_plugin_fixt: AsyncGenerator[CeleryTaskQueuePlugin, None]):
        plugin = await anext(celery_queue_plugin_fixt)
        plugin._celery_app = None
        revoked = await plugin.revoke_task("task_id")
        assert revoked is False

    async def test_revoke_task_control_fails(self, celery_queue_plugin_fixt: AsyncGenerator[CeleryTaskQueuePlugin, None], mock_celery_app_instance: MagicMock, caplog: pytest.LogCaptureFixture):
        caplog.set_level(logging.ERROR, logger=PLUGIN_LOGGER_NAME)
        plugin = await anext(celery_queue_plugin_fixt)
        mock_celery_app_instance.control.revoke.side_effect = CeleryError("Control command failed")
        revoked = await plugin.revoke_task("task_id")
        assert revoked is False
        assert f"{plugin.plugin_id}: Error revoking Celery task 'task_id': Control command failed" in caplog.text

    async def test_teardown(self, celery_queue_plugin_fixt: AsyncGenerator[CeleryTaskQueuePlugin, None], caplog: pytest.LogCaptureFixture):
        caplog.set_level(logging.INFO, logger=PLUGIN_LOGGER_NAME)
        plugin = await anext(celery_queue_plugin_fixt)
        assert plugin._celery_app is not None
        await plugin.teardown()
        assert plugin._celery_app is None
        assert f"{plugin.plugin_id}: Teardown complete." in caplog.text
