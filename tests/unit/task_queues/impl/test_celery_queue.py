import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Conditional import based on CELERY_AVAILABLE
try:
    from celery import Celery
    from celery.result import AsyncResult
    CELERY_AVAILABLE_FOR_TEST = True
except ImportError:
    CELERY_AVAILABLE_FOR_TEST = False
    Celery = MagicMock # type: ignore
    AsyncResult = MagicMock # type: ignore

from genie_tooling.task_queues.impl.celery_queue import CeleryTaskQueuePlugin

@pytest.fixture
def mock_celery_app_instance() -> MagicMock:
    app = MagicMock(spec=Celery)
    app.send_task = MagicMock(return_value=MagicMock(id="mock_task_id"))
    app.control.revoke = MagicMock()
    return app

@pytest.fixture
async def celery_queue_plugin(mock_celery_app_instance: MagicMock) -> CeleryTaskQueuePlugin:
    with patch("genie_tooling.task_queues.impl.celery_queue.Celery", return_value=mock_celery_app_instance) as mock_celery_constructor, \
         patch("genie_tooling.task_queues.impl.celery_queue.CELERY_AVAILABLE", True):
        plugin = CeleryTaskQueuePlugin()
        await plugin.setup(config={"celery_broker_url": "dummy", "celery_backend_url": "dummy"})
        return plugin

@pytest.mark.skipif(not CELERY_AVAILABLE_FOR_TEST, reason="Celery library not installed")
@pytest.mark.asyncio
async def test_celery_submit_task(celery_queue_plugin: CeleryTaskQueuePlugin, mock_celery_app_instance: MagicMock):
    plugin = await celery_queue_plugin
    task_id = await plugin.submit_task("test.task", args=(1,), kwargs={"a": 2})
    assert task_id == "mock_task_id"
    mock_celery_app_instance.send_task.assert_called_once_with("test.task", args=(1,), kwargs={"a": 2})

# Add more tests for status, result, revoke, setup failures, etc.
