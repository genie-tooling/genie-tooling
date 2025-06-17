"""CeleryTaskQueuePlugin: Integrates with Celery for distributed task execution."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from genie_tooling.task_queues.abc import DistributedTaskQueuePlugin, TaskStatus

logger = logging.getLogger(__name__)

try:
    from celery import Celery
    from celery.result import AsyncResult
    CELERY_AVAILABLE = True
except ImportError:
    Celery = None # type: ignore
    AsyncResult = None # type: ignore
    CELERY_AVAILABLE = False
    logger.warning(
        "CeleryTaskQueuePlugin: 'celery' library not installed. "
        "This plugin will not be functional. Please install it: poetry add celery"
    )

class CeleryTaskQueuePlugin(DistributedTaskQueuePlugin):
    plugin_id: str = "celery_task_queue_v1"
    description: str = "Integrates with Celery for distributed task execution."

    _celery_app: Optional[Celery] = None

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        if not CELERY_AVAILABLE or not Celery:
            logger.error(f"{self.plugin_id}: Celery library not available. Cannot initialize.")
            return
        cfg = config or {}
        app_name = cfg.get("celery_app_name", "genie_celery_tasks")
        broker_url = cfg.get("celery_broker_url", "redis://localhost:6379/0") # Example
        backend_url = cfg.get("celery_backend_url", "redis://localhost:6379/0") # Example
        # include_paths = cfg.get("celery_include_task_paths", []) # e.g., ["my_project.tasks"]

        try:
            self._celery_app = Celery(app_name, broker=broker_url, backend=backend_url)
            # self._celery_app.conf.update(include=include_paths)
            # TODO: Consider how tasks are discovered/registered if this plugin is generic.
            # For now, assumes tasks are defined and discoverable by the Celery worker.
            logger.info(f"{self.plugin_id}: Celery app '{app_name}' configured. Broker: {broker_url}, Backend: {backend_url}")
        except Exception as e:
            logger.error(f"{self.plugin_id}: Failed to initialize Celery app: {e}", exc_info=True)
            self._celery_app = None

    async def submit_task(
        self, task_name: str, args: Tuple = (), kwargs: Optional[Dict[str, Any]] = None,
        queue_name: Optional[str] = None, task_options: Optional[Dict[str, Any]] = None
    ) -> str:
        if not self._celery_app:
            raise RuntimeError(f"{self.plugin_id}: Celery app not initialized.")
        kwargs = kwargs or {}
        options = task_options or {}
        if queue_name:
            options["queue"] = queue_name
        try:
            async_result = self._celery_app.send_task(task_name, args=args, kwargs=kwargs, **options)
            return async_result.id
        except Exception as e:
            logger.error(f"{self.plugin_id}: Error submitting task '{task_name}' to Celery: {e}", exc_info=True)
            raise

    async def get_task_status(self, task_id: str, queue_name: Optional[str] = None) -> TaskStatus:
        if not self._celery_app or not AsyncResult:
            return "unknown"
        try:
            result = AsyncResult(task_id, app=self._celery_app)
            # Celery states: PENDING, STARTED, SUCCESS, FAILURE, RETRY, REVOKED
            status_map: Dict[str, TaskStatus] = {
                "PENDING": "pending", "STARTED": "running", "SUCCESS": "success",
                "FAILURE": "failure", "RETRY": "running", # Treat retry as running
                "REVOKED": "revoked",
            }
            return status_map.get(result.state, "unknown")
        except Exception as e:
            logger.error(f"{self.plugin_id}: Error getting Celery task status for '{task_id}': {e}", exc_info=True)
            return "unknown"

    async def get_task_result(
        self, task_id: str, queue_name: Optional[str] = None, timeout_seconds: Optional[float] = None
    ) -> Any:
        if not self._celery_app or not AsyncResult:
            raise RuntimeError(f"{self.plugin_id}: Celery app not initialized.")
        try:
            result_obj = AsyncResult(task_id, app=self._celery_app)
            # Celery's result.get() can block, which is not ideal in async.
            # However, for a simple implementation, we might use it with run_in_executor
            # or rely on the user polling status first.
            # For now, let's assume a more direct (potentially blocking if not careful) approach
            # or that the user checks status before calling get_result.
            # A truly async get would involve Celery's event mechanisms or custom polling.
            # This is a simplification for V1.
            if not result_obj.ready():
                if timeout_seconds is not None and timeout_seconds <= 0: # Immediate return if not ready and no wait
                    raise TimeoutError("Task not ready and no timeout specified for waiting.")
                # This is a blocking call if used directly in async code without care.
                # Consider raising if not ready and no timeout, or requiring polling.
                # For now, we'll let Celery's get handle timeout.
                return result_obj.get(timeout=timeout_seconds)
            return result_obj.result # If already ready
        except Exception as e: # Celery's get() can raise exceptions for task failures
            logger.error(f"{self.plugin_id}: Error getting Celery task result for '{task_id}': {e}", exc_info=True)
            raise

    async def revoke_task(self, task_id: str, queue_name: Optional[str] = None, terminate: bool = False) -> bool:
        if not self._celery_app:
            return False
        try:
            self._celery_app.control.revoke(task_id, terminate=terminate)
            return True
        except Exception as e:
            logger.error(f"{self.plugin_id}: Error revoking Celery task '{task_id}': {e}", exc_info=True)
            return False

    async def teardown(self) -> None:
        # Celery app doesn't have a formal close/teardown method for the client object itself.
        # Connections are managed by the broker/backend.
        self._celery_app = None
        logger.info(f"{self.plugin_id}: Teardown complete.")
