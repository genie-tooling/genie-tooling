"""Abstract Base Class/Protocol for DistributedTaskQueue Plugins."""
import logging
from typing import Any, Dict, Literal, Optional, Protocol, Tuple, runtime_checkable

from genie_tooling.core.types import Plugin

logger = logging.getLogger(__name__)

TaskStatus = Literal["pending", "running", "success", "failure", "revoked", "unknown"]

@runtime_checkable
class DistributedTaskQueuePlugin(Plugin, Protocol):
    """Protocol for a plugin that interacts with a distributed task queue system."""
    plugin_id: str

    async def submit_task(
        self,
        task_name: str, # Name of the task registered with the queue system
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        queue_name: Optional[str] = None, # Specific queue to send to
        task_options: Optional[Dict[str, Any]] = None # Options like countdown, eta, priority
    ) -> str:
        """
        Submits a task to the distributed queue.
        Returns:
            str: The unique ID of the submitted task.
        Raises:
            Exception: If submission fails.
        """
        logger.error(f"DistributedTaskQueuePlugin '{self.plugin_id}' submit_task not implemented.")
        raise NotImplementedError

    async def get_task_status(self, task_id: str, queue_name: Optional[str] = None) -> TaskStatus:
        """Gets the status of a previously submitted task."""
        logger.warning(f"DistributedTaskQueuePlugin '{self.plugin_id}' get_task_status not implemented.")
        return "unknown"

    async def get_task_result(
        self, task_id: str, queue_name: Optional[str] = None, timeout_seconds: Optional[float] = None
    ) -> Any:
        """
        Retrieves the result of a completed task.
        May block until the task is complete or timeout occurs.
        Raises:
            TimeoutError: If timeout is specified and exceeded.
            Exception: If task failed or other error.
        """
        logger.error(f"DistributedTaskQueuePlugin '{self.plugin_id}' get_task_result not implemented.")
        raise NotImplementedError

    async def revoke_task(self, task_id: str, queue_name: Optional[str] = None, terminate: bool = False) -> bool:
        """
        Revokes (cancels) a pending or running task.
        Args:
            terminate: If True, attempt to terminate a running task.
        Returns:
            True if revocation was successful or task was already completed/revoked.
        """
        logger.warning(f"DistributedTaskQueuePlugin '{self.plugin_id}' revoke_task not implemented.")
        return False
