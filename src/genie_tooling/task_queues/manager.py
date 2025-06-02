"""DistributedTaskQueueManager: Orchestrates DistributedTaskQueuePlugin."""
import logging
from typing import Any, Dict, Optional, Tuple, cast

from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.task_queues.abc import DistributedTaskQueuePlugin, TaskStatus

logger = logging.getLogger(__name__)

class DistributedTaskQueueManager:
    def __init__(
        self,
        plugin_manager: PluginManager,
        default_queue_id: Optional[str] = None,
        queue_configurations: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self._plugin_manager = plugin_manager
        self._default_queue_id = default_queue_id
        self._queue_configurations = queue_configurations or {}
        self._active_queues: Dict[str, DistributedTaskQueuePlugin] = {}
        logger.info("DistributedTaskQueueManager initialized.")

    async def _get_queue_plugin(self, queue_id: Optional[str] = None) -> Optional[DistributedTaskQueuePlugin]:
        target_id = queue_id or self._default_queue_id
        if not target_id:
            logger.error("No task queue ID specified and no default is set.")
            return None

        if target_id in self._active_queues:
            return self._active_queues[target_id]

        config = self._queue_configurations.get(target_id, {})
        try:
            instance_any = await self._plugin_manager.get_plugin_instance(target_id, config=config)
            if instance_any and isinstance(instance_any, DistributedTaskQueuePlugin):
                plugin_instance = cast(DistributedTaskQueuePlugin, instance_any)
                self._active_queues[target_id] = plugin_instance
                logger.info(f"Activated DistributedTaskQueuePlugin: {target_id}")
                return plugin_instance
            elif instance_any:
                logger.warning(f"Plugin '{target_id}' loaded but is not a valid DistributedTaskQueuePlugin.")
            else:
                logger.warning(f"DistributedTaskQueuePlugin '{target_id}' not found or failed to load.")
        except Exception as e:
            logger.error(f"Error loading DistributedTaskQueuePlugin '{target_id}': {e}", exc_info=True)
        return None

    async def submit_task(
        self,
        task_name: str,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        queue_id: Optional[str] = None,
        task_options: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        plugin = await self._get_queue_plugin(queue_id)
        if not plugin:
            return None
        try:
            return await plugin.submit_task(task_name, args, kwargs, queue_id, task_options)
        except Exception as e:
            logger.error(f"Error submitting task '{task_name}' via plugin '{plugin.plugin_id}': {e}", exc_info=True)
            return None # Or re-raise specific exception

    async def get_task_status(self, task_id: str, queue_id: Optional[str] = None) -> TaskStatus:
        plugin = await self._get_queue_plugin(queue_id)
        if not plugin:
            return "unknown"
        return await plugin.get_task_status(task_id, queue_id)

    async def get_task_result(
        self, task_id: str, queue_id: Optional[str] = None, timeout_seconds: Optional[float] = None
    ) -> Any:
        plugin = await self._get_queue_plugin(queue_id)
        if not plugin:
            raise RuntimeError("Task queue plugin not available to fetch result.")
        return await plugin.get_task_result(task_id, queue_id, timeout_seconds)

    async def revoke_task(self, task_id: str, queue_id: Optional[str] = None, terminate: bool = False) -> bool:
        plugin = await self._get_queue_plugin(queue_id)
        if not plugin:
            return False
        return await plugin.revoke_task(task_id, queue_id, terminate)

    async def teardown(self) -> None:
        logger.info("DistributedTaskQueueManager tearing down active queues...")
        for queue_id, instance in list(self._active_queues.items()):
            try:
                await instance.teardown()
            except Exception as e:
                logger.error(f"Error tearing down task queue plugin '{queue_id}': {e}", exc_info=True)
        self._active_queues.clear()
        logger.info("DistributedTaskQueueManager teardown complete.")
