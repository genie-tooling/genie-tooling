"""DistributedTaskInvocationStrategy: Offloads tool execution to a task queue."""
import asyncio
import logging
from typing import Any, Dict, Optional

from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.invocation_strategies.abc import InvocationStrategy
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.task_queues.abc import DistributedTaskQueuePlugin
from genie_tooling.tools.abc import Tool

logger = logging.getLogger(__name__)

# Name of the generic Celery/RQ task that knows how to execute a Genie tool.
# This task would need to be defined in the worker environment.
GENERIC_TOOL_EXECUTION_TASK_NAME = "genie_tooling.worker_tasks.execute_genie_tool_task"

class DistributedTaskInvocationStrategy(InvocationStrategy):
    plugin_id: str = "distributed_task_invocation_strategy_v1"
    description: str = "Invokes tools by submitting them as tasks to a distributed task queue."

    _task_queue_plugin: Optional[DistributedTaskQueuePlugin] = None
    _default_task_queue_id: Optional[str] = None # From MiddlewareConfig

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        plugin_manager = cfg.get("plugin_manager")
        if not plugin_manager or not isinstance(plugin_manager, PluginManager):
            logger.error(f"{self.plugin_id}: PluginManager not provided in config. Cannot load task queue plugin.")
            return

        # This strategy needs to know which task queue plugin to use.
        # This could come from its own config or a global default.
        # For now, assume it's passed via invoker_config or a specific config key.
        task_queue_plugin_id = cfg.get("task_queue_plugin_id") # Specific config for this strategy instance
        if not task_queue_plugin_id:
            # Try to get a default from a hypothetical global config if available
            # This part is a bit hand-wavy without knowing how Genie passes global defaults to strategies
            # For now, let's assume it must be configured directly for the strategy.
            logger.error(f"{self.plugin_id}: 'task_queue_plugin_id' not provided in strategy config.")
            return

        queue_plugin_instance = await plugin_manager.get_plugin_instance(task_queue_plugin_id)
        if queue_plugin_instance and isinstance(queue_plugin_instance, DistributedTaskQueuePlugin):
            self._task_queue_plugin = queue_plugin_instance
            logger.info(f"{self.plugin_id}: Initialized with task queue plugin '{task_queue_plugin_id}'.")
        else:
            logger.error(f"{self.plugin_id}: Task queue plugin '{task_queue_plugin_id}' not found or invalid.")

    async def invoke(
        self,
        tool: Tool,
        params: Dict[str, Any],
        key_provider: KeyProvider, # For potentially passing specific keys to the task
        context: Optional[Dict[str, Any]],
        invoker_config: Dict[str, Any] # Contains plugin_manager, other IDs
    ) -> Any:
        if not self._task_queue_plugin:
            logger.error(f"{self.plugin_id}: Task queue plugin not available. Cannot submit task.")
            return {"error": "Task queue system unavailable for tool execution."}

        # Prepare task arguments
        # WARNING: Passing the entire key_provider or context can be a security risk
        # and might not be serializable. For a robust solution, only pass necessary,
        # serializable data. For API keys, the worker should ideally fetch them itself
        # or be provided with short-lived credentials specific to the task.
        # This is a simplified example.
        task_kwargs = {
            "tool_id": tool.identifier,
            "tool_params": params,
            # "key_provider_info": {"type": "placeholder", "details": "Securely pass/fetch keys in worker"},
            "context_info": context, # Context might contain non-serializable items
        }
        # Example: If a tool needs a specific API key, fetch it here and pass the value.
        # tool_metadata = await tool.get_metadata()
        # required_key_name = tool_metadata.get("key_requirements", [{}])[0].get("name") if tool_metadata.get("key_requirements") else None
        # if required_key_name:
        #     task_kwargs["api_key_value"] = await key_provider.get_key(required_key_name)


        try:
            logger.info(f"{self.plugin_id}: Submitting tool '{tool.identifier}' as task '{GENERIC_TOOL_EXECUTION_TASK_NAME}'.")
            task_id = await self._task_queue_plugin.submit_task(
                task_name=GENERIC_TOOL_EXECUTION_TASK_NAME,
                kwargs=task_kwargs
                # queue_name can be specified here if needed
                # task_options can be specified here (e.g., for Celery: countdown, eta)
            )
            logger.info(f"{self.plugin_id}: Task for tool '{tool.identifier}' submitted with ID: {task_id}.")

            # For V1, this strategy will poll for the result.
            # More advanced versions could return a future or use callbacks.
            # This polling loop is simplified.
            polling_timeout = invoker_config.get("distributed_task_timeout_seconds", 60.0)
            polling_interval = invoker_config.get("distributed_task_poll_interval_seconds", 2.0)
            elapsed_time = 0.0

            while elapsed_time < polling_timeout:
                status = await self._task_queue_plugin.get_task_status(task_id)
                logger.debug(f"Task {task_id} status: {status}")
                if status == "success":
                    return await self._task_queue_plugin.get_task_result(task_id)
                elif status == "failure":
                    # Attempt to get the error result from the task
                    try:
                        failure_result = await self._task_queue_plugin.get_task_result(task_id)
                        error_message = f"Task execution failed: {str(failure_result)}"
                    except Exception as e_res:
                        error_message = f"Task execution failed, and result retrieval also failed: {e_res}"
                    logger.error(f"{self.plugin_id}: {error_message} (Task ID: {task_id})")
                    return {"error": error_message, "task_id": task_id}
                elif status in ["revoked", "unknown"]:
                    error_message = f"Task '{task_id}' was {status} or status is unknown."
                    logger.error(f"{self.plugin_id}: {error_message}")
                    return {"error": error_message, "task_id": task_id}

                await asyncio.sleep(polling_interval)
                elapsed_time += polling_interval

            logger.warning(f"{self.plugin_id}: Polling timeout for task '{task_id}' after {polling_timeout}s.")
            return {"error": "Task polling timed out.", "task_id": task_id}

        except Exception as e:
            logger.error(f"{self.plugin_id}: Error during distributed task invocation for tool '{tool.identifier}': {e}", exc_info=True)
            return {"error": f"Failed to invoke tool via task queue: {str(e)}"}

    async def teardown(self) -> None:
        # The task queue plugin itself is managed by PluginManager or Genie,
        # so this strategy doesn't need to tear it down.
        self._task_queue_plugin = None
        logger.info(f"{self.plugin_id}: Teardown complete.")
