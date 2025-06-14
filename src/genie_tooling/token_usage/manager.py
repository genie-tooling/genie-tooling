"""TokenUsageManager: Orchestrates TokenUsageRecorderPlugins."""
import logging
from typing import Any, Dict, List, Optional, cast

from genie_tooling.core.plugin_manager import PluginManager

from .abc import TokenUsageRecorderPlugin
from .types import TokenUsageRecord

logger = logging.getLogger(__name__)

class TokenUsageManager:
    def __init__(self, plugin_manager: PluginManager, default_recorder_ids: Optional[List[str]] = None, recorder_configurations: Optional[Dict[str, Dict[str, Any]]] = None):
        self._plugin_manager = plugin_manager
        self._default_recorder_ids = default_recorder_ids or []
        self._recorder_configurations = recorder_configurations or {}
        self._active_recorders: List[TokenUsageRecorderPlugin] = []
        self._initialized = False
        logger.info("TokenUsageManager initialized.")

    async def _initialize_recorders(self) -> None:
        if self._initialized:
            return

        logger.debug(f"Initializing token usage recorders. Default IDs: {self._default_recorder_ids}")
        for recorder_id in self._default_recorder_ids:
            config = self._recorder_configurations.get(recorder_id, {})
            try:
                instance_any = await self._plugin_manager.get_plugin_instance(recorder_id, config=config)
                if instance_any and isinstance(instance_any, TokenUsageRecorderPlugin):
                    self._active_recorders.append(cast(TokenUsageRecorderPlugin, instance_any))
                    logger.info(f"Activated TokenUsageRecorderPlugin: {recorder_id}")
                elif instance_any:
                     logger.warning(f"Plugin '{recorder_id}' loaded but is not a valid TokenUsageRecorderPlugin.")
                else:
                    logger.warning(f"TokenUsageRecorderPlugin '{recorder_id}' not found or failed to load.")
            except Exception as e:
                logger.error(f"Error loading TokenUsageRecorderPlugin '{recorder_id}': {e}", exc_info=True)
        self._initialized = True

    async def record_usage(self, record: TokenUsageRecord) -> None:
        if not self._initialized:
            await self._initialize_recorders()

        if not self._active_recorders:
            # logger.debug("No active token usage recorders, skipping record_usage.")
            return

        # logger.debug(f"Recording token usage: {record}")
        for recorder in self._active_recorders:
            try:
                await recorder.record_usage(record)
            except Exception as e:
                logger.error(f"Error recording token usage with recorder '{recorder.plugin_id}': {e}", exc_info=True)

    async def get_summary(self, recorder_id: Optional[str] = None, filter_criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self._initialized:
            await self._initialize_recorders()

        summaries = {}
        recorders_to_query = []

        if recorder_id:
            for rec in self._active_recorders:
                if rec.plugin_id == recorder_id:
                    recorders_to_query.append(rec)
                    break
            if not recorders_to_query:
                return {"error": f"Recorder '{recorder_id}' not active or found."}
        else: # Get summary from all active recorders
            recorders_to_query = self._active_recorders

        if not recorders_to_query:
            return {"error": "No active token usage recorders to query."}

        for recorder in recorders_to_query:
            try:
                summaries[recorder.plugin_id] = await recorder.get_summary(filter_criteria)
            except Exception as e:
                logger.error(f"Error getting summary from recorder '{recorder.plugin_id}': {e}", exc_info=True)
                summaries[recorder.plugin_id] = {"error": f"Failed to get summary: {e!s}"}

        return summaries[recorder_id] if recorder_id and len(recorders_to_query) == 1 else summaries


    async def teardown(self) -> None:
        logger.info("TokenUsageManager tearing down active recorders...")
        for recorder in self._active_recorders:
            try:
                await recorder.teardown()
            except Exception as e:
                logger.error(f"Error tearing down token usage recorder '{recorder.plugin_id}': {e}", exc_info=True)
        self._active_recorders.clear()
        self._initialized = False
        logger.info("TokenUsageManager teardown complete.")
