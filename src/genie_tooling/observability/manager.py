"""InteractionTracingManager: Orchestrates InteractionTracerPlugins."""
import asyncio
import logging
from typing import Any, Dict, List, Optional, Type, cast

from genie_tooling.core.plugin_manager import PluginManager

from .abc import InteractionTracerPlugin
from .types import TraceEvent

logger = logging.getLogger(__name__)

class InteractionTracingManager:
    def __init__(self, plugin_manager: PluginManager, default_tracer_ids: Optional[List[str]] = None, tracer_configurations: Optional[Dict[str, Dict[str, Any]]] = None):
        self._plugin_manager = plugin_manager
        self._default_tracer_ids = default_tracer_ids or []
        self._tracer_configurations = tracer_configurations or {}
        self._active_tracers: List[InteractionTracerPlugin] = []
        self._initialized = False
        logger.info("InteractionTracingManager initialized.")

    async def _initialize_tracers(self) -> None:
        if self._initialized:
            return
        
        logger.debug(f"Initializing tracers. Default IDs: {self._default_tracer_ids}")
        for tracer_id in self._default_tracer_ids:
            config = self._tracer_configurations.get(tracer_id, {})
            try:
                instance_any = await self._plugin_manager.get_plugin_instance(tracer_id, config=config)
                if instance_any and isinstance(instance_any, InteractionTracerPlugin):
                    self._active_tracers.append(cast(InteractionTracerPlugin, instance_any))
                    logger.info(f"Activated InteractionTracerPlugin: {tracer_id}")
                elif instance_any:
                    logger.warning(f"Plugin '{tracer_id}' loaded but is not a valid InteractionTracerPlugin.")
                else:
                    logger.warning(f"InteractionTracerPlugin '{tracer_id}' not found or failed to load.")
            except Exception as e:
                logger.error(f"Error loading InteractionTracerPlugin '{tracer_id}': {e}", exc_info=True)
        self._initialized = True

    async def trace_event(self, event_name: str, data: Dict[str, Any], component: Optional[str] = None, correlation_id: Optional[str] = None) -> None:
        if not self._initialized:
            await self._initialize_tracers()

        if not self._active_tracers:
            # logger.debug("No active tracers, skipping trace_event.")
            return

        event = TraceEvent(
            event_name=event_name,
            data=data,
            component=component,
            correlation_id=correlation_id,
            timestamp=asyncio.get_event_loop().time() # Using loop time for consistency
        )
        
        # logger.debug(f"Tracing event: {event_name} from component {component or 'N/A'}")
        for tracer in self._active_tracers:
            try:
                await tracer.record_trace(event)
            except Exception as e:
                logger.error(f"Error recording trace with tracer '{tracer.plugin_id}': {e}", exc_info=True)

    async def teardown(self) -> None:
        logger.info("InteractionTracingManager tearing down active tracers...")
        for tracer in self._active_tracers:
            try:
                await tracer.teardown()
            except Exception as e:
                logger.error(f"Error tearing down tracer '{tracer.plugin_id}': {e}", exc_info=True)
        self._active_tracers.clear()
        self._initialized = False
        logger.info("InteractionTracingManager teardown complete.")
