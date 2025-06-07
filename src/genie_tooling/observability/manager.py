import asyncio
import logging
from typing import Any, Dict, List, Optional, cast

from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.log_adapters.abc import LogAdapter as LogAdapterPlugin

from .abc import InteractionTracerPlugin
from .types import TraceEvent

logger = logging.getLogger(__name__)

class InteractionTracingManager:
    def __init__(
        self,
        plugin_manager: PluginManager,
        default_tracer_ids: Optional[List[str]] = None,
        tracer_configurations: Optional[Dict[str, Dict[str, Any]]] = None,
        log_adapter_instance: Optional[LogAdapterPlugin] = None
    ):
        self._plugin_manager = plugin_manager
        self._default_tracer_ids = default_tracer_ids or []
        self._tracer_configurations = tracer_configurations or {}
        self._active_tracers: List[InteractionTracerPlugin] = []
        self._initialized = False
        self._log_adapter_instance = log_adapter_instance
        logger.info("InteractionTracingManager initialized.")
        if self._log_adapter_instance: logger.info(f"InteractionTracingManager will use LogAdapter: {self._log_adapter_instance.plugin_id}")

    async def _initialize_tracers(self) -> None:
        if self._initialized: return
        logger.debug(f"Initializing tracers. Default IDs: {self._default_tracer_ids}")
        for tracer_id in self._default_tracer_ids:
            config = self._tracer_configurations.get(tracer_id, {}).copy()
            config["plugin_manager_for_console_tracer"] = self._plugin_manager # Pass PM for ConsoleTracer's potential fallback
            if self._log_adapter_instance and tracer_id == "console_tracer_plugin_v1": config["log_adapter_instance_for_console_tracer"] = self._log_adapter_instance
            try:
                instance_any = await self._plugin_manager.get_plugin_instance(tracer_id, config=config)
                if instance_any and isinstance(instance_any, InteractionTracerPlugin): self._active_tracers.append(cast(InteractionTracerPlugin, instance_any)); logger.info(f"Activated InteractionTracerPlugin: {tracer_id}")
                elif instance_any: logger.warning(f"Plugin '{tracer_id}' loaded but is not a valid InteractionTracerPlugin.")
                else: logger.warning(f"InteractionTracerPlugin '{tracer_id}' not found or failed to load.")
            except Exception as e: logger.error(f"Error loading InteractionTracerPlugin '{tracer_id}': {e}", exc_info=True)
        self._initialized = True

    async def trace_event(self, event_name: str, data: Dict[str, Any], component: Optional[str] = None, correlation_id: Optional[str] = None) -> None:
        if not self._initialized: await self._initialize_tracers()
        if not self._active_tracers: return
        event = TraceEvent(event_name=event_name, data=data, component=component, correlation_id=correlation_id, timestamp=asyncio.get_event_loop().time())
        for tracer in self._active_tracers:
            try: await tracer.record_trace(event)
            except Exception as e: logger.error(f"Error recording trace with tracer '{tracer.plugin_id}': {e}", exc_info=True)

    async def teardown(self) -> None:
        logger.info("InteractionTracingManager tearing down active tracers...")
        for tracer in self._active_tracers:
            try: await tracer.teardown()
            except Exception as e: logger.error(f"Error tearing down tracer '{tracer.plugin_id}': {e}", exc_info=True)
        self._active_tracers.clear(); self._initialized = False; self._log_adapter_instance = None
        logger.info("InteractionTracingManager teardown complete.")
