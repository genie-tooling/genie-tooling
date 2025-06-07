import json
import logging
from typing import Any, Dict, Optional, cast

from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.log_adapters.abc import LogAdapter as LogAdapterPlugin
from genie_tooling.log_adapters.impl.default_adapter import DefaultLogAdapter
from genie_tooling.observability.abc import InteractionTracerPlugin
from genie_tooling.observability.types import TraceEvent

logger = logging.getLogger(__name__)

class ConsoleTracerPlugin(InteractionTracerPlugin):
    plugin_id: str = "console_tracer_plugin_v1"
    description: str = "Prints trace events to the console/standard output via a configured LogAdapter."

    _log_adapter_to_use: Optional[LogAdapterPlugin] = None
    _log_level: int = logging.INFO

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._log_adapter_to_use = cfg.get("log_adapter_instance_for_console_tracer")
        if not self._log_adapter_to_use:
            logger.warning(f"{self.plugin_id}: LogAdapter not provided. Attempting to load default LogAdapter.")
            plugin_manager = cfg.get("plugin_manager_for_console_tracer") # Key used in InteractionTracingManager
            if plugin_manager and isinstance(plugin_manager, PluginManager):
                try:
                    default_adapter_config = {"plugin_manager": plugin_manager}
                    adapter_any = await plugin_manager.get_plugin_instance(DefaultLogAdapter.plugin_id, config=default_adapter_config)
                    if adapter_any and isinstance(adapter_any, LogAdapterPlugin): self._log_adapter_to_use = cast(LogAdapterPlugin, adapter_any); logger.info(f"{self.plugin_id}: Successfully loaded fallback DefaultLogAdapter.")
                    else: logger.error(f"{self.plugin_id}: Failed to load fallback DefaultLogAdapter. Trace events might not be processed correctly.")
                except Exception as e_load_adapter: logger.error(f"{self.plugin_id}: Error loading fallback DefaultLogAdapter: {e_load_adapter}", exc_info=True)
            else: logger.error(f"{self.plugin_id}: PluginManager not available in config. Cannot load fallback LogAdapter.")
        if self._log_adapter_to_use: logger.info(f"{self.plugin_id}: Initialized. Will use LogAdapter '{self._log_adapter_to_use.plugin_id}' for processing traces.")
        else: log_level_str = cfg.get("log_level", "INFO").upper(); self._log_level = getattr(logging, log_level_str, logging.INFO); logger.warning(f"{self.plugin_id}: No LogAdapter available. Falling back to direct logging at level {logging.getLevelName(self._log_level)}.")

    async def record_trace(self, event: TraceEvent) -> None:
        if self._log_adapter_to_use: await self._log_adapter_to_use.process_event(event_type=event["event_name"], data=dict(event), schema_for_data=None)
        else:
            try: data_str = json.dumps(event["data"], default=str, indent=2); data_str = data_str[:1000] + "..." if len(data_str) > 1000 else data_str
            except Exception: data_str = str(event["data"])[:1000] + "..."
            log_message = (f"CONSOLE_TRACE (direct) :: Event: {event['event_name']} | Component: {event.get('component', 'N/A')} | CorrID: {event.get('correlation_id', 'N/A')} | Data: {data_str}")
            logger.log(self._log_level, log_message)

    async def teardown(self) -> None:
        self._log_adapter_to_use = None
        logger.debug(f"{self.plugin_id}: Teardown complete.")
