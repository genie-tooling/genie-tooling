"""ConsoleTracerPlugin: Prints trace events to the console."""
import json
import logging
from typing import Any, Dict, Optional, cast

from genie_tooling.core.plugin_manager import PluginManager # For loading fallback LogAdapter
from genie_tooling.log_adapters.abc import LogAdapter as LogAdapterPlugin
from genie_tooling.log_adapters.impl.default_adapter import DefaultLogAdapter
from genie_tooling.observability.abc import InteractionTracerPlugin
from genie_tooling.observability.types import TraceEvent

logger = logging.getLogger(__name__) # Logger for the plugin itself

class ConsoleTracerPlugin(InteractionTracerPlugin):
    plugin_id: str = "console_tracer_plugin_v1"
    description: str = "Prints trace events to the console/standard output via a configured LogAdapter."

    _log_adapter_to_use: Optional[LogAdapterPlugin] = None
    _log_level: int = logging.INFO # Default log level for trace messages IF using direct logging

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._log_adapter_to_use = cfg.get("log_adapter_instance_for_console_tracer")

        if not self._log_adapter_to_use:
            logger.warning(
                f"{self.plugin_id}: LogAdapter not provided via 'log_adapter_instance_for_console_tracer' in config. "
                "Attempting to load default LogAdapter."
            )
            plugin_manager = cfg.get("plugin_manager_for_console_tracer")
            if plugin_manager and isinstance(plugin_manager, PluginManager):
                try:
                    # Attempt to load DefaultLogAdapter if none was passed
                    # This DefaultLogAdapter will need its own config, including its own PluginManager
                    # to load its redactor.
                    default_adapter_config = {"plugin_manager": plugin_manager} # Pass PM to DefaultLogAdapter
                    adapter_any = await plugin_manager.get_plugin_instance(DefaultLogAdapter.plugin_id, config=default_adapter_config)
                    if adapter_any and isinstance(adapter_any, LogAdapterPlugin):
                        self._log_adapter_to_use = cast(LogAdapterPlugin, adapter_any)
                        logger.info(f"{self.plugin_id}: Successfully loaded fallback DefaultLogAdapter.")
                    else:
                        logger.error(f"{self.plugin_id}: Failed to load fallback DefaultLogAdapter. Trace events might not be processed correctly.")
                except Exception as e_load_adapter:
                    logger.error(f"{self.plugin_id}: Error loading fallback DefaultLogAdapter: {e_load_adapter}", exc_info=True)
            else:
                logger.error(f"{self.plugin_id}: PluginManager not available in config. Cannot load fallback LogAdapter.")

        if self._log_adapter_to_use:
            logger.info(f"{self.plugin_id}: Initialized. Will use LogAdapter '{self._log_adapter_to_use.plugin_id}' for processing traces.")
        else:
            # Fallback to direct logging if no adapter could be set up
            log_level_str = cfg.get("log_level", "INFO").upper() # Legacy, for direct logging
            self._log_level = getattr(logging, log_level_str, logging.INFO)
            logger.warning(f"{self.plugin_id}: No LogAdapter available. Falling back to direct logging at level {logging.getLevelName(self._log_level)}.")


    async def record_trace(self, event: TraceEvent) -> None:
        if self._log_adapter_to_use:
            # Pass the entire event as data, event_name as type
            # schema_for_data is None as TraceEvent is not a user-defined schema here
            await self._log_adapter_to_use.process_event(
                event_type=event["event_name"],
                data=dict(event), # Convert TypedDict to plain dict for process_event
                schema_for_data=None
            )
        else:
            # Fallback direct logging (legacy behavior if LogAdapter failed)
            try:
                data_str = json.dumps(event["data"], default=str, indent=2)
                if len(data_str) > 1000:
                    data_str = data_str[:1000] + "..."
            except Exception:
                data_str = str(event["data"])[:1000] + "..."

            log_message = (
                f"CONSOLE_TRACE (direct) :: Event: {event['event_name']} | "
                f"Component: {event.get('component', 'N/A')} | "
                f"CorrID: {event.get('correlation_id', 'N/A')} | "
                f"Data: {data_str}"
            )
            logger.log(self._log_level, log_message)


    async def teardown(self) -> None:
        # The LogAdapter used by this tracer is managed externally (by Genie or InteractionTracingManager)
        # So, this tracer should not tear it down.
        self._log_adapter_to_use = None
        logger.debug(f"{self.plugin_id}: Teardown complete.")
