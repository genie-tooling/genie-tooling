"""ConsoleTracerPlugin: Prints trace events to the console."""
import json
import logging
from typing import Any, Dict, Optional

from genie_tooling.observability.abc import InteractionTracerPlugin
from genie_tooling.observability.types import TraceEvent

logger = logging.getLogger(__name__) # Logger for the plugin itself

class ConsoleTracerPlugin(InteractionTracerPlugin):
    plugin_id: str = "console_tracer_plugin_v1"
    description: str = "Prints trace events to the console/standard output."

    _log_level: int = logging.INFO # Default log level for trace messages

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        log_level_str = cfg.get("log_level", "INFO").upper()
        self._log_level = getattr(logging, log_level_str, logging.INFO)
        logger.info(f"{self.plugin_id}: Initialized. Trace events will be logged at level {logging.getLevelName(self._log_level)}.")

    async def record_trace(self, event: TraceEvent) -> None:
        try:
            # Basic serialization for logging complex data
            data_str = json.dumps(event['data'], default=str, indent=2)
            if len(data_str) > 1000: # Truncate very long data
                data_str = data_str[:1000] + "..."
        except Exception:
            data_str = str(event['data'])[:1000] + "..."
            
        log_message = (
            f"TRACE :: Event: {event['event_name']} | "
            f"Component: {event.get('component', 'N/A')} | "
            f"CorrID: {event.get('correlation_id', 'N/A')} | "
            f"Data: {data_str}"
        )
        # Use the plugin's own logger to output, respecting its configured level
        logger.log(self._log_level, log_message)

    async def teardown(self) -> None:
        logger.debug(f"{self.plugin_id}: Teardown complete.")
