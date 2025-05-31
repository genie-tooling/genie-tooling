"""Abstract Base Class/Protocol for InteractionTracer Plugins."""
import logging
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from genie_tooling.core.types import Plugin

from .types import TraceEvent

logger = logging.getLogger(__name__)

@runtime_checkable
class InteractionTracerPlugin(Plugin, Protocol):
    """Protocol for a plugin that records interaction traces."""
    plugin_id: str # From Plugin

    async def record_trace(self, event: TraceEvent) -> None:
        """
        Records a single trace event.
        Implementations should handle batching or asynchronous sending if needed.
        """
        logger.warning(f"InteractionTracerPlugin '{self.plugin_id}' record_trace method not fully implemented.")
        pass
