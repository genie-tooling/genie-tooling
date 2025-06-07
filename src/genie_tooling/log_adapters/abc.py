"""Abstract Base Classes/Protocols for LogAdapter Plugins."""
import logging
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from genie_tooling.core.types import Plugin

logger = logging.getLogger(__name__)

@runtime_checkable
class LogAdapter(Plugin, Protocol):
    """Protocol for a logging/monitoring adapter."""
    plugin_id: str
    description: str

    async def setup(self, config: Dict[str, Any]) -> None:
        """
        Configures logging handlers or integrates with external monitoring systems.
        This method is called after the adapter is instantiated.
        Args:
            config: Adapter-specific configuration dictionary. May include 'plugin_manager'
                    if this adapter needs to load other plugins (e.g., a RedactorPlugin).
        """
        pass

    async def process_event(self, event_type: str, data: Dict[str, Any], schema_for_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Processes a structured event (e.g., for monitoring or detailed logging).
        Data should be pre-sanitized by this method or by a configured RedactorPlugin.
        Args:
            event_type: A string identifying the type of event (e.g., "tool_invoked", "tool_error").
            data: A dictionary containing event-specific data.
            schema_for_data: Optional JSON schema corresponding to 'data', to aid redaction.
        """
        pass
