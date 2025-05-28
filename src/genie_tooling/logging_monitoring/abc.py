"""Abstract Base Classes/Protocols for Logging/Monitoring and Redaction Plugins."""
import logging
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from genie_tooling.core.types import Plugin

logger = logging.getLogger(__name__)

@runtime_checkable
class LogAdapter(Plugin, Protocol):
    """Protocol for a logging/monitoring adapter."""
    # plugin_id: str (from Plugin protocol)
    description: str # Human-readable description of this adapter

    async def setup_logging(self, config: Dict[str, Any]) -> None:
        """
        Configures logging handlers or integrates with external monitoring systems.
        This method is called after the adapter is instantiated.
        Args:
            config: Adapter-specific configuration dictionary. May include 'plugin_manager'
                    if this adapter needs to load other plugins (e.g., a RedactorPlugin).
        """
        logger.warning(f"LogAdapter '{self.plugin_id}' setup_logging method not fully implemented.")
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
        logger.warning(f"LogAdapter '{self.plugin_id}' process_event method not fully implemented.")
        pass

@runtime_checkable
class Redactor(Plugin, Protocol):
    """Protocol for a data redaction plugin."""
    # plugin_id: str (from Plugin protocol)
    description: str # Human-readable description of this redactor

    def sanitize(self, data: Any, schema_hints: Optional[Dict[str, Any]] = None) -> Any:
        """
        Sanitizes data to remove or mask sensitive information.
        This method is synchronous as redaction is typically CPU-bound.
        Args:
            data: The data to sanitize (can be any Python object, commonly dicts or lists).
            schema_hints: Optional JSON schema corresponding to 'data'. If provided,
                          the redactor can use schema annotations (e.g., 'x-sensitive', 'format')
                          to guide redaction.
        Returns:
            The sanitized data, with sensitive parts replaced or removed.
        """
        logger.warning(f"Redactor '{self.plugin_id}' sanitize method not fully implemented. Returning data as is.")
        return data # Default: no redaction
