"""Abstract Base Classes/Protocols for Redactor Plugins."""
import logging
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from genie_tooling.core.types import Plugin

logger = logging.getLogger(__name__)

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
