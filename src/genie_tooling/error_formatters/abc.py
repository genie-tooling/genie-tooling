"""Abstract Base Class/Protocol for ErrorFormatter Plugins."""
import logging
from typing import Any, Protocol, runtime_checkable

from genie_tooling.core.types import Plugin, StructuredError

logger = logging.getLogger(__name__)

@runtime_checkable
class ErrorFormatter(Plugin, Protocol):
    """Protocol for formatting StructuredError for different consumers (e.g., LLM, logs)."""
    # plugin_id: str (from Plugin protocol)

    def format(self, structured_error: StructuredError, target_format: str = "llm") -> Any:
        """
        Formats the structured_error.
        Args:
            structured_error: The error dictionary produced by an ErrorHandler.
            target_format: A hint for the desired output format (e.g., "llm", "json", "human_log").
                           Default is "llm".
        Returns:
            The formatted error (e.g., a string for LLM, a dict for JSON).
        This method is synchronous.
        """
        ...
