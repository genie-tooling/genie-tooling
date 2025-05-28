"""Abstract Base Class/Protocol for ErrorHandler Plugins."""
import logging
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from genie_tooling.core.types import Plugin, StructuredError

logger = logging.getLogger(__name__)

@runtime_checkable
class ErrorHandler(Plugin, Protocol):
    """Protocol for handling exceptions during tool execution and converting them to StructuredError."""
    # plugin_id: str (from Plugin protocol)

    def handle(self, exception: Exception, tool: Any, context: Optional[Dict[str, Any]]) -> StructuredError:
        """
        Handles an exception that occurred during tool.execute() or related steps.
        Args:
            exception: The exception instance caught.
            tool: The Tool instance that was being executed (typed as Any to avoid circular import).
                  Implementations can cast or use getattr to access tool.identifier.
            context: Optional context dictionary active during the call.
        Returns:
            A StructuredError dictionary.
        This method is synchronous as error classification is typically CPU-bound.
        """
        ...
