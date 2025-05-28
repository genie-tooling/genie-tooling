"""Abstract Base Class/Protocol for OutputTransformer Plugins."""
import logging
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from genie_tooling.core.types import Plugin

logger = logging.getLogger(__name__)

class OutputTransformationException(ValueError):
    """Custom exception for output transformation errors."""
    def __init__(self, message: str, original_output: Any = None, schema: Optional[Dict[str,Any]] = None):
        super().__init__(message)
        self.original_output = original_output
        self.schema = schema

@runtime_checkable
class OutputTransformer(Plugin, Protocol):
    """Protocol for output transformers."""
    # plugin_id: str (from Plugin protocol)

    def transform(self, output: Any, schema: Dict[str, Any]) -> Any:
        """
        Transforms raw tool output according to an output_schema or desired format.
        Raises OutputTransformationException on failure.
        This method is synchronous as transformation is typically CPU-bound.
        """
        ...
