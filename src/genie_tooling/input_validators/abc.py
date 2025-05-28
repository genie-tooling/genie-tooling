"""Abstract Base Class/Protocol for InputValidator Plugins."""
import logging
from typing import Any, Dict, Protocol, runtime_checkable, Optional

from genie_tooling.core.types import Plugin

logger = logging.getLogger(__name__)

class InputValidationException(ValueError):
    """Custom exception for input validation errors, providing more context."""
    def __init__(self, message: str, errors: Any = None, params: Optional[Dict[str,Any]] = None):
        super().__init__(message)
        self.errors = errors
        self.params = params

@runtime_checkable
class InputValidator(Plugin, Protocol):
    """Protocol for input parameter validators."""
    plugin_id: str

    def validate(self, params: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates parameters against a schema.
        Should raise InputValidationException on failure.
        May return params (possibly coerced or with defaults applied by validator).
        This method is synchronous as validation is typically CPU-bound.
        """
        ...
