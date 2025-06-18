# src/genie_tooling/error_handlers/impl/default_handler.py
import logging
from typing import Any, Dict, Optional

from genie_tooling.core.types import StructuredError

# Updated import path for ErrorHandler
from genie_tooling.error_handlers.abc import ErrorHandler

# To avoid circular dependency if ErrorHandler needs to reference Tool,
# we can use `Any` for the tool type hint in the handle method,
# or a forward reference if Tool itself is defined elsewhere and imported by type checking.
# from genie_tooling.tools.abc import Tool # This would be a direct import

logger = logging.getLogger(__name__)

class DefaultErrorHandler(ErrorHandler):
    """
    A default error handler that converts common Python exceptions
    into a standardized StructuredError format.
    """
    plugin_id: str = "default_error_handler_v1"
    description: str = "Default handler for common Python exceptions during tool execution."

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the DefaultErrorHandler.

        This plugin currently has no configurable options.

        Args:
            config: Configuration dictionary (not currently used).
        """
        pass

    def handle(self, exception: Exception, tool: Any, context: Optional[Dict[str, Any]]) -> StructuredError:
        tool_id = getattr(tool, "identifier", "unknown_tool_instance")
        error_type = exception.__class__.__name__
        message = str(exception)

        details: Dict[str, Any] = {
            "tool_id": tool_id,
            "exception_type": f"{type(exception).__module__}.{type(exception).__name__}",
            "exception_args": list(exception.args)
        }

        if isinstance(exception, (ConnectionError, TimeoutError)):
            details["error_category"] = "NetworkError"
        elif isinstance(exception, (ValueError, TypeError, KeyError, AttributeError, IndexError)):
            details["error_category"] = "UsageError"

        # Example for custom validation exception (if InputValidationException was defined and used)
        # from genie_tooling.input_validators.abc import InputValidationException
        # if isinstance(exception, InputValidationException):
        #    error_type = "InputValidationError"
        #    details["validation_errors"] = exception.errors
        #    details["input_params"] = exception.params

        logger.debug(f"DefaultErrorHandler: Handling {error_type} for tool '{tool_id}': {message}", exc_info=True)

        return {
            "type": error_type,
            "message": message,
            "details": details
        }

    async def teardown(self) -> None:
        pass