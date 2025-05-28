"""Error Handling and Formatting components for tool invocation."""
import logging
from typing import Any, Dict, Optional, Protocol, cast, runtime_checkable

from genie_tooling.core.types import Plugin, StructuredError

# from genie_tooling.tools.abc import Tool # Avoid direct import for Tool here to prevent circularity

logger = logging.getLogger(__name__)

DEFAULT_ERROR_HANDLER_ID = "default_error_handler_v1"
LLM_ERROR_FORMATTER_ID = "llm_error_formatter_v1"
JSON_ERROR_FORMATTER_ID = "json_error_formatter_v1"

DEFAULT_INVOKER_ERROR_FORMATTER_ID = LLM_ERROR_FORMATTER_ID

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


class DefaultErrorHandler(ErrorHandler):
    """
    A default error handler that converts common Python exceptions
    into a standardized StructuredError format.
    """
    plugin_id: str = "default_error_handler_v1"
    description: str = "Default handler for common Python exceptions during tool execution."

    def handle(self, exception: Exception, tool: Any, context: Optional[Dict[str, Any]]) -> StructuredError:
        tool_id = getattr(tool, "identifier", "unknown_tool_instance")
        error_type = exception.__class__.__name__
        message = str(exception)

        details: Dict[str, Any] = {
            "tool_id": tool_id,
            "exception_type": f"{type(exception).__module__}.{type(exception).__name__}",
            "exception_args": list(exception.args)
        }

        # Add more specific details based on exception type if needed
        if isinstance(exception, (ConnectionError, TimeoutError)): # httpx.TimeoutException also inherits TimeoutError
            details["error_category"] = "NetworkError"
        elif isinstance(exception, (ValueError, TypeError, KeyError, AttributeError, IndexError)):
            details["error_category"] = "UsageError" # Programming or data error by user/tool
        # Example for custom validation exception
        # from .validation import InputValidationException # Local import to avoid top-level circular
        # if isinstance(exception, InputValidationException):
        #    error_type = "InputValidationError" # Override generic ValueError
        #    details["validation_errors"] = exception.errors # If it carries more detailed errors
        #    details["input_params"] = exception.params


        logger.debug(f"DefaultErrorHandler: Handling {error_type} for tool '{tool_id}': {message}", exc_info=True)

        return {
            "type": error_type,
            "message": message,
            "details": details
        }

    # async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass
    # async def teardown(self) -> None: pass


class LLMErrorFormatter(ErrorFormatter):
    """Formats errors into a concise string suitable for feeding back to LLMs."""
    plugin_id: str = "llm_error_formatter_v1"
    description: str = "Formats errors as concise strings for LLM consumption."

    def format(self, structured_error: StructuredError, target_format: str = "llm") -> str:
        if target_format != "llm":
            logger.warning(f"LLMErrorFormatter received target_format '{target_format}', but only supports 'llm'.")

        err_type = structured_error.get("type", "UnknownError")
        err_message = structured_error.get("message", "An unspecified error occurred.")
        tool_id = structured_error.get("details", {}).get("tool_id")

        # Make message more context-aware for the LLM
        if tool_id:
            formatted_msg = f"Error executing tool '{tool_id}' ({err_type}): {err_message}"
        else:
            formatted_msg = f"Error ({err_type}): {err_message}"

        # Optionally add a suggestion if present in structured_error
        suggestion = structured_error.get("suggestion")
        if suggestion:
            formatted_msg += f" Suggestion: {suggestion}"

        logger.debug(f"LLMErrorFormatter: Formatted error: {formatted_msg}")
        return formatted_msg

    # async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass
    # async def teardown(self) -> None: pass


class JSONErrorFormatter(ErrorFormatter):
    """Formats errors as a JSON structure (effectively returning the StructuredError dict)."""
    plugin_id: str = "json_error_formatter_v1"
    description: str = "Formats errors as JSON objects (returns the StructuredError dictionary)."

    def format(self, structured_error: StructuredError, target_format: str = "json") -> Dict[str, Any]:
        if target_format != "json":
            logger.warning(f"JSONErrorFormatter received target_format '{target_format}', but primarily returns JSON dict.")

        logger.debug(f"JSONErrorFormatter: Returning structured error: {structured_error}")
        return cast(Dict[str, Any], structured_error) # Ensure it's treated as a dict by type checker

    # async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass
    # async def teardown(self) -> None: pass
