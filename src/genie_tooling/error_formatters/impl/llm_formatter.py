"""LLMErrorFormatter: Formats errors for LLM consumption."""
import logging

from genie_tooling.core.types import StructuredError

# Updated import path for ErrorFormatter
from genie_tooling.error_formatters.abc import ErrorFormatter

logger = logging.getLogger(__name__)

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

        if tool_id:
            formatted_msg = f"Error executing tool '{tool_id}' ({err_type}): {err_message}"
        else:
            formatted_msg = f"Error ({err_type}): {err_message}"

        suggestion = structured_error.get("suggestion")
        if suggestion:
            formatted_msg += f" Suggestion: {suggestion}"

        logger.debug(f"LLMErrorFormatter: Formatted error: {formatted_msg}")
        return formatted_msg

    # async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass
    # async def teardown(self) -> None: pass
