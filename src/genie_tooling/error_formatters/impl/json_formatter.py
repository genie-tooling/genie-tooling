"""JSONErrorFormatter: Formats errors as JSON objects."""
import logging
from typing import Any, Dict, cast

from genie_tooling.core.types import StructuredError

# Updated import path for ErrorFormatter
from genie_tooling.error_formatters.abc import ErrorFormatter

logger = logging.getLogger(__name__)

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
