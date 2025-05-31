"""Abstract Base Class/Protocol for TokenUsageRecorder Plugins."""
import logging
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from genie_tooling.core.types import Plugin

from .types import TokenUsageRecord

logger = logging.getLogger(__name__)

@runtime_checkable
class TokenUsageRecorderPlugin(Plugin, Protocol):
    """Protocol for a plugin that records LLM token usage."""
    plugin_id: str # From Plugin

    async def record_usage(self, record: TokenUsageRecord) -> None:
        """Records a single token usage event."""
        logger.warning(f"TokenUsageRecorderPlugin '{self.plugin_id}' record_usage method not fully implemented.")
        pass

    async def get_summary(self, filter_criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Retrieves a summary of token usage.
        The structure of the summary is implementation-dependent.
        """
        logger.warning(f"TokenUsageRecorderPlugin '{self.plugin_id}' get_summary method not fully implemented.")
        return {"error": "Not implemented"}

    async def clear_records(self, filter_criteria: Optional[Dict[str, Any]] = None) -> bool:
        """Clears recorded usage data, optionally based on criteria."""
        logger.warning(f"TokenUsageRecorderPlugin '{self.plugin_id}' clear_records method not fully implemented.")
        return False
