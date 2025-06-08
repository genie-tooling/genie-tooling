"""Abstract Base Class/Protocol for ToolLookupProvider Plugins."""
import logging
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from genie_tooling.core.types import Plugin
from genie_tooling.lookup.types import (
    RankedToolResult,
)

logger = logging.getLogger(__name__)

@runtime_checkable
class ToolLookupProvider(Plugin, Protocol):
    """
    Protocol for a plugin that finds relevant tools based on a query.
    """
    description: str # Human-readable description of this provider's method

    async def index_tools(self, tools_data: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None) -> None:
        """
        Builds or completely replaces an internal index using formatted tool data.
        This is for full, batch re-indexing. For dynamic updates, use add/update/remove_tool.
        """
        logger.warning(f"ToolLookupProvider '{self.plugin_id}' index_tools method not fully implemented.")
        pass

    async def add_tool(self, tool_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Adds a single tool to the index.
        Returns:
            True if the tool was added successfully, False otherwise.
        """
        logger.warning(f"ToolLookupProvider '{self.plugin_id}' add_tool method not implemented.")
        return False

    async def update_tool(self, tool_id: str, tool_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Updates an existing tool in the index. This may be an add/overwrite operation.
        Returns:
            True if the tool was updated successfully, False otherwise.
        """
        logger.warning(f"ToolLookupProvider '{self.plugin_id}' update_tool method not implemented.")
        return False

    async def remove_tool(self, tool_id: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Removes a single tool from the index by its ID.
        Returns:
            True if the tool was removed or did not exist, False on failure.
        """
        logger.warning(f"ToolLookupProvider '{self.plugin_id}' remove_tool method not implemented.")
        return False

    async def find_tools(
        self,
        natural_language_query: str,
        top_k: int = 5,
        config: Optional[Dict[str, Any]] = None
    ) -> List[RankedToolResult]:
        """
        Searches the indexed tools based on the natural_language_query.
        Returns:
            A list of RankedToolResult objects, sorted by relevance (highest score first).
        """
        logger.warning(f"ToolLookupProvider '{self.plugin_id}' find_tools method not fully implemented.")
        return []
