"""Abstract Base Class/Protocol for ToolLookupProvider Plugins."""
import logging
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from genie_tooling.core.types import Plugin
from genie_tooling.lookup.types import (
    RankedToolResult,  # This import will need to be updated later if RankedToolResult is also moved
)

logger = logging.getLogger(__name__)

@runtime_checkable
class ToolLookupProvider(Plugin, Protocol):
    """
    Protocol for a plugin that finds relevant tools based on a query.
    """
    # plugin_id: str (from Plugin protocol)
    description: str # Human-readable description of this provider's method

    async def index_tools(self, tools_data: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None) -> None:
        """
        Builds or updates an internal index using formatted tool data.
        'tools_data' is a list of dictionaries, where each dict is typically the output of a
        DefinitionFormatterPlugin, expected to contain at least 'identifier' and some
        textual representation for matching (e.g., 'lookup_text_representation').

        Args:
            tools_data: List of formatted data for each tool.
            config: Provider-specific configuration, might include `plugin_manager` if this
                    provider needs to load sub-plugins (e.g., an embedder).

        This method might be a no-op for stateless providers that search on-the-fly.
        For stateful providers (e.g., embedding-based), this prepares the search index.
        """
        logger.warning(f"ToolLookupProvider '{self.plugin_id}' index_tools method not fully implemented.")
        pass # Default no-op for stateless providers

    async def find_tools(
        self,
        natural_language_query: str,
        top_k: int = 5,
        config: Optional[Dict[str, Any]] = None
    ) -> List[RankedToolResult]:
        """
        Searches the indexed tools (or performs a stateless search)
        based on the natural_language_query.

        Args:
            natural_language_query: The user's query for a tool's capability.
            top_k: The maximum number of ranked results to return.
            config: Provider-specific runtime configuration for the search operation.
                    Might include `plugin_manager` if query-time operations need sub-plugins.

        Returns:
            A list of RankedToolResult objects, sorted by relevance (highest score first).
        """
        logger.warning(f"ToolLookupProvider '{self.plugin_id}' find_tools method not fully implemented.")
        return []
