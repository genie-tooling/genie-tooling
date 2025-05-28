"""Abstract Base Classes/Protocols for Retriever Plugins."""
import logging
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

from genie_tooling.core.types import ( # Assuming RetrievedChunk and Plugin are from here
    Plugin,
    RetrievedChunk,
)

logger = logging.getLogger(__name__)

@runtime_checkable
class RetrieverPlugin(Plugin, Protocol):
    """Retrieves relevant chunks based on a query, typically by composing an embedder and a vector store."""
    # plugin_id: str (from Plugin)
    # Configuration passed to setup or retrieve should specify which embedder and vector store
    # plugin IDs to use, along with their respective configurations (including any KeyProvider).
    async def retrieve(self, query: str, top_k: int, config: Optional[Dict[str, Any]] = None) -> List[RetrievedChunk]:
        """
        Retrieves relevant chunks for the given query.
        Args:
            query: The natural language query string.
            top_k: The number of top results to return.
            config: Retriever-specific configuration. This should include details for
                      the embedder (e.g., "embedder_id", "embedder_config") and
                      vector store (e.g., "vector_store_id", "vector_store_config")
                      that this retriever instance will use. It should also contain "plugin_manager".
        Returns:
            A list of RetrievedChunk objects.
        """
        logger.warning(f"RetrieverPlugin '{self.plugin_id}' retrieve method not fully implemented.")
        return []
