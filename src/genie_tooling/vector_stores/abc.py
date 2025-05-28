"""Abstract Base Classes/Protocols for Vector Store Plugins."""
import logging
from typing import (
    Any,
    AsyncIterable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

from genie_tooling.core.types import (
    Chunk,
    EmbeddingVector, # Make sure this is imported if "EmbeddingVector" is used as a type hint
    Plugin,
    RetrievedChunk,
)

logger = logging.getLogger(__name__)

@runtime_checkable
class VectorStorePlugin(Plugin, Protocol):
    """Interface for interacting with a vector database."""
    # plugin_id: str (from Plugin)

    async def add(self, embeddings: AsyncIterable[Tuple[Chunk, EmbeddingVector]], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: # Returns status/count
        """
        Adds chunks and their embeddings to the vector store.
        Should handle batching internally if the input stream is large.
        Args:
            embeddings: An async iterable of (Chunk, EmbeddingVector) tuples.
            config: Vector store-specific configuration (e.g., collection name, batch_size).
        Returns:
            A dictionary with status, e.g., {"added_count": int, "errors": List[str]}
        """
        logger.warning(f"VectorStorePlugin '{self.plugin_id}' add method not fully implemented.")
        return {"added_count": 0, "errors": ["Not implemented"]}

    async def search(self, query_embedding: EmbeddingVector, top_k: int, filter_metadata: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None) -> List[RetrievedChunk]:
        """
        Searches the vector store for chunks similar to the query_embedding.
        Args:
            query_embedding: The embedding vector of the query.
            top_k: The number of top results to return.
            filter_metadata: Optional metadata filter to apply during search.
            config: Search-specific configuration.
        Returns:
            A list of RetrievedChunk objects.
        """
        logger.warning(f"VectorStorePlugin '{self.plugin_id}' search method not fully implemented.")
        return []

    async def delete(self, ids: Optional[List[str]] = None, filter_metadata: Optional[Dict[str, Any]] = None, delete_all: bool = False, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Deletes items from the vector store.
        Args:
            ids: Optional list of chunk IDs to delete.
            filter_metadata: Optional metadata filter to select items for deletion.
            delete_all: If True, delete all items in the collection/store.
            config: Deletion-specific configuration.
        Returns:
            True if deletion was successful (or partially successful), False otherwise.
        """
        logger.warning(f"VectorStorePlugin '{self.plugin_id}' delete method not fully implemented.")
        return False

    # Optional: Some vector stores require explicit collection creation.
    # async def ensure_collection(self, collection_name: str, embedding_dim: int, config: Optional[Dict[str, Any]] = None) -> bool:
    #     """Ensures a collection/index exists with the specified parameters."""
    #     logger.warning(f"VectorStorePlugin '{self.plugin_id}' ensure_collection method not implemented.")
    #     return False
