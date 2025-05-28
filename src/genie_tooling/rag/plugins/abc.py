"""Abstract Base Classes/Protocols for RAG Component Plugins."""
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
    Document,
    EmbeddingVector,
    Plugin,
    RetrievedChunk,
)

logger = logging.getLogger(__name__)

@runtime_checkable
class DocumentLoaderPlugin(Plugin, Protocol):
    """Loads documents from a source into an async stream of Document objects."""
    # plugin_id: str (from Plugin)
    async def load(self, source_uri: str, config: Optional[Dict[str, Any]] = None) -> AsyncIterable[Document]:
        """
        Loads documents from the given source URI.
        Args:
            source_uri: The URI of the data source (e.g., file path, URL, database connection string).
            config: Loader-specific configuration dictionary.
        Yields:
            Document objects.
        """
        # Example of how to make an async generator that does nothing if not implemented:
        logger.warning(f"DocumentLoaderPlugin '{self.plugin_id}' load method not fully implemented.")
        if False: # pylint: disable=false-condition
            yield # type: ignore
        return


@runtime_checkable
class TextSplitterPlugin(Plugin, Protocol):
    """Splits an async stream of Documents into an async stream of Chunks."""
    # plugin_id: str (from Plugin)
    async def split(self, documents: AsyncIterable[Document], config: Optional[Dict[str, Any]] = None) -> AsyncIterable[Chunk]:
        """
        Splits documents into smaller chunks.
        Args:
            documents: An async iterable of Document objects.
            config: Splitter-specific configuration (e.g., chunk_size, chunk_overlap).
        Yields:
            Chunk objects.
        """
        logger.warning(f"TextSplitterPlugin '{self.plugin_id}' split method not fully implemented.")
        if False: # pylint: disable=false-condition
            yield # type: ignore
        return


@runtime_checkable
class EmbeddingGeneratorPlugin(Plugin, Protocol):
    """Generates embeddings for an async stream of Chunks."""
    # plugin_id: str (from Plugin)
    # Config passed to setup or embed method might include model name, API key details
    # (KeyProvider instance should be passed in config if keys are needed).
    async def embed(self, chunks: AsyncIterable[Chunk], config: Optional[Dict[str, Any]] = None) -> AsyncIterable[Tuple[Chunk, EmbeddingVector]]:
        """
        Generates embeddings for each chunk.
        Args:
            chunks: An async iterable of Chunk objects.
            config: Embedder-specific configuration (e.g., model_name, batch_size, key_provider).
        Yields:
            Tuples of (Chunk, EmbeddingVector).
        """
        logger.warning(f"EmbeddingGeneratorPlugin '{self.plugin_id}' embed method not fully implemented.")
        if False: # pylint: disable=false-condition
            yield # type: ignore
        return

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
