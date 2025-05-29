# src/genie_tooling/rag/__init__.py
"""Retrieval Augmented Generation (RAG) components."""

# Import RAG plugin ABCs from their new top-level locations
from genie_tooling.document_loaders.abc import DocumentLoaderPlugin

# Expose some common concrete RAG implementations from their new locations
from genie_tooling.document_loaders.impl import FileSystemLoader, WebPageLoader
from genie_tooling.embedding_generators.abc import EmbeddingGeneratorPlugin
from genie_tooling.retrievers.abc import RetrieverPlugin
from genie_tooling.text_splitters.abc import TextSplitterPlugin
from genie_tooling.text_splitters.impl import CharacterRecursiveTextSplitter
from genie_tooling.vector_stores.abc import VectorStorePlugin

from .manager import RAGManager
from .types import (  # Aliases from core.types for RAG context
    Chunk,
    Document,
    EmbeddingVector,
    RetrievedChunk,
)

# Other concrete implementations like embedders, vector stores, retrievers are usually
# loaded via plugin ID by the RAGManager or Genie facade, so not typically re-exported here.

__all__ = [
    "RAGManager",
    "Document", "Chunk", "RetrievedChunk", "EmbeddingVector", # Core RAG types
    # RAG Plugin ABCs
    "DocumentLoaderPlugin",
    "TextSplitterPlugin",
    "EmbeddingGeneratorPlugin",
    "VectorStorePlugin",
    "RetrieverPlugin",
    # Common Concrete RAG Implementations
    "FileSystemLoader",
    "WebPageLoader",
    "CharacterRecursiveTextSplitter",
]
