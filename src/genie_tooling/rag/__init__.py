# src/genie_tooling/rag/__init__.py
# src/genie_tooling/rag/__init__.py
"""Retrieval Augmented Generation (RAG) components."""

# Import RAG plugin ABCs from their new top-level locations
from genie_tooling.document_loaders.abc import DocumentLoaderPlugin
from genie_tooling.embedding_generators.abc import EmbeddingGeneratorPlugin
from genie_tooling.retrievers.abc import RetrieverPlugin
from genie_tooling.text_splitters.abc import TextSplitterPlugin
from genie_tooling.vector_stores.abc import VectorStorePlugin

# Expose some common concrete RAG implementations from their new locations
from genie_tooling.document_loaders.impl import FileSystemLoader, WebPageLoader
from genie_tooling.text_splitters.impl import CharacterRecursiveTextSplitter


from .manager import RAGManager
from .types import (
    Chunk,
    Document,
    EmbeddingVector,
    RetrievedChunk,
)

__all__ = [
    "RAGManager",
    "Document", "Chunk", "RetrievedChunk", "EmbeddingVector",
    "DocumentLoaderPlugin",
    "TextSplitterPlugin",
    "EmbeddingGeneratorPlugin",
    "VectorStorePlugin",
    "RetrieverPlugin",
    "FileSystemLoader",
    "WebPageLoader",
    "CharacterRecursiveTextSplitter",
]