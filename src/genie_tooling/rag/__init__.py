"""Retrieval Augmented Generation (RAG) components."""
from .manager import RAGManager
from .plugins.abc import (
    DocumentLoaderPlugin,
    EmbeddingGeneratorPlugin,
    RetrieverPlugin,
    TextSplitterPlugin,
    VectorStorePlugin,
)

# Expose some common concrete implementations for convenience if desired
from .plugins.impl.loaders.file_system import FileSystemLoader
from .plugins.impl.loaders.web_page import WebPageLoader
from .plugins.impl.splitters.character_recursive import CharacterRecursiveTextSplitter
from .types import (  # These are aliases from core.types
    Chunk,
    Document,
    EmbeddingVector,
    RetrievedChunk,
)

# Embedders, VectorStores, Retrievers might require more setup so not typically exported here.

__all__ = [
    "RAGManager", "Document", "Chunk", "RetrievedChunk", "EmbeddingVector",
    "DocumentLoaderPlugin", "TextSplitterPlugin", "EmbeddingGeneratorPlugin",
    "VectorStorePlugin", "RetrieverPlugin",
    "FileSystemLoader", "WebPageLoader", "CharacterRecursiveTextSplitter",
]
