"""Pluggable components for RAG pipelines."""
from .abc import (
    DocumentLoaderPlugin,
    EmbeddingGeneratorPlugin,
    RetrieverPlugin,
    TextSplitterPlugin,
    VectorStorePlugin,
)
from .impl.embedders.openai_embed import OpenAIEmbeddingGenerator
from .impl.embedders.sentence_transformer import SentenceTransformerEmbedder
from .impl.loaders.file_system import FileSystemLoader
from .impl.loaders.web_page import WebPageLoader
from .impl.retrievers.basic_similarity import BasicSimilarityRetriever
from .impl.splitters.character_recursive import CharacterRecursiveTextSplitter
from .impl.vector_stores.chromadb_store import ChromaDBVectorStore
from .impl.vector_stores.faiss_store import FAISSVectorStore

__all__ = [
    "DocumentLoaderPlugin", "TextSplitterPlugin", "EmbeddingGeneratorPlugin",
    "VectorStorePlugin", "RetrieverPlugin",
    "FileSystemLoader", "WebPageLoader",
    "CharacterRecursiveTextSplitter",
    "SentenceTransformerEmbedder", "OpenAIEmbeddingGenerator",
    "FAISSVectorStore", "ChromaDBVectorStore",
    "BasicSimilarityRetriever",
]
