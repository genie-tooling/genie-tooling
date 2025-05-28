"""Concrete implementations of RetrieverPlugins."""
from .basic_similarity import BasicSimilarityRetriever

# Add other retriever types like MMRRetriever, SelfQueryRetriever etc.

__all__ = ["BasicSimilarityRetriever"]
