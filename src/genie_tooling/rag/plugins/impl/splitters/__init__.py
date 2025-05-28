"""Concrete implementations of TextSplitterPlugins."""
from .character_recursive import CharacterRecursiveTextSplitter

# Add other splitters like TokenTextSplitter, MarkdownTextSplitter etc.

__all__ = ["CharacterRecursiveTextSplitter"]
