"""Core framework components like the PluginManager and base types."""
from .plugin_manager import PluginManager
from .types import (
    AsyncStreamable,
    Chunk,
    Document,
    EmbeddingVector,
    Plugin,  # <<< CHANGED: Exporting the direct PluginType
    PluginType,
    RetrievedChunk,
    StructuredError,
)

__all__ = [
    "PluginManager", "Plugin", "PluginType", # <<< CHANGED
    "Document", "Chunk", "EmbeddingVector",
    "RetrievedChunk", "StructuredError", "AsyncStreamable"
]
