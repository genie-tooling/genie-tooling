"""Core shared types and protocols for the middleware."""
import logging
from typing import (
    Any,
    AsyncIterable,
    Dict,
    List,
    Optional,
    Protocol,
    TypedDict,
    TypeVar,
    runtime_checkable,
)

logger = logging.getLogger(__name__) # Use specific logger

@runtime_checkable
class Plugin(Protocol):
    """Base protocol for all plugins."""
    @property
    def plugin_id(self) -> str:
        ...

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Optional asynchronous setup method for plugins. Called after instantiation."""
        pass

    async def teardown(self) -> None:
        """Optional asynchronous teardown method for plugins. Called before application shutdown."""
        pass

# Define PluginType directly, bound to the Plugin protocol defined above.
PluginType = TypeVar("PluginType", bound=Plugin) # <<< CHANGED: Defined directly as PluginType

# --- RAG Specific Types ---
class Document(Protocol):
    """Represents a loaded document before splitting."""
    content: str
    metadata: Dict[str, Any] = {}
    id: Optional[str] = None

class Chunk(Protocol):
    """Represents a chunk of a document after splitting."""
    content: str
    metadata: Dict[str, Any] = {}
    id: Optional[str] = None

EmbeddingVector = List[float]

class RetrievedChunk(Chunk, Protocol):
    """Represents a chunk retrieved from a vector store, with a relevance score."""
    score: float
    rank: Optional[int] = None

# --- General Utility Types ---
T = TypeVar("T") # General TypeVar, not bound to Plugin
AsyncStreamable = AsyncIterable[T]

class StructuredError(TypedDict, total=False):
    """Standardized structure for reporting errors, especially to LLMs."""
    type: str
    message: str
    details: Optional[Dict[str, Any]]
    suggestion: Optional[str]
