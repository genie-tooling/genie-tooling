"""Abstract Base Classes/Protocols for Embedding Generator Plugins."""
import logging
from typing import (
    Any,
    AsyncIterable,
    Dict,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

from genie_tooling.core.types import (
    Chunk,
    EmbeddingVector,
    Plugin,
)

logger = logging.getLogger(__name__)

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
