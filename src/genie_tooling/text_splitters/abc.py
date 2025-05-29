"""Abstract Base Classes/Protocols for Text Splitter Plugins."""
import logging
from typing import (
    Any,
    AsyncIterable,
    Dict,
    Optional,
    Protocol,
    runtime_checkable,
)

from genie_tooling.core.types import (
    Chunk,
    Document,  # Make sure this is imported if "Document" is used as a type hint
    Plugin,
)

logger = logging.getLogger(__name__)

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
