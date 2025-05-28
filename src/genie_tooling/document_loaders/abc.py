"""Abstract Base Classes/Protocols for Document Loader Plugins."""
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
    Document,
    Plugin,
)

logger = logging.getLogger(__name__)

@runtime_checkable
class DocumentLoaderPlugin(Plugin, Protocol):
    """Loads documents from a source into an async stream of Document objects."""
    # plugin_id: str (from Plugin)
    async def load(self, source_uri: str, config: Optional[Dict[str, Any]] = None) -> AsyncIterable[Document]:
        """
        Loads documents from the given source URI.
        Args:
            source_uri: The URI of the data source (e.g., file path, URL, database connection string).
            config: Loader-specific configuration dictionary.
        Yields:
            Document objects.
        """
        # Example of how to make an async generator that does nothing if not implemented:
        logger.warning(f"DocumentLoaderPlugin '{self.plugin_id}' load method not fully implemented.")
        if False: # pylint: disable=false-condition
            yield # type: ignore
        return
