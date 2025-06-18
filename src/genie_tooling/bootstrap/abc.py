# src/genie_tooling/bootstrap/abc.py
"""Abstract Base Class/Protocol for Bootstrap Plugins."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from genie_tooling.core.types import Plugin

if TYPE_CHECKING:
    from genie_tooling.genie import Genie

logger = logging.getLogger(__name__)

@runtime_checkable
class BootstrapPlugin(Plugin, Protocol):
    """
    Protocol for a plugin that runs once after the Genie framework
    is fully initialized but before it is returned to the user.
    Ideal for one-time setup tasks like database migrations,
    pre-caching, or initial RAG indexing.
    """
    async def bootstrap(self, genie: "Genie") -> None:
        """
        The main execution method for the bootstrap plugin.

        Args:
            genie: The fully initialized Genie instance, providing access
                   to all framework components (e.g., genie.rag, genie.llm).
        """
        logger.warning(f"BootstrapPlugin '{self.plugin_id}' bootstrap method not fully implemented.")
        pass