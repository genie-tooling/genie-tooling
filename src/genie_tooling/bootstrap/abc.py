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

    This is the primary mechanism for creating first-class extensions to the
    framework, allowing new interfaces to be attached directly to the `genie`
    object (e.g., `genie.my_extension`).
    """
    async def bootstrap(self, genie: "Genie") -> None:
        """
        The main execution method for the bootstrap plugin.

        This method is called at the end of `Genie.create()`. It provides a
        powerful hook for one-time setup tasks, such as database migrations,
        pre-caching data, or initializing and attaching new extension interfaces
        to the `genie` object.

        Args:
            genie: The fully initialized Genie instance. This provides access
                   to all framework components (e.g., `genie.llm`, `genie.rag`),
                   shared components (`genie.get_default_embedder()`), and the
                   internal PluginManager (`genie._plugin_manager`) for advanced use cases.
        """
        logger.warning(f"BootstrapPlugin '{self.plugin_id}' bootstrap method not fully implemented.")
        pass
