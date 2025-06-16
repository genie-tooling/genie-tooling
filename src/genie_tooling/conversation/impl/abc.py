# src/genie_tooling/conversation/impl/abc.py
"""Abstract Base Class for ConversationStateProvider Plugins."""
import logging
from typing import Optional, Protocol, runtime_checkable

from genie_tooling.core.types import Plugin

from ..types import ConversationState

logger = logging.getLogger(__name__)

@runtime_checkable
class ConversationStateProviderPlugin(Plugin, Protocol):
    plugin_id: str
    async def load_state(self, session_id: str) -> Optional[ConversationState]:
        logger.warning(f"ConversationStateProviderPlugin '{self.plugin_id}' load_state not implemented.")
        return None
    async def save_state(self, state: ConversationState) -> None:
        logger.warning(f"ConversationStateProviderPlugin '{self.plugin_id}' save_state not implemented.")
        pass
    async def delete_state(self, session_id: str) -> bool:
        logger.warning(f"ConversationStateProviderPlugin '{self.plugin_id}' delete_state not implemented.")
        return False
