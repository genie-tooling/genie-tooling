### src/genie_tooling/prompts/conversation/impl/abc.py

"""Abstract Base Class for ConversationStateProvider Plugins."""
import logging
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from genie_tooling.core.types import Plugin
from genie_tooling.llm_providers.types import ChatMessage # For ConversationState

from ..types import ConversationState # CORRECTED IMPORT

logger = logging.getLogger(__name__)

@runtime_checkable
class ConversationStateProviderPlugin(Plugin, Protocol):
    """Protocol for a plugin that stores and retrieves conversation state."""
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