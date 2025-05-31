# src/genie_tooling/prompts/conversation/impl/in_memory_state_provider.py
import asyncio
import logging
from typing import Any, Dict, List, Optional

from genie_tooling.prompts.conversation.impl.abc import ConversationStateProviderPlugin
from genie_tooling.prompts.conversation.types import ConversationState

logger = logging.getLogger(__name__)

class InMemoryStateProviderPlugin(ConversationStateProviderPlugin):
    plugin_id: str = "in_memory_conversation_state_v1"
    description: str = "Stores conversation state in an in-memory dictionary."

    _store: Dict[str, ConversationState]
    _lock: asyncio.Lock

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._store = {}
        self._lock = asyncio.Lock()
        logger.info(f"{self.plugin_id}: Initialized in-memory conversation store.")

    async def load_state(self, session_id: str) -> Optional[ConversationState]:
        async with self._lock:
            state = self._store.get(session_id)
            if state:
                # Return a copy to prevent external modification of the stored object
                return ConversationState(session_id=state["session_id"], history=list(state["history"]), metadata=dict(state.get("metadata") or {}))
            return None

    async def save_state(self, state: ConversationState) -> None:
        if not state or "session_id" not in state:
            logger.error(f"{self.plugin_id}: Attempted to save invalid state (missing session_id). State: {state}")
            return
        
        session_id = state["session_id"]
        async with self._lock:
            # Store a copy to ensure immutability of the input 'state' object from caller's perspective
            self._store[session_id] = ConversationState(session_id=state["session_id"], history=list(state["history"]), metadata=dict(state.get("metadata") or {}))
        logger.debug(f"{self.plugin_id}: Saved state for session_id '{session_id}'.")

    async def delete_state(self, session_id: str) -> bool:
        async with self._lock:
            if session_id in self._store:
                del self._store[session_id]
                logger.debug(f"{self.plugin_id}: Deleted state for session_id '{session_id}'.")
                return True
            logger.debug(f"{self.plugin_id}: Attempted to delete non-existent state for session_id '{session_id}'.")
            return False

    async def teardown(self) -> None:
        async with self._lock:
            self._store.clear()
        logger.info(f"{self.plugin_id}: Teardown complete, in-memory store cleared.")
