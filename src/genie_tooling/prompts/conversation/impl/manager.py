"""ConversationStateManager: Orchestrates ConversationStateProviderPlugin."""
import asyncio
import logging
from typing import Any, Dict, Optional, cast

from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.llm_providers.types import ChatMessage
from genie_tooling.observability.manager import InteractionTracingManager

from ..types import ConversationState
from .abc import ConversationStateProviderPlugin

logger = logging.getLogger(__name__)

class ConversationStateManager:
    def __init__(
        self,
        plugin_manager: PluginManager,
        default_provider_id: Optional[str] = None,
        provider_configurations: Optional[Dict[str, Dict[str, Any]]] = None,
        tracing_manager: Optional[InteractionTracingManager] = None,
    ):
        self._plugin_manager = plugin_manager
        self._default_provider_id = default_provider_id
        self._provider_configurations = provider_configurations or {}
        self._tracing_manager = tracing_manager
        logger.info("ConversationStateManager initialized.")

    async def _trace(self, event_name: str, data: Dict, level: str = "info"):
        if self._tracing_manager:
            await self._tracing_manager.trace_event(f"log.{level}", {"message": event_name, **data}, "ConversationStateManager")

    async def _get_provider(self, provider_id: Optional[str] = None) -> Optional[ConversationStateProviderPlugin]:
        provider_id_to_use = provider_id or self._default_provider_id
        if not provider_id_to_use:
            await self._trace("log.error", {"message": "No conversation state provider ID specified and no default is set."})
            return None

        provider = await self._plugin_manager.get_plugin_instance(
            provider_id_to_use, config=self._provider_configurations.get(provider_id_to_use, {})
        )
        if not provider or not isinstance(provider, ConversationStateProviderPlugin):
            await self._trace("log.error", {"message": f"ConversationStateProviderPlugin '{provider_id_to_use}' not found or invalid."})
            return None
        return cast(ConversationStateProviderPlugin, provider)

    async def load_state(self, session_id: str, provider_id: Optional[str] = None) -> Optional[ConversationState]:
        provider = await self._get_provider(provider_id)
        if not provider:
             return None
        return await provider.load_state(session_id)

    async def save_state(self, state: ConversationState, provider_id: Optional[str] = None) -> None:
        provider = await self._get_provider(provider_id)
        if provider:
            await provider.save_state(state)

    async def add_message(self, session_id: str, message: ChatMessage, provider_id: Optional[str] = None) -> None:
        provider = await self._get_provider(provider_id)
        if not provider:
             return

        state = await provider.load_state(session_id)
        current_time = asyncio.get_event_loop().time()
        if not state:
            state = ConversationState(
                session_id=session_id,
                history=[message],
                metadata={"created_at": current_time, "last_updated": current_time}
            )
        else:
            state["history"].append(message)
            if not state.get("metadata"):
                 state["metadata"] = {}
            state["metadata"]["last_updated"] = current_time


        await provider.save_state(state)

    async def delete_state(self, session_id: str, provider_id: Optional[str] = None) -> bool:
        provider = await self._get_provider(provider_id)
        if not provider:
            return False
        return await provider.delete_state(session_id)

    async def teardown(self) -> None:
        await self._trace("log.info", {"message": "Tearing down..."})
        pass
