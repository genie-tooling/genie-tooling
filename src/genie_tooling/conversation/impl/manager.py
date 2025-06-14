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
            event_data_with_msg = data.copy()
            if "message" not in event_data_with_msg:
                if "error" in data: event_data_with_msg["message"] = str(data["error"])
                else: event_data_with_msg["message"] = event_name.split(".")[-1].replace("_", " ").capitalize()
            final_event_name = event_name
            if not event_name.startswith("log.") and level in ["debug", "info", "warning", "error", "critical"]:
                 final_event_name = f"log.{level}"
            await self._tracing_manager.trace_event(final_event_name, event_data_with_msg, "ConversationStateManager")

    async def _get_provider(self, provider_id: Optional[str] = None) -> Optional[ConversationStateProviderPlugin]:
        provider_id_to_use = provider_id or self._default_provider_id
        if not provider_id_to_use:
            await self._trace("log.error", {"message": "No conversation state provider ID specified and no default is set."})
            return None
        provider_config = self._provider_configurations.get(provider_id_to_use, {})
        provider_any = await self._plugin_manager.get_plugin_instance(provider_id_to_use, config=provider_config)
        if not provider_any or not isinstance(provider_any, ConversationStateProviderPlugin):
            await self._trace("log.error", {"message": f"ConversationStateProviderPlugin '{provider_id_to_use}' not found or invalid."})
            return None
        return cast(ConversationStateProviderPlugin, provider_any)

    async def load_state(self, session_id: str, provider_id: Optional[str] = None) -> Optional[ConversationState]:
        provider = await self._get_provider(provider_id)
        if not provider: return None
        return await provider.load_state(session_id)

    async def save_state(self, state: ConversationState, provider_id: Optional[str] = None) -> None:
        provider = await self._get_provider(provider_id)
        if provider: await provider.save_state(state)

    async def add_message(self, session_id: str, message: ChatMessage, provider_id: Optional[str] = None) -> None:
        provider = await self._get_provider(provider_id)
        if not provider: return
        state = await provider.load_state(session_id)
        current_time = asyncio.get_event_loop().time()
        if not state:
            state = ConversationState(session_id=session_id, history=[message], metadata={"created_at": current_time, "last_updated": current_time})
        else:
            state["history"].append(message)
            if state.get("metadata") is None: state["metadata"] = {}
            state["metadata"]["last_updated"] = current_time # type: ignore
        await provider.save_state(state)

    async def delete_state(self, session_id: str, provider_id: Optional[str] = None) -> bool:
        provider = await self._get_provider(provider_id)
        if not provider: return False
        return await provider.delete_state(session_id)

    async def teardown(self) -> None:
        await self._trace("log.info", {"message": "Tearing down..."})
        pass
