### src/genie_tooling/llm_providers/abc.py
# src/genie_tooling/llm_providers/abc.py
import logging
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from genie_tooling.core.types import Plugin
from genie_tooling.security.key_provider import KeyProvider

from .types import ChatMessage, LLMChatResponse, LLMCompletionResponse

logger = logging.getLogger(__name__)

@runtime_checkable
class LLMProviderPlugin(Plugin, Protocol):
    """
    Protocol for a plugin that interacts with a Large Language Model provider.
    """
    plugin_id: str
    description: str

    async def setup(self, config: Optional[Dict[str, Any]], key_provider: KeyProvider) -> None:
        """
        Initializes the LLM provider.
        The base Plugin.setup is a pass, so calling super().setup(config) from here
        is mostly for form, unless Plugin.setup gains functionality.
        Concrete implementations should call their own super().setup(config, key_provider)
        if they inherit from LLMProviderPlugin and override setup.
        """
        # await super().setup(config) # Plugin.setup only takes config and is a pass.
                                    # Let concrete implementations handle their super calls.
        logger.debug(f"LLMProviderPlugin '{getattr(self, 'plugin_id', 'UnknownPluginID')}': Base setup logic (if any) would run here.")
        # Implementations should store key_provider and use it to fetch keys if needed at this level.

    async def generate(self, prompt: str, **kwargs: Any) -> LLMCompletionResponse:
        logger.error(f"LLMProviderPlugin '{getattr(self, 'plugin_id', 'UnknownPluginID')}' generate method not implemented.")
        raise NotImplementedError(f"LLMProviderPlugin '{getattr(self, 'plugin_id', 'UnknownPluginID')}' does not implement 'generate'.")

    async def chat(self, messages: List[ChatMessage], **kwargs: Any) -> LLMChatResponse:
        logger.error(f"LLMProviderPlugin '{getattr(self, 'plugin_id', 'UnknownPluginID')}' chat method not implemented.")
        raise NotImplementedError(f"LLMProviderPlugin '{getattr(self, 'plugin_id', 'UnknownPluginID')}' does not implement 'chat'.")

    async def get_model_info(self) -> Dict[str, Any]:
        logger.debug(f"LLMProviderPlugin '{getattr(self, 'plugin_id', 'UnknownPluginID')}' get_model_info method not implemented. Returning empty dict.")
        return {}
###<END-OF-FILE>###
