### src/genie_tooling/llm_providers/abc.py
# src/genie_tooling/llm_providers/abc.py
import logging
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from genie_tooling.core.types import Plugin
from genie_tooling.security.key_provider import KeyProvider # Still needed for type hint in docstring

from .types import ChatMessage, LLMChatResponse, LLMCompletionResponse

logger = logging.getLogger(__name__)

@runtime_checkable
class LLMProviderPlugin(Plugin, Protocol):
    """
    Protocol for a plugin that interacts with a Large Language Model provider.
    """
    plugin_id: str
    description: str

    async def setup(self, config: Optional[Dict[str, Any]]) -> None:
        """
        Initializes the LLM provider.
        The 'config' dictionary is expected to contain 'key_provider: KeyProvider'
        if the specific LLM provider implementation requires API keys.
        """
        # Implementations should extract 'key_provider' from config if needed.
        # Example:
        # self._key_provider = config.get("key_provider")
        # if not isinstance(self._key_provider, KeyProvider):
        #     logger.error(f"{getattr(self, 'plugin_id', 'Unknown')}: KeyProvider not found in config or invalid type.")
        #     # Handle error appropriately, perhaps by raising an exception or setting a state.
        await super().setup(config) # Call Plugin's default setup
        logger.debug(f"LLMProviderPlugin '{getattr(self, 'plugin_id', 'UnknownPluginID')}': Base setup logic (if any) completed.")


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
