### src/genie_tooling/llm_providers/abc.py
# src/genie_tooling/llm_providers/abc.py
import logging
from typing import (
    Any,
    AsyncIterable,
    Dict,
    List,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

from genie_tooling.core.types import Plugin

from .types import (
    ChatMessage,
    LLMChatChunk,  # Added for streaming
    LLMChatResponse,
    LLMCompletionChunk,  # Added for streaming
    LLMCompletionResponse,
)

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
        await super().setup(config)
        logger.debug(f"LLMProviderPlugin '{getattr(self, 'plugin_id', 'UnknownPluginID')}': Base setup logic (if any) completed.")


    async def generate(
        self, prompt: str, stream: bool = False, **kwargs: Any
    ) -> Union[LLMCompletionResponse, AsyncIterable[LLMCompletionChunk]]:
        logger.error(f"LLMProviderPlugin '{getattr(self, 'plugin_id', 'UnknownPluginID')}' generate method not implemented.")
        raise NotImplementedError(f"LLMProviderPlugin '{getattr(self, 'plugin_id', 'UnknownPluginID')}' does not implement 'generate'.")

    async def chat(
        self, messages: List[ChatMessage], stream: bool = False, **kwargs: Any
    ) -> Union[LLMChatResponse, AsyncIterable[LLMChatChunk]]:
        logger.error(f"LLMProviderPlugin '{getattr(self, 'plugin_id', 'UnknownPluginID')}' chat method not implemented.")
        raise NotImplementedError(f"LLMProviderPlugin '{getattr(self, 'plugin_id', 'UnknownPluginID')}' does not implement 'chat'.")

    async def get_model_info(self) -> Dict[str, Any]:
        logger.debug(f"LLMProviderPlugin '{getattr(self, 'plugin_id', 'UnknownPluginID')}' get_model_info method not implemented. Returning empty dict.")
        return {}
###<END-OF-FILE>###
