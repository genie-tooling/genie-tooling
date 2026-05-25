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
    LLMChatChunk,
    LLMChatResponse,
    LLMCompletionChunk,
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
        """Generate a chat completion.

        Canonical kwargs across providers (M4 / M5):
          * ``response_schema``: an optional ``type[pydantic.BaseModel]``.
            When supplied, providers that support native structured outputs
            (OpenAI ``response_format``, Anthropic tool-use round-trip,
            Gemini ``response_schema``) will guarantee the response shape
            matches the model and surface the JSON as the message content.
            Providers without native support (Ollama, llama.cpp) ignore the
            arg; callers should fall back to ``PydanticOutputParserPlugin``.
          * ``tools``: list of OpenAI-function-spec tool definitions.
          * ``tool_choice``: provider-specific tool selection hint.
          * ``temperature``, ``top_p``, ``max_tokens``, ``stop``: standard
            sampling controls.
          * ``ChatMessage.content`` may be a string OR a list of content
            blocks (per M5) — providers that support vision (OpenAI,
            Anthropic, Gemini) handle the list shape; others should treat
            non-string content as the concatenated text.
        """
        logger.error(f"LLMProviderPlugin '{getattr(self, 'plugin_id', 'UnknownPluginID')}' chat method not implemented.")
        raise NotImplementedError(f"LLMProviderPlugin '{getattr(self, 'plugin_id', 'UnknownPluginID')}' does not implement 'chat'.")

    async def get_model_info(self) -> Dict[str, Any]:
        logger.debug(f"LLMProviderPlugin '{getattr(self, 'plugin_id', 'UnknownPluginID')}' get_model_info method not implemented. Returning empty dict.")
        return {}
###<END-OF-FILE>###
