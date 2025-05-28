
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
    plugin_id: str # Inherited from Plugin, unique identifier for this provider plugin
    description: str # Human-readable description of the LLM provider this plugin supports

    async def setup(self, config: Optional[Dict[str, Any]], key_provider: KeyProvider) -> None:
        """
        Initializes the LLM provider client and any other necessary setup.
        Args:
            config: Provider-specific configuration dictionary. May include model names,
                    API base URLs, or other provider-specific settings.
            key_provider: An instance of KeyProvider to fetch necessary API keys.
        """
        # Default implementation can be a no-op if setup is handled by specific plugin.
        # Call super().setup() if Plugin's default setup needs to run.
        await super().setup(config)
        logger.debug(f"LLMProviderPlugin '{self.plugin_id}': Default setup called.")
        # Implementations should store key_provider and use it to fetch keys.

    async def generate(self, prompt: str, **kwargs: Any) -> LLMCompletionResponse:
        """
        Generates a text completion for the given prompt.
        Args:
            prompt: The input prompt string.
            **kwargs: Provider-specific parameters (e.g., max_tokens, temperature, model).
        Returns:
            An LLMCompletionResponse dictionary.
        """
        # This is a protocol; implementations must provide this.
        # Example of a default that indicates not implemented:
        logger.error(f"LLMProviderPlugin '{self.plugin_id}' generate method not implemented.")
        raise NotImplementedError(f"LLMProviderPlugin '{self.plugin_id}' does not implement 'generate'.")

    async def chat(self, messages: List[ChatMessage], **kwargs: Any) -> LLMChatResponse:
        """
        Generates a chat completion based on a list of messages.
        Args:
            messages: A list of ChatMessage dictionaries representing the conversation history.
            **kwargs: Provider-specific parameters (e.g., model, temperature, tool_choice).
        Returns:
            An LLMChatResponse dictionary.
        """
        # This is a protocol; implementations must provide this.
        logger.error(f"LLMProviderPlugin '{self.plugin_id}' chat method not implemented.")
        raise NotImplementedError(f"LLMProviderPlugin '{self.plugin_id}' does not implement 'chat'.")

    async def get_model_info(self) -> Dict[str, Any]:
        """
        (Optional) Retrieves information about the underlying model(s) this provider uses.
        Could include details like context window size, supported features, etc.
        Returns:
            A dictionary containing model information.
        """
        logger.debug(f"LLMProviderPlugin '{self.plugin_id}' get_model_info method not implemented. Returning empty dict.")
        return {}

    # teardown is inherited from Plugin protocol.
    # Implementations should override it if they have resources to release.
    # async def teardown(self) -> None:
    #     await super().teardown()
    #     logger.debug(f"LLMProviderPlugin '{self.plugin_id}': Default teardown called.")