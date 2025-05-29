### src/genie_tooling/llm_providers/impl/openai_provider.py
import logging
from typing import Any, Dict, List, Optional, cast

from genie_tooling.llm_providers.abc import LLMProviderPlugin
from genie_tooling.llm_providers.types import (
    ChatMessage,
    LLMChatResponse,
    LLMCompletionResponse,
    LLMUsageInfo,
)
from genie_tooling.llm_providers.types import (
    ToolCall as GenieToolCall,
)
from genie_tooling.security.key_provider import KeyProvider

logger = logging.getLogger(__name__)

# Attempt to import openai, make it optional
try:
    from openai import APIError, AsyncOpenAI, RateLimitError  # type: ignore
    from openai.types.chat import (
        ChatCompletionMessage as OpenAIChatMessage,  # type: ignore
    )
    from openai.types.chat import (
        ChatCompletionMessageToolCall as OpenAIToolCall,  # type: ignore
    )
    from openai.types.chat.chat_completion import Choice as OpenAIChoice  # type: ignore
    from openai.types.completion_usage import (
        CompletionUsage as OpenAIUsage,  # type: ignore
    )

except ImportError:
    AsyncOpenAI = None # type: ignore
    APIError = Exception # type: ignore
    RateLimitError = Exception # type: ignore
    OpenAIChatMessage = Dict # type: ignore
    OpenAIToolCall = Dict # type: ignore
    OpenAIChoice = Dict # type: ignore
    OpenAIUsage = Dict # type: ignore
    logger.warning(
        "OpenAILLMProviderPlugin: 'openai' library not installed or version < 1.0. "
        "This plugin will not be functional. Please install it: pip install openai>=1.0"
    )

class OpenAILLMProviderPlugin(LLMProviderPlugin):
    plugin_id: str = "openai_llm_provider_v1"
    description: str = "LLM provider for OpenAI models (GPT-3.5, GPT-4, etc.) using the openai library."

    _client: Optional[AsyncOpenAI] = None
    _model_name: str
    _api_key_name: str = "OPENAI_API_KEY"
    _key_provider: Optional[KeyProvider] = None

    async def setup(self, config: Optional[Dict[str, Any]]) -> None:
        await super().setup(config)
        if not AsyncOpenAI:
            logger.error(f"{self.plugin_id}: 'openai' library (>=1.0) is not available. Cannot proceed.")
            return

        cfg = config or {}
        self._key_provider = cfg.get("key_provider")
        if not self._key_provider or not isinstance(self._key_provider, KeyProvider):
            logger.error(f"{self.plugin_id}: KeyProvider not found in config or is invalid. Cannot fetch API key.")
            return

        self._api_key_name = cfg.get("api_key_name", self._api_key_name)
        self._model_name = cfg.get("model_name", "gpt-3.5-turbo")

        api_key = await self._key_provider.get_key(self._api_key_name)
        if not api_key:
            logger.error(f"{self.plugin_id}: API key '{self._api_key_name}' not found via KeyProvider.")
            return

        try:
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=cfg.get("openai_api_base"), # For Azure OpenAI or proxies
                organization=cfg.get("openai_organization"),
                # Default retries in openai client v1.x are 2. Can be configured if needed.
            )
            logger.info(f"{self.plugin_id}: Initialized OpenAI client for model '{self._model_name}'.")
        except Exception as e:
            logger.error(f"{self.plugin_id}: Failed to initialize OpenAI client: {e}", exc_info=True)
            self._client = None

    def _parse_openai_response_message(self, message: OpenAIChatMessage) -> ChatMessage:
        """Converts an OpenAI ChatCompletionMessage to our internal ChatMessage format."""
        genie_tool_calls: Optional[List[GenieToolCall]] = None
        if message.tool_calls:
            genie_tool_calls = []
            for tc in message.tool_calls:
                if tc.type == "function":
                    genie_tool_calls.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                    })
        return {
            "role": cast(Any, message.role), # OpenAI role aligns with ours
            "content": message.content,
            "tool_calls": genie_tool_calls,
            # 'name' and 'tool_call_id' are for input messages, not typically in assistant response message like this
        }

    def _parse_openai_usage(self, usage_data: Optional[OpenAIUsage]) -> Optional[LLMUsageInfo]:
        if not usage_data:
            return None
        return {
            "prompt_tokens": usage_data.prompt_tokens,
            "completion_tokens": usage_data.completion_tokens,
            "total_tokens": usage_data.total_tokens,
        }


    async def generate(self, prompt: str, **kwargs: Any) -> LLMCompletionResponse:
        if not self._client:
            raise RuntimeError(f"{self.plugin_id}: Client not initialized.")

        # For `generate` (completion), we use the chat completion endpoint with a user message
        # as the older `/completions` endpoint is legacy for gpt-3.5-turbo and newer models.
        messages: List[ChatMessage] = [{"role": "user", "content": prompt}]
        model_to_use = kwargs.pop("model", self._model_name)
        common_params = {
            "temperature": kwargs.get("temperature"), "top_p": kwargs.get("top_p"),
            "max_tokens": kwargs.get("max_tokens"), "stop": kwargs.get("stop_sequences"),
            "presence_penalty": kwargs.get("presence_penalty"), "frequency_penalty": kwargs.get("frequency_penalty"),
        }
        # Filter out None values
        request_params = {k: v for k, v in common_params.items() if v is not None}

        try:
            response = await self._client.chat.completions.create(
                model=model_to_use,
                messages=cast(Any, messages), # Cast because OpenAI's type hint is specific
                **request_params
            )
            text_content = ""
            finish_reason = None
            if response.choices:
                text_content = response.choices[0].message.content or ""
                finish_reason = response.choices[0].finish_reason
            usage = self._parse_openai_usage(response.usage)

            return {
                "text": text_content,
                "finish_reason": finish_reason,
                "usage": usage,
                "raw_response": response.model_dump(exclude_none=True),
            }
        except APIError as e:
            logger.error(f"{self.plugin_id} OpenAI API Error during generate: {e.status_code} - {e.message}", exc_info=True)
            raise RuntimeError(f"OpenAI API Error: {e.status_code} - {e.message}") from e
        except Exception as e:
            logger.error(f"{self.plugin_id} Unexpected error during generate: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error in OpenAI generate: {str(e)}") from e

    async def chat(self, messages: List[ChatMessage], **kwargs: Any) -> LLMChatResponse:
        if not self._client:
            raise RuntimeError(f"{self.plugin_id}: Client not initialized.")

        model_to_use = kwargs.pop("model", self._model_name)
        # Prepare tools for OpenAI format if provided
        tools_for_api = kwargs.get("tools") # Expects OpenAI's tool format
        tool_choice_for_api = kwargs.get("tool_choice") # Expects OpenAI's tool_choice format

        common_params = {
            "temperature": kwargs.get("temperature"), "top_p": kwargs.get("top_p"),
            "max_tokens": kwargs.get("max_tokens"), "stop": kwargs.get("stop_sequences"),
            "presence_penalty": kwargs.get("presence_penalty"), "frequency_penalty": kwargs.get("frequency_penalty"),
        }
        request_params = {k: v for k, v in common_params.items() if v is not None}
        if tools_for_api: request_params["tools"] = tools_for_api
        if tool_choice_for_api: request_params["tool_choice"] = tool_choice_for_api


        try:
            response = await self._client.chat.completions.create(
                model=model_to_use,
                messages=cast(Any, messages),
                **request_params
            )

            genie_message: ChatMessage = {"role": "assistant", "content": None} # Default
            finish_reason = None
            if response.choices:
                openai_msg = response.choices[0].message
                genie_message = self._parse_openai_response_message(openai_msg)
                finish_reason = response.choices[0].finish_reason

            usage = self._parse_openai_usage(response.usage)

            return {
                "message": genie_message,
                "finish_reason": finish_reason,
                "usage": usage,
                "raw_response": response.model_dump(exclude_none=True),
            }
        except APIError as e:
            logger.error(f"{self.plugin_id} OpenAI API Error during chat: {e.status_code} - {e.message}", exc_info=True)
            raise RuntimeError(f"OpenAI API Error: {e.status_code} - {e.message}") from e
        except Exception as e:
            logger.error(f"{self.plugin_id} Unexpected error during chat: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error in OpenAI chat: {str(e)}") from e

    async def get_model_info(self) -> Dict[str, Any]:
        # OpenAI API doesn't have a simple "get model info" endpoint like some others.
        # Typically, one checks capabilities based on model name prefixes or documentation.
        # We can list models if an admin key or specific permissions are available, but
        # for a general purpose key, just returning configured info is safer.
        return {
            "provider": "OpenAI",
            "configured_model_name": self._model_name,
            "notes": "Detailed model info (token limits, etc.) typically found in OpenAI documentation for the specified model.",
        }

    async def teardown(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
        self._key_provider = None
        logger.info(f"{self.plugin_id}: Teardown complete.")
        await super().teardown()
###<END-OF-FILE>###
