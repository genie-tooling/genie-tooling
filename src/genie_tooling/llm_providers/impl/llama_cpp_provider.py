import json
import logging
from typing import Any, AsyncIterable, Dict, List, Optional, Union, cast

import httpx
from pydantic import BaseModel

from genie_tooling.llm_providers.abc import LLMProviderPlugin
from genie_tooling.llm_providers.types import (
    ChatMessage,
    LLMChatChunk,
    LLMChatChunkDeltaMessage,
    LLMChatResponse,
    LLMCompletionChunk,
    LLMCompletionResponse,
    LLMUsageInfo,
)
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.utils.gbnf import (
    create_dynamic_models_from_dictionaries,
    generate_gbnf_grammar_from_pydantic_models,  # Changed to the more direct function
)

logger = logging.getLogger(__name__)

class LlamaCppLLMProviderPlugin(LLMProviderPlugin):
    plugin_id: str = "llama_cpp_llm_provider_v1"
    description: str = "LLM provider for llama.cpp server endpoints (/completion, /chat/completions)."

    _http_client: Optional[httpx.AsyncClient] = None
    _base_url: str
    _default_model_alias: Optional[str] = None # llama.cpp server might not use model names in requests
    _request_timeout: float = 120.0
    _api_key_name: Optional[str] = None # If llama.cpp server is secured
    _key_provider: Optional[KeyProvider] = None

    async def setup(self, config: Optional[Dict[str, Any]]) -> None:
        await super().setup(config)
        cfg = config or {}
        self._base_url = cfg.get("base_url", "http://localhost:8080").rstrip("/")
        self._default_model_alias = cfg.get("model_name") # Often not used by llama.cpp server directly
        self._request_timeout = float(cfg.get("request_timeout_seconds", self._request_timeout))
        self._api_key_name = cfg.get("api_key_name") # e.g., "LLAMA_CPP_API_KEY"
        self._key_provider = cfg.get("key_provider")

        headers = {}
        if self._api_key_name and self._key_provider:
            api_key = await self._key_provider.get_key(self._api_key_name)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            else:
                logger.warning(f"{self.plugin_id}: API key '{self._api_key_name}' configured but not found via KeyProvider.")

        self._http_client = httpx.AsyncClient(timeout=self._request_timeout, headers=headers)
        logger.info(
            f"{self.plugin_id}: Initialized. Base URL: {self._base_url}, "
            f"Default Model Alias (if used): {self._default_model_alias}, "
            f"API Key Name: {self._api_key_name or 'None'}"
        )

    async def _make_request(
        self, endpoint: str, payload: Dict[str, Any], stream: bool = False
    ) -> Union[Dict[str, Any], AsyncIterable[Dict[str, Any]]]:
        if not self._http_client:
            raise RuntimeError(f"{self.plugin_id}: HTTP client not initialized.")
        url = f"{self._base_url}{endpoint}"
        # llama.cpp /completion endpoint uses 'stream' directly in payload
        # /chat/completions uses 'stream' in payload too.
        payload["stream"] = stream

        try:
            response = await self._http_client.post(url, json=payload)
            response.raise_for_status()

            if stream:
                async def stream_generator():
                    try:
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                line_content = line[len("data: "):]
                                if line_content == "[DONE]": # llama.cpp specific stream end
                                    break
                                try:
                                    yield json.loads(line_content)
                                except json.JSONDecodeError:
                                    logger.error(f"{self.plugin_id}: Failed to decode JSON stream chunk: {line_content}")
                            elif line.strip(): # Handle non-event-stream lines if any
                                logger.debug(f"{self.plugin_id}: Received non-data line in stream: {line}")
                    finally:
                        await response.aclose()
                return stream_generator()
            else:
                try:
                    return response.json()
                finally:
                    await response.aclose()
        except httpx.HTTPStatusError as e:
            err_body = e.response.text
            logger.error(f"{self.plugin_id}: HTTP error calling {url}: {e.response.status_code} - {err_body}", exc_info=True)
            raise RuntimeError(f"llama.cpp API error: {e.response.status_code} - {err_body}") from e
        except httpx.RequestError as e:
            logger.error(f"{self.plugin_id}: Request error calling {url}: {e}", exc_info=True)
            raise RuntimeError(f"llama.cpp request failed: {e}") from e
        except json.JSONDecodeError as e:
            logger.error(f"{self.plugin_id}: Failed to decode JSON response from {url}: {e}", exc_info=True)
            raise RuntimeError(f"llama.cpp response JSON decode error: {e}") from e

    async def generate(
        self, prompt: str, stream: bool = False, **kwargs: Any
    ) -> Union[LLMCompletionResponse, AsyncIterable[LLMCompletionChunk]]:
        payload: Dict[str, Any] = {"prompt": prompt, "n_predict": kwargs.get("max_tokens", -1)} # -1 for infinite
        if "temperature" in kwargs:
             payload["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
             payload["top_k"] = kwargs["top_k"]
        if "stop_sequences" in kwargs and kwargs["stop_sequences"]:
             payload["stop"] = kwargs["stop_sequences"]

        output_schema = kwargs.get("output_schema")
        gbnf_grammar: Optional[str] = None
        if output_schema:
            try:
                if isinstance(output_schema, type) and issubclass(output_schema, BaseModel):
                    gbnf_grammar = generate_gbnf_grammar_from_pydantic_models([output_schema])
                elif isinstance(output_schema, dict): # Assume JSON schema dict
                    dynamic_models = create_dynamic_models_from_dictionaries([output_schema])
                    if dynamic_models:
                        gbnf_grammar = generate_gbnf_grammar_from_pydantic_models(dynamic_models)
                if gbnf_grammar:
                    payload["grammar"] = gbnf_grammar
                    logger.info(f"{self.plugin_id}: Using GBNF grammar for structured output via generate().")
            except Exception as e_gbnf:
                logger.error(f"{self.plugin_id}: Failed to generate GBNF grammar from output_schema: {e_gbnf}", exc_info=True)
                # Decide: fail hard or proceed without grammar? For now, proceed without.
                # raise ValueError(f"Failed to process output_schema for GBNF: {e_gbnf}") from e_gbnf

        if stream:
            async def stream_generate_chunks() -> AsyncIterable[LLMCompletionChunk]:
                response_stream = await self._make_request("/completion", payload, stream=True)
                if not isinstance(response_stream, AsyncIterable):
                    raise RuntimeError("Expected stream from _make_request for generate")

                final_usage: Optional[LLMUsageInfo] = None

                async for chunk_data in response_stream:
                    if not isinstance(chunk_data, dict):
                        continue
                    text_delta = chunk_data.get("content", "")
                    is_done = chunk_data.get("stop", False)
                    finish_reason_str: Optional[str] = None
                    if is_done:
                        if chunk_data.get("stopped_eos"):
                             finish_reason_str = "stop"
                        elif chunk_data.get("stopped_word"):
                            finish_reason_str = "stop_sequence"
                        elif chunk_data.get("stopped_limit"):
                            finish_reason_str = "length"
                        else:
                             finish_reason_str = "unknown_stop"

                    current_chunk: LLMCompletionChunk = {"text_delta": text_delta, "raw_chunk": chunk_data}
                    if finish_reason_str:
                        current_chunk["finish_reason"] = finish_reason_str
                        final_usage = {
                            "prompt_tokens": chunk_data.get("tokens_evaluated"),
                            "completion_tokens": chunk_data.get("tokens_predicted"),
                            "total_tokens": (chunk_data.get("tokens_evaluated",0) or 0) + (chunk_data.get("tokens_predicted",0) or 0)
                        }
                        current_chunk["usage_delta"] = final_usage
                    yield current_chunk
            return stream_generate_chunks()
        else:
            response_data = await self._make_request("/completion", payload, stream=False)
            if not isinstance(response_data, dict):
                 raise RuntimeError("Expected dict from _make_request for non-streaming generate")

            usage_info: LLMUsageInfo = {
                "prompt_tokens": response_data.get("tokens_evaluated"),
                "completion_tokens": response_data.get("tokens_predicted"),
            }
            if usage_info.get("prompt_tokens") is not None and usage_info.get("completion_tokens") is not None:
                usage_info["total_tokens"] = (usage_info["prompt_tokens"] or 0) + (usage_info["completion_tokens"] or 0)

            finish_reason = "unknown"
            if response_data.get("stopped_eos"):
                 finish_reason = "stop"
            elif response_data.get("stopped_word"):
                finish_reason = "stop_sequence"
            elif response_data.get("stopped_limit"):
                finish_reason = "length"

            return {
                "text": response_data.get("content", ""),
                "finish_reason": finish_reason,
                "usage": usage_info,
                "raw_response": response_data,
            }

    async def chat(
        self, messages: List[ChatMessage], stream: bool = False, **kwargs: Any
    ) -> Union[LLMChatResponse, AsyncIterable[LLMChatChunk]]:
        # llama.cpp /chat/completions endpoint is OpenAI compatible.
        # GBNF grammar support here depends on the llama.cpp server version and if it
        # mirrors OpenAI's `grammar` or similar parameter for chat.
        # For now, we'll add it if `output_schema` is provided, assuming it might work.

        payload: Dict[str, Any] = {"messages": messages}
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
             payload["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            payload["top_k"] = kwargs["top_k"]
        # `max_tokens` for OpenAI chat is `max_tokens`, llama.cpp /chat/completions might use `n_predict` or map it.
        # Let's assume it maps `max_tokens` if present in kwargs.
        if "max_tokens" in kwargs:
             payload["max_tokens"] = kwargs["max_tokens"]
        if "stop_sequences" in kwargs and kwargs["stop_sequences"]:
             payload["stop"] = kwargs["stop_sequences"]
        if "tools" in kwargs:
            payload["tools"] = kwargs["tools"]
        if "tool_choice" in kwargs:
             payload["tool_choice"] = kwargs["tool_choice"]

        output_schema = kwargs.get("output_schema")
        gbnf_grammar: Optional[str] = None
        if output_schema:
            try:
                if isinstance(output_schema, type) and issubclass(output_schema, BaseModel):
                    gbnf_grammar = generate_gbnf_grammar_from_pydantic_models([output_schema])
                elif isinstance(output_schema, dict):
                    dynamic_models = create_dynamic_models_from_dictionaries([output_schema])
                    if dynamic_models:
                        gbnf_grammar = generate_gbnf_grammar_from_pydantic_models(dynamic_models)
                if gbnf_grammar:
                    # Common way to pass grammar to OpenAI-compatible chat endpoints if supported
                    payload["grammar"] = gbnf_grammar
                    logger.info(f"{self.plugin_id}: Using GBNF grammar for structured chat output.")
            except Exception as e_gbnf:
                logger.error(f"{self.plugin_id}: Failed to generate GBNF grammar for chat: {e_gbnf}", exc_info=True)

        if stream:
            async def stream_chat_chunks() -> AsyncIterable[LLMChatChunk]:
                response_stream = await self._make_request("/chat/completions", payload, stream=True)
                if not isinstance(response_stream, AsyncIterable):
                     raise RuntimeError("Expected stream from _make_request for chat")

                async for chunk_data in response_stream:
                    if not isinstance(chunk_data, dict):
                         continue
                    choice = chunk_data.get("choices", [{}])[0]
                    delta = choice.get("delta", {})
                    delta_message: LLMChatChunkDeltaMessage = {}
                    if "role" in delta:
                         delta_message["role"] = delta["role"] # type: ignore
                    if "content" in delta:
                        delta_message["content"] = delta["content"]
                    if "tool_calls" in delta:
                        delta_message["tool_calls"] = delta["tool_calls"] # type: ignore

                    current_chunk: LLMChatChunk = {"message_delta": delta_message, "raw_chunk": chunk_data}
                    if choice.get("finish_reason"):
                        current_chunk["finish_reason"] = choice.get("finish_reason")
                        # llama.cpp /chat/completions might include usage in the final chunk
                        usage_data = chunk_data.get("usage")
                        if usage_data:
                            current_chunk["usage_delta"] = {
                                "prompt_tokens": usage_data.get("prompt_tokens"),
                                "completion_tokens": usage_data.get("completion_tokens"),
                                "total_tokens": usage_data.get("total_tokens"),
                            }
                    yield current_chunk
            return stream_chat_chunks()
        else:
            response_data = await self._make_request("/chat/completions", payload, stream=False)
            if not isinstance(response_data, dict):
                 raise RuntimeError("Expected dict from _make_request for non-streaming chat")

            choice = response_data.get("choices", [{}])[0]
            assistant_message_raw = choice.get("message", {"role": "assistant", "content": ""})
            assistant_message: ChatMessage = {
                "role": cast(Any, assistant_message_raw.get("role", "assistant")),
                "content": assistant_message_raw.get("content"),
            }
            if "tool_calls" in assistant_message_raw:
                assistant_message["tool_calls"] = assistant_message_raw["tool_calls"]

            usage_data = response_data.get("usage")
            usage_info: Optional[LLMUsageInfo] = None
            if usage_data:
                usage_info = {
                    "prompt_tokens": usage_data.get("prompt_tokens"),
                    "completion_tokens": usage_data.get("completion_tokens"),
                    "total_tokens": usage_data.get("total_tokens"),
                }
            return {
                "message": assistant_message,
                "finish_reason": choice.get("finish_reason"),
                "usage": usage_info,
                "raw_response": response_data,
            }

    async def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "llama.cpp",
            "base_url": self._base_url,
            "configured_model_alias": self._default_model_alias or "N/A (server default)",
            "notes": "llama.cpp server usually serves a single pre-loaded model. Specific model details depend on server startup configuration. Supports GBNF grammar for structured output via `output_schema` in `generate()` and potentially `chat()`.",
        }

    async def teardown(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        self._key_provider = None
        logger.info(f"{self.plugin_id}: Teardown complete.")
        await super().teardown()
