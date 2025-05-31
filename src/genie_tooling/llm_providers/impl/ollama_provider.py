import json
import logging
from typing import Any, AsyncIterable, Dict, List, Optional, Union

import httpx

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

logger = logging.getLogger(__name__)

class OllamaLLMProviderPlugin(LLMProviderPlugin):
    plugin_id: str = "ollama_llm_provider_v1"
    description: str = "LLM provider for interacting with a local or remote Ollama instance."

    _http_client: Optional[httpx.AsyncClient] = None
    _base_url: str
    _default_model: str
    _request_timeout: float = 120.0

    async def setup(self, config: Optional[Dict[str, Any]]) -> None:
        await super().setup(config)

        cfg = config or {}
        logger.debug(f"OllamaLLMProviderPlugin ({self.plugin_id}) setup: Received 'config' argument type: {type(config)}, value: {config}")
        logger.debug(f"OllamaLLMProviderPlugin ({self.plugin_id}) setup: 'cfg' (after 'config or {{}}') type: {type(cfg)}, value: {cfg}")

        self._base_url = cfg.get("base_url", "http://localhost:11434").rstrip("/")

        model_name_from_config = cfg.get("model_name")
        self._default_model = model_name_from_config if model_name_from_config is not None else "llama2"

        logger.debug(f"OllamaLLMProviderPlugin ({self.plugin_id}) setup: 'model_name' from cfg: '{model_name_from_config}', resulting self._default_model: '{self._default_model}'")

        self._request_timeout = float(cfg.get("request_timeout_seconds", self._request_timeout))

        self._http_client = httpx.AsyncClient(timeout=self._request_timeout)
        logger.info(f"{self.plugin_id}: Initialized. Base URL: {self._base_url}, Default Model: {self._default_model}")

    async def _make_request(
        self, endpoint: str, payload: Dict[str, Any], stream: bool = False
    ) -> Union[Dict[str, Any], AsyncIterable[Dict[str, Any]]]:
        if not self._http_client:
            raise RuntimeError(f"{self.plugin_id}: HTTP client not initialized.")

        url = f"{self._base_url}{endpoint}"
        payload["stream"] = stream

        try:
            response = await self._http_client.post(url, json=payload)
            response.raise_for_status() # Raise for 4xx/5xx errors

            if stream:
                async def stream_generator():
                    try:
                        async for line in response.aiter_lines():
                            if line:
                                try:
                                    yield json.loads(line)
                                except json.JSONDecodeError:
                                    logger.error(f"{self.plugin_id}: Failed to decode JSON stream chunk: {line}")
                                    # Optionally yield an error chunk or skip
                    finally:
                        await response.aclose() # Ensure response is closed
                return stream_generator()
            else:
                try:
                    return response.json()
                finally:
                    await response.aclose() # Ensure response is closed

        except httpx.HTTPStatusError as e:
            err_body = ""
            try:
                err_body = e.response.json().get("error", e.response.text)
            except json.JSONDecodeError:
                err_body = e.response.text
            logger.error(f"{self.plugin_id}: HTTP error calling {url}: {e.response.status_code} - {err_body}", exc_info=True)
            raise RuntimeError(f"Ollama API error: {e.response.status_code} - {err_body}") from e
        except httpx.RequestError as e:
            logger.error(f"{self.plugin_id}: Request error calling {url}: {e}", exc_info=True)
            raise RuntimeError(f"Ollama request failed: {e}") from e
        except json.JSONDecodeError as e: # Should only happen for non-streaming if response isn't JSON
            logger.error(f"{self.plugin_id}: Failed to decode JSON response from {url}: {e}", exc_info=True)
            raise RuntimeError(f"Ollama response JSON decode error: {e}") from e


    async def generate(
        self, prompt: str, stream: bool = False, **kwargs: Any
    ) -> Union[LLMCompletionResponse, AsyncIterable[LLMCompletionChunk]]:
        model_name = kwargs.pop("model", self._default_model)
        payload = {
            "model": model_name,
            "prompt": prompt,
            "options": kwargs.get("options", {}),
            **kwargs
        }

        if stream:
            async def stream_generate_chunks() -> AsyncIterable[LLMCompletionChunk]:
                response_stream = await self._make_request("/api/generate", payload, stream=True)
                if not isinstance(response_stream, AsyncIterable): # Should not happen if stream=True
                    raise RuntimeError("Expected stream from _make_request for generate")

                full_text = ""
                final_usage: Optional[LLMUsageInfo] = None
                final_raw_response: Optional[Dict[str, Any]] = None

                async for chunk_data in response_stream:
                    if not isinstance(chunk_data, dict): continue # Skip malformed chunks

                    text_delta = chunk_data.get("response", "")
                    full_text += text_delta
                    chunk_finish_reason = "done" if chunk_data.get("done") else None

                    current_chunk: LLMCompletionChunk = {"text_delta": text_delta, "raw_chunk": chunk_data}
                    if chunk_finish_reason:
                        current_chunk["finish_reason"] = chunk_finish_reason
                        final_usage = {
                            "prompt_tokens": chunk_data.get("prompt_eval_count"),
                            "completion_tokens": chunk_data.get("eval_count"),
                        }
                        if final_usage.get("prompt_tokens") is not None and final_usage.get("completion_tokens") is not None:
                            final_usage["total_tokens"] = final_usage["prompt_tokens"] + final_usage["completion_tokens"] # type: ignore
                        current_chunk["usage_delta"] = final_usage # Send final usage with last chunk
                        final_raw_response = chunk_data

                    yield current_chunk
            return stream_generate_chunks()
        else:
            response_data = await self._make_request("/api/generate", payload, stream=False)
            if not isinstance(response_data, dict): # Should not happen
                 raise RuntimeError("Expected dict from _make_request for non-streaming generate")

            usage_info: LLMUsageInfo = {
                "prompt_tokens": response_data.get("prompt_eval_count"),
                "completion_tokens": response_data.get("eval_count"),
            }
            if usage_info.get("prompt_tokens") is not None and usage_info.get("completion_tokens") is not None:
                usage_info["total_tokens"] = usage_info["prompt_tokens"] + usage_info["completion_tokens"] # type: ignore
            return {
                "text": response_data.get("response", ""),
                "finish_reason": "done" if response_data.get("done") else "unknown",
                "usage": usage_info,
                "raw_response": response_data,
            }

    async def chat(
        self, messages: List[ChatMessage], stream: bool = False, **kwargs: Any
    ) -> Union[LLMChatResponse, AsyncIterable[LLMChatChunk]]:
        model_name = kwargs.pop("model", self._default_model)
        logger.debug(f"OllamaLLMProviderPlugin ({self.plugin_id}) chat: Using model_name: '{model_name}' (derived from kwargs or self._default_model: '{self._default_model}')")
        payload = {
            "model": model_name,
            "messages": messages,
            "options": kwargs.get("options", {}),
            **kwargs
        }

        if stream:
            async def stream_chat_chunks() -> AsyncIterable[LLMChatChunk]:
                response_stream = await self._make_request("/api/chat", payload, stream=True)
                if not isinstance(response_stream, AsyncIterable):
                     raise RuntimeError("Expected stream from _make_request for chat")

                final_usage: Optional[LLMUsageInfo] = None
                final_raw_response: Optional[Dict[str, Any]] = None

                async for chunk_data in response_stream:
                    if not isinstance(chunk_data, dict): continue

                    delta_message_raw = chunk_data.get("message", {})
                    delta_message: LLMChatChunkDeltaMessage = {}
                    if "role" in delta_message_raw: delta_message["role"] = delta_message_raw["role"]
                    if "content" in delta_message_raw: delta_message["content"] = delta_message_raw["content"]

                    chunk_finish_reason = "done" if chunk_data.get("done") else None
                    current_chunk: LLMChatChunk = {"message_delta": delta_message, "raw_chunk": chunk_data}

                    if chunk_finish_reason:
                        current_chunk["finish_reason"] = chunk_finish_reason
                        final_usage = {
                            "prompt_tokens": chunk_data.get("prompt_eval_count"),
                            "completion_tokens": chunk_data.get("eval_count"),
                        }
                        if final_usage.get("prompt_tokens") is not None and final_usage.get("completion_tokens") is not None:
                            final_usage["total_tokens"] = final_usage["prompt_tokens"] + final_usage["completion_tokens"] # type: ignore
                        current_chunk["usage_delta"] = final_usage
                        final_raw_response = chunk_data
                    yield current_chunk
            return stream_chat_chunks()
        else:
            response_data = await self._make_request("/api/chat", payload, stream=False)
            if not isinstance(response_data, dict):
                 raise RuntimeError("Expected dict from _make_request for non-streaming chat")

            assistant_message: ChatMessage = response_data.get("message", {"role": "assistant", "content": ""})
            usage_info: LLMUsageInfo = {
                "prompt_tokens": response_data.get("prompt_eval_count"),
                "completion_tokens": response_data.get("eval_count"),
            }
            if usage_info.get("prompt_tokens") is not None and usage_info.get("completion_tokens") is not None:
                usage_info["total_tokens"] = usage_info["prompt_tokens"] + usage_info["completion_tokens"] # type: ignore
            return {
                "message": assistant_message,
                "finish_reason": "done" if response_data.get("done") else "unknown",
                "usage": usage_info,
                "raw_response": response_data,
            }

    async def get_model_info(self) -> Dict[str, Any]:
        if not self._http_client: return {"error": "HTTP client not initialized"}
        info: Dict[str, Any] = {"provider": "Ollama", "base_url": self._base_url, "default_model_configured": self._default_model}
        try:
            tags_response_any = await self._make_request("/api/tags", {})
            if isinstance(tags_response_any, dict) and "models" in tags_response_any:
                info["available_models_brief"] = [m.get("name") for m in tags_response_any["models"] if m.get("name")]

            if self._default_model: # Always try to show details for the configured default
                model_details_response_any = await self._make_request("/api/show", {"name": self._default_model})
                if isinstance(model_details_response_any, dict):
                    info["default_model_details"] = {
                        "parameters": model_details_response_any.get("parameters"),
                        "template": model_details_response_any.get("template"),
                        "family": model_details_response_any.get("details", {}).get("family"),
                    }
        except Exception as e:
            info["model_info_error"] = str(e)
        return info

    async def teardown(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
            logger.info(f"{self.plugin_id}: HTTP client closed.")
        await super().teardown()
