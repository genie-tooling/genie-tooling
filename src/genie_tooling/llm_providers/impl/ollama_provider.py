# src/genie_tooling/llm_providers/impl/ollama_provider.py
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from genie_tooling.llm_providers.abc import LLMProviderPlugin
from genie_tooling.llm_providers.types import (
    ChatMessage,
    LLMChatResponse,
    LLMCompletionResponse,
    LLMUsageInfo,
)
from genie_tooling.security.key_provider import KeyProvider

logger = logging.getLogger(__name__)

class OllamaLLMProviderPlugin(LLMProviderPlugin):
    plugin_id: str = "ollama_llm_provider_v1"
    description: str = "LLM provider for interacting with a local or remote Ollama instance."

    _http_client: Optional[httpx.AsyncClient] = None
    _base_url: str
    _default_model: str
    _request_timeout: float = 120.0 # Default timeout for Ollama requests

    async def setup(self, config: Optional[Dict[str, Any]], key_provider: KeyProvider) -> None:
        await super().setup(config, key_provider) # KeyProvider not typically used by local Ollama
        
        cfg = config or {}
        self._base_url = cfg.get("base_url", "http://localhost:11434").rstrip("/")
        self._default_model = cfg.get("model_name", "llama2") # A common default for Ollama
        self._request_timeout = float(cfg.get("request_timeout_seconds", self._request_timeout))

        self._http_client = httpx.AsyncClient(timeout=self._request_timeout)
        logger.info(f"{self.plugin_id}: Initialized. Base URL: {self._base_url}, Default Model: {self._default_model}")

    async def _make_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self._http_client:
            raise RuntimeError(f"{self.plugin_id}: HTTP client not initialized.")
        
        url = f"{self._base_url}{endpoint}"
        try:
            response = await self._http_client.post(url, json=payload)
            response.raise_for_status() # Raise an exception for HTTP error codes (4xx or 5xx)
            
            # Ollama streams JSON objects separated by newlines if stream=True
            # For stream=False, it should be a single JSON object.
            # We'll assume stream=False for simplicity in this V1 plugin.
            return response.json()
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
        except json.JSONDecodeError as e:
            logger.error(f"{self.plugin_id}: Failed to decode JSON response from {url}: {e}", exc_info=True)
            raise RuntimeError(f"Ollama response JSON decode error: {e}") from e


    async def generate(self, prompt: str, **kwargs: Any) -> LLMCompletionResponse:
        model_name = kwargs.pop("model", self._default_model)
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False, # Keep it simple for V1 completion
            "options": kwargs.get("options", {}), # Pass through temperature, top_p etc.
            **kwargs # Pass other top-level params like format, keep_alive
        }
        
        logger.debug(f"{self.plugin_id}: Sending generate request to Ollama. Model: {model_name}, Prompt: '{prompt[:50]}...'")
        response_data = await self._make_request("/api/generate", payload)

        usage_info: LLMUsageInfo = {
            "prompt_tokens": response_data.get("prompt_eval_count"),
            "completion_tokens": response_data.get("eval_count"),
            # total_tokens might not be directly provided, sum if components exist
        }
        if usage_info.get("prompt_tokens") is not None and usage_info.get("completion_tokens") is not None:
            usage_info["total_tokens"] = usage_info["prompt_tokens"] + usage_info["completion_tokens"] # type: ignore

        return {
            "text": response_data.get("response", ""),
            "finish_reason": "done" if response_data.get("done") else "unknown", # Ollama's 'done' is boolean
            "usage": usage_info,
            "raw_response": response_data,
        }

    async def chat(self, messages: List[ChatMessage], **kwargs: Any) -> LLMChatResponse:
        model_name = kwargs.pop("model", self._default_model)
        
        # Ollama expects messages in {"role": "user/assistant/system", "content": "..."}
        # Our ChatMessage type is already compatible.
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False, # Keep it simple for V1
            "options": kwargs.get("options", {}),
            **kwargs # Pass other top-level params like format, keep_alive
        }

        logger.debug(f"{self.plugin_id}: Sending chat request to Ollama. Model: {model_name}, Messages: {len(messages)}")
        response_data = await self._make_request("/api/chat", payload)

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
        """Attempts to get info about the default model if set, or list available models."""
        if not self._http_client:
            return {"error": "HTTP client not initialized"}
        
        info: Dict[str, Any] = {"provider": "Ollama", "base_url": self._base_url, "default_model_configured": self._default_model}
        try:
            # Get list of local models
            tags_response = await self._make_request("/api/tags", {})
            if isinstance(tags_response, dict) and "models" in tags_response:
                info["available_models_brief"] = [m.get("name") for m in tags_response["models"] if m.get("name")]
            
            # Get detailed info if a default model is configured
            if self._default_model and self._default_model != "llama2": # Don't spam for the absolute default
                show_payload = {"name": self._default_model}
                model_details = await self._make_request("/api/show", show_payload)
                info["default_model_details"] = {
                    "parameters": model_details.get("parameters"),
                    "template": model_details.get("template"),
                    "family": model_details.get("details", {}).get("family"),
                }
        except Exception as e:
            logger.warning(f"{self.plugin_id}: Could not fetch extended model info from Ollama: {e}")
            info["model_info_error"] = str(e)
        return info

    async def teardown(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
            logger.info(f"{self.plugin_id}: HTTP client closed.")
        await super().teardown()