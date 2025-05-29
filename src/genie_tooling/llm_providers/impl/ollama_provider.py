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

    async def _make_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self._http_client:
            raise RuntimeError(f"{self.plugin_id}: HTTP client not initialized.")

        url = f"{self._base_url}{endpoint}"
        try:
            response = await self._http_client.post(url, json=payload)
            response.raise_for_status()
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
            "stream": False,
            "options": kwargs.get("options", {}),
            **kwargs
        }
        response_data = await self._make_request("/api/generate", payload)
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

    async def chat(self, messages: List[ChatMessage], **kwargs: Any) -> LLMChatResponse:
        model_name = kwargs.pop("model", self._default_model)
        logger.debug(f"OllamaLLMProviderPlugin ({self.plugin_id}) chat: Using model_name: '{model_name}' (derived from kwargs or self._default_model: '{self._default_model}')")
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": kwargs.get("options", {}),
            **kwargs
        }
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
        if not self._http_client: return {"error": "HTTP client not initialized"}
        info: Dict[str, Any] = {"provider": "Ollama", "base_url": self._base_url, "default_model_configured": self._default_model}
        try:
            tags_response = await self._make_request("/api/tags", {})
            if isinstance(tags_response, dict) and "models" in tags_response:
                info["available_models_brief"] = [m.get("name") for m in tags_response["models"] if m.get("name")]
            if self._default_model and self._default_model != "llama2":
                model_details = await self._make_request("/api/show", {"name": self._default_model})
                info["default_model_details"] = {
                    "parameters": model_details.get("parameters"), "template": model_details.get("template"),
                    "family": model_details.get("details", {}).get("family"),
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
