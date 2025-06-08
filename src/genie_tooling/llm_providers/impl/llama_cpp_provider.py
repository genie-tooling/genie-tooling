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
    generate_gbnf_grammar_from_pydantic_models,
)

logger = logging.getLogger(__name__)

class LlamaCppLLMProviderPlugin(LLMProviderPlugin):
    plugin_id: str = "llama_cpp_llm_provider_v1"
    description: str = "LLM provider for llama.cpp server endpoints (/v1/completion, /v1/chat/completions)."

    _http_client: Optional[httpx.AsyncClient] = None
    _base_url: str
    _default_model_alias: Optional[str] = None
    _request_timeout: float = 120.0
    _api_key_name: Optional[str] = None
    _key_provider: Optional[KeyProvider] = None

    async def setup(self, config: Optional[Dict[str, Any]]) -> None:
        await super().setup(config)
        cfg = config or {}
        self._base_url = cfg.get("base_url", "http://localhost:8080").rstrip("/")
        self._default_model_alias = cfg.get("model_name")
        self._request_timeout = float(cfg.get("request_timeout_seconds", self._request_timeout))
        self._api_key_name = cfg.get("api_key_name")
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
        # Ensure stream parameter is in the payload for llama.cpp
        payload["stream"] = stream

        # ADDED DEBUG LOG FOR PAYLOAD
        logger.debug(f"{self.plugin_id}: Sending payload to {url}: {json.dumps(payload, indent=2, default=str)}")


        try:
            response = await self._http_client.post(url, json=payload)
            response.raise_for_status()

            if stream:
                async def stream_generator():
                    try:
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                line_content = line[len("data: "):]
                                if line_content == "[DONE]":
                                    break
                                try:
                                    yield json.loads(line_content)
                                except json.JSONDecodeError:
                                    logger.error(f"{self.plugin_id}: Failed to decode JSON stream chunk: {line_content}")
                            elif line.strip():
                                logger.debug(f"{self.plugin_id}: Received non-data line in stream: {line}")
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
                json_err = e.response.json()
                if isinstance(json_err, dict) and "error" in json_err and isinstance(json_err["error"], dict):
                    err_body = json_err["error"].get("message", json.dumps(json_err["error"]))
                elif isinstance(json_err, dict) and "detail" in json_err:
                    err_body = json_err["detail"] if isinstance(json_err["detail"], str) else json.dumps(json_err["detail"])
            except json.JSONDecodeError:
                err_body = e.response.text
            logger.error(f"{self.plugin_id}: HTTP error calling {url}: {e.response.status_code} - {err_body}", exc_info=False) # Set exc_info=False for cleaner prod logs
            raise RuntimeError(f"llama.cpp API error: {e.response.status_code} - {err_body}") from e
        except httpx.RequestError as e:
            logger.error(f"{self.plugin_id}: Request error calling {url}: {e}", exc_info=True)
            raise RuntimeError(f"llama.cpp request failed: {e}") from e
        except json.JSONDecodeError as e: # Should only happen for non-streaming if response isn't JSON
            logger.error(f"{self.plugin_id}: Failed to decode JSON response from {url}: {e}", exc_info=True)
            raise RuntimeError(f"llama.cpp response JSON decode error: {e}") from e


    async def generate(
        self, prompt: str, stream: bool = False, **kwargs: Any
    ) -> Union[LLMCompletionResponse, AsyncIterable[LLMCompletionChunk]]:
        payload: Dict[str, Any] = {"prompt": prompt}

        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]
        elif "n_predict" in kwargs:
            payload["max_tokens"] = kwargs["n_predict"]

        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            payload["top_k"] = kwargs["top_k"]
        if "stop_sequences" in kwargs and kwargs["stop_sequences"]:
            payload["stop"] = kwargs["stop_sequences"]
        if "seed" in kwargs:
            payload["seed"] = kwargs["seed"]

        output_schema = kwargs.get("output_schema")
        if output_schema:
            try:
                gbnf_grammar: Optional[str] = None
                if isinstance(output_schema, type) and issubclass(output_schema, BaseModel):
                    gbnf_grammar = generate_gbnf_grammar_from_pydantic_models([output_schema])
                elif isinstance(output_schema, dict):
                    dynamic_models = create_dynamic_models_from_dictionaries([output_schema])
                    if dynamic_models:
                        gbnf_grammar = generate_gbnf_grammar_from_pydantic_models(dynamic_models)
                if gbnf_grammar:
                    payload["grammar"] = gbnf_grammar
                    logger.info(f"{self.plugin_id}: Using GBNF grammar for structured output via generate().")
                    if "max_tokens" not in payload:
                        payload["max_tokens"] = kwargs.get("n_predict", 1024) # Default to 1024 if GBNF and not set
            except Exception as e_gbnf:
                logger.error(f"{self.plugin_id}: Failed to generate GBNF grammar from output_schema: {e_gbnf}", exc_info=True)

        endpoint = "/v1/completions"
        force_stream_processing = "grammar" in payload

        if stream or force_stream_processing:
            async def format_chunks_for_caller() -> AsyncIterable[LLMCompletionChunk]:
                server_stream_request = stream or force_stream_processing
                response_stream = await self._make_request(endpoint, payload, stream=server_stream_request)

                if not isinstance(response_stream, AsyncIterable):
                    if isinstance(response_stream, dict) and not server_stream_request:
                        chunk_data = response_stream
                        text_delta = ""
                        finish_reason_str: Optional[str] = None
                        raw_usage_data: Optional[Dict[str, Any]] = None
                        if "choices" in chunk_data and chunk_data["choices"]:
                            choice = chunk_data["choices"][0]
                            text_delta = choice.get("text", "")
                            finish_reason_str = choice.get("finish_reason")
                        elif "content" in chunk_data:
                            text_delta = chunk_data.get("content", "")
                            if chunk_data.get("stopped_eos"):
                                finish_reason_str = "stop"
                        current_chunk: LLMCompletionChunk = {"text_delta": text_delta, "raw_chunk": chunk_data}
                        if finish_reason_str:
                            current_chunk["finish_reason"] = finish_reason_str
                        raw_usage_data = chunk_data.get("usage")
                        if not raw_usage_data and ("tokens_evaluated" in chunk_data or "tokens_predicted" in chunk_data):
                            raw_usage_data = {"prompt_tokens":
                            chunk_data.get("tokens_evaluated"), "completion_tokens": chunk_data.get("tokens_predicted")}
                        if raw_usage_data:
                            usage_delta: LLMUsageInfo = {
                                "prompt_tokens": raw_usage_data.get("prompt_tokens"),
                                "completion_tokens": raw_usage_data.get("completion_tokens"),
                                "total_tokens": raw_usage_data.get("total_tokens")
                                if raw_usage_data.get("total_tokens") is not None
                                else ((raw_usage_data.get("prompt_tokens",0) or 0) + (raw_usage_data.get("completion_tokens",0) or 0))
                            }
                            current_chunk["usage_delta"] = usage_delta
                        yield current_chunk
                        return
                    raise RuntimeError("Expected stream from _make_request for generate when server_stream_request was True")

                async for chunk_data in response_stream:
                    if not isinstance(chunk_data, dict):
                        continue
                    text_delta = ""
                    finish_reason_str = None
                    raw_usage_data = None
                    is_final_chunk_from_server = False
                    if "choices" in chunk_data and chunk_data["choices"]:
                        choice = chunk_data["choices"][0]
                        text_delta = choice.get("text", "")
                        if choice.get("finish_reason") is not None:
                            finish_reason_str = choice.get("finish_reason")
                            is_final_chunk_from_server = True
                    elif "content" in chunk_data:
                        text_delta = chunk_data.get("content", "")
                        if chunk_data.get("stop", False):
                            is_final_chunk_from_server = True
                            if chunk_data.get("stopped_eos"):
                                finish_reason_str = "stop"
                            elif chunk_data.get("stopped_word"):
                                finish_reason_str = "stop_sequence"
                            elif chunk_data.get("stopped_limit"):
                                finish_reason_str = "length"
                            else: finish_reason_str = "unknown_stop"
                    current_chunk: LLMCompletionChunk = {"text_delta": text_delta, "raw_chunk": chunk_data}
                    if finish_reason_str:
                        current_chunk["finish_reason"] = finish_reason_str
                    if is_final_chunk_from_server:
                        raw_usage_data = chunk_data.get("usage")
                        if not raw_usage_data and ("tokens_evaluated" in chunk_data or "tokens_predicted" in chunk_data):
                             raw_usage_data = {"prompt_tokens": chunk_data.get("tokens_evaluated"), "completion_tokens": chunk_data.get("tokens_predicted")}
                        if raw_usage_data:
                            usage_delta_val: LLMUsageInfo = {
                                "prompt_tokens": raw_usage_data.get("prompt_tokens"),
                                "completion_tokens": raw_usage_data.get("completion_tokens"),
                                "total_tokens": raw_usage_data.get("total_tokens")
                                if raw_usage_data.get("total_tokens") is not None
                                else ((raw_usage_data.get("prompt_tokens",0) or 0) + (raw_usage_data.get("completion_tokens",0) or 0))
                            }
                            current_chunk["usage_delta"] = usage_delta_val
                    yield current_chunk
            if stream:
                return format_chunks_for_caller()
            else:
                accumulated_text = ""
                final_finish_reason: Optional[str] = "unknown"
                final_usage_info: Optional[LLMUsageInfo] = None
                final_raw_resp: Any = {}
                async for chk in format_chunks_for_caller():
                    accumulated_text += chk.get("text_delta", "")
                    if chk.get("finish_reason"):
                        final_finish_reason = chk.get("finish_reason")
                    if chk.get("usage_delta"):
                        final_usage_info = chk.get("usage_delta")
                    final_raw_resp = chk.get("raw_chunk", {})
                return {"text": accumulated_text, "finish_reason": final_finish_reason, "usage": final_usage_info, "raw_response": final_raw_resp}
        else:
            response_data = await self._make_request(endpoint, payload, stream=False)
            if not isinstance(response_data, dict):
                 raise RuntimeError("Expected dict from _make_request for non-streaming generate")
            text_content = ""
            finish_reason = "unknown"
            if "choices" in response_data and response_data["choices"]:
                choice = response_data["choices"][0]
                text_content = choice.get("text", "")
                finish_reason = choice.get("finish_reason", "unknown")
            elif "content" in response_data:
                text_content = response_data.get("content", "")
                if response_data.get("stopped_eos"):
                    finish_reason = "stop"
                elif response_data.get("stopped_word"):
                    finish_reason = "stop_sequence"
                elif response_data.get("stopped_limit"):
                    finish_reason = "length"
            usage_info: Optional[LLMUsageInfo] = None
            raw_usage = response_data.get("usage")
            if raw_usage:
                usage_info = {"prompt_tokens": raw_usage.get("prompt_tokens"), "completion_tokens": raw_usage.get("completion_tokens"), "total_tokens": raw_usage.get("total_tokens")}
            elif "tokens_evaluated" in response_data:
                usage_info = {"prompt_tokens": response_data.get("tokens_evaluated"), "completion_tokens": response_data.get("tokens_predicted")}
                if usage_info.get("prompt_tokens") is not None and usage_info.get("completion_tokens") is not None:
                    usage_info["total_tokens"] = (usage_info["prompt_tokens"] or 0) + (usage_info["completion_tokens"] or 0)
            return {"text": text_content, "finish_reason": finish_reason, "usage": usage_info, "raw_response": response_data}

    async def chat(
        self, messages: List[ChatMessage], stream: bool = False, **kwargs: Any
    ) -> Union[LLMChatResponse, AsyncIterable[LLMChatChunk]]:
        payload: Dict[str, Any] = {"messages": messages}
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            payload["top_k"] = kwargs["top_k"]
        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]
        if "stop_sequences" in kwargs and kwargs["stop_sequences"]:
            payload["stop"] = kwargs["stop_sequences"]
        if "tools" in kwargs:
            payload["tools"] = kwargs["tools"]
        if "tool_choice" in kwargs:
            payload["tool_choice"] = kwargs["tool_choice"]

        output_schema = kwargs.get("output_schema")
        if output_schema:
            try:
                gbnf_grammar: Optional[str] = None
                if isinstance(output_schema, type) and issubclass(output_schema, BaseModel):
                    gbnf_grammar = generate_gbnf_grammar_from_pydantic_models([output_schema])
                elif isinstance(output_schema, dict):
                    dynamic_models = create_dynamic_models_from_dictionaries([output_schema])
                    if dynamic_models:
                        gbnf_grammar = generate_gbnf_grammar_from_pydantic_models(dynamic_models)
                if gbnf_grammar:
                    payload["grammar"] = gbnf_grammar
                    logger.info(f"{self.plugin_id}: Using GBNF grammar for structured chat output.")
                    if "max_tokens" not in payload and "n_predict" not in kwargs: # Ensure max_tokens for GBNF chat
                        payload["max_tokens"] = 1024 # Default for GBNF chat if not specified
            except Exception as e_gbnf:
                logger.error(f"{self.plugin_id}: Failed to generate GBNF grammar for chat: {e_gbnf}", exc_info=True)

        endpoint = "/v1/chat/completions"
        force_stream_processing_chat = "grammar" in payload

        if stream or force_stream_processing_chat:
            async def format_chat_chunks_for_caller() -> AsyncIterable[LLMChatChunk]:
                server_stream_request_chat = stream or force_stream_processing_chat
                response_stream = await self._make_request(endpoint, payload, stream=server_stream_request_chat)
                if not isinstance(response_stream, AsyncIterable):
                    if isinstance(response_stream, dict) and not server_stream_request_chat:
                        chunk_data = response_stream
                        choice = chunk_data.get("choices", [{}])[0]
                        delta = choice.get("message", {})
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
                            usage_data = chunk_data.get("usage")
                            if usage_data:
                                current_chunk["usage_delta"] = {"prompt_tokens": usage_data.get("prompt_tokens"), "completion_tokens": usage_data.get("completion_tokens"), "total_tokens": usage_data.get("total_tokens")}
                        yield current_chunk
                        return
                    raise RuntimeError("Expected stream from _make_request for chat when server_stream_request_chat was True")

                async for chunk_data in response_stream:
                    if not isinstance(chunk_data, dict):
                        continue
                    choice = chunk_data.get("choices", [{}])[0]
                    delta = choice.get("delta", {})
                    delta_message: LLMChatChunkDeltaMessage = {}
                    if "role" in delta:
                        delta_message["role"] = delta["role"] # type: ignore
                    if delta.get("content") is not None :
                        delta_message["content"] = delta["content"]
                    if "tool_calls" in delta:
                        delta_message["tool_calls"] = delta["tool_calls"] # type: ignore
                    current_chunk: LLMChatChunk = {"message_delta": delta_message, "raw_chunk": chunk_data}
                    if choice.get("finish_reason") is not None:
                        current_chunk["finish_reason"] = choice.get("finish_reason")
                        usage_data = chunk_data.get("usage")
                        if usage_data:
                            current_chunk["usage_delta"] = {"prompt_tokens": usage_data.get("prompt_tokens"), "completion_tokens": usage_data.get("completion_tokens"), "total_tokens": usage_data.get("total_tokens")}
                    yield current_chunk
            if stream:
                return format_chat_chunks_for_caller()
            else:
                accumulated_content: Optional[str] = None
                accumulated_tool_calls: List[Any] = []
                final_role_acc: Optional[str] = "assistant"
                final_finish_reason_acc: Optional[str] = "unknown"
                final_usage_acc: Optional[LLMUsageInfo] = None
                final_raw_resp_acc: Any = {}
                async for chk in format_chat_chunks_for_caller():
                    delta = chk.get("message_delta", {})
                    if delta.get("role"):
                        final_role_acc = delta.get("role") # type: ignore
                    content_delta = delta.get("content")
                    if content_delta is not None:
                        accumulated_content = (accumulated_content or "") + content_delta
                    if delta.get("tool_calls"):
                        accumulated_tool_calls.extend(delta.get("tool_calls",[]))
                    if chk.get("finish_reason"):
                        final_finish_reason_acc = chk.get("finish_reason")
                    if chk.get("usage_delta"):
                        final_usage_acc = chk.get("usage_delta")
                    final_raw_resp_acc = chk.get("raw_chunk", {})
                final_msg: ChatMessage = {"role": cast(Any, final_role_acc)}
                if accumulated_content is not None:
                    final_msg["content"] = accumulated_content
                if accumulated_tool_calls:
                    final_msg["tool_calls"] = accumulated_tool_calls
                return {"message": final_msg, "finish_reason": final_finish_reason_acc, "usage": final_usage_acc, "raw_response": final_raw_resp_acc}
        else:
            response_data = await self._make_request(endpoint, payload, stream=False)
            if not isinstance(response_data, dict):
                 raise RuntimeError("Expected dict from _make_request for non-streaming chat")
            choice = response_data.get("choices", [{}])[0]
            assistant_message_raw = choice.get("message", {"role": "assistant", "content": ""})
            assistant_message: ChatMessage = {"role": cast(Any, assistant_message_raw.get("role", "assistant")), "content": assistant_message_raw.get("content")}
            if "tool_calls" in assistant_message_raw:
                assistant_message["tool_calls"] = assistant_message_raw["tool_calls"]
            usage_data = response_data.get("usage")
            usage_info: Optional[LLMUsageInfo] = None
            if usage_data:
                usage_info = {"prompt_tokens": usage_data.get("prompt_tokens"), "completion_tokens": usage_data.get("completion_tokens"), "total_tokens": usage_data.get("total_tokens")}
            return {"message": assistant_message, "finish_reason": choice.get("finish_reason"), "usage": usage_info, "raw_response": response_data}

    async def get_model_info(self) -> Dict[str, Any]:
        if not self._http_client:
            return {"error": "HTTP client not initialized"}

        info: Dict[str, Any] = {
            "provider": "llama.cpp",
            "base_url": self._base_url,
            "configured_model_alias": self._default_model_alias or "N/A (server default)",
            "notes": "llama.cpp server usually serves a single pre-loaded model. Specific model details depend on server startup configuration. Supports GBNF grammar for structured output via `output_schema` in `generate()` and potentially `chat()`.",
        }
        try:
            # llama.cpp server provides an OpenAI-compatible /v1/models endpoint
            response = await self._http_client.get(f"{self._base_url}/v1/models")
            response.raise_for_status()
            models_data = response.json()
            if "data" in models_data and isinstance(models_data["data"], list):
                info["available_models_on_server"] = [m.get("id") for m in models_data["data"]]
        except Exception as e:
            logger.warning(f"{self.plugin_id}: Could not retrieve model info from /v1/models endpoint: {e}")
            info["model_info_error"] = str(e)
        return info


    async def teardown(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        self._key_provider = None
        logger.info(f"{self.plugin_id}: Teardown complete.")
        await super().teardown()
