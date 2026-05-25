from __future__ import annotations

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


def _to_openai_messages(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    """Translate Genie ChatMessages (which may use the M5 multimodal
    ContentBlock list form) into OpenAI Chat Completion message dicts.

    For text-only ``content`` (string), passes through unchanged.
    For multimodal ``content`` (list of ContentBlocks), translates each
    block into OpenAI's content-part shape:

      Genie TextBlock                  → OpenAI ``{"type": "text", "text": ...}``
      Genie ImageBlock (url source)    → OpenAI ``{"type": "image_url", "image_url": {"url": ...}}``
      Genie ImageBlock (base64 source) → OpenAI ``{"type": "image_url", "image_url": {"url": "data:<mime>;base64,<data>"}}``
    """
    out: List[Dict[str, Any]] = []
    for m in messages:
        new_m: Dict[str, Any] = {k: v for k, v in m.items() if k != "content"}
        content = m.get("content")
        if isinstance(content, list):
            parts: List[Dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    parts.append({"type": "text", "text": str(block)})
                    continue
                btype = block.get("type")
                if btype == "text":
                    parts.append({"type": "text", "text": block.get("text", "")})
                elif btype == "image":
                    src = block.get("source", "base64")
                    if src == "url":
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": block.get("url", "")},
                            }
                        )
                    else:
                        media_type = block.get("media_type", "image/png")
                        data = block.get("data", "")
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{data}"
                                },
                            }
                        )
                else:
                    # Unknown block — degrade to JSON text
                    import json as _json
                    parts.append({"type": "text", "text": _json.dumps(block)})
            new_m["content"] = parts
        else:
            new_m["content"] = content
        out.append(new_m)
    return out


def _build_openai_response_format(response_schema: Any) -> Dict[str, Any]:
    """Translate a Pydantic BaseModel class into OpenAI's
    ``response_format={"type": "json_schema", ...}`` shape (M4).

    OpenAI requires ``strict: true`` schemas to set ``additionalProperties:
    False`` on every nested object and to mark every property as required.
    We patch the Pydantic-generated schema to satisfy those rules so the
    caller doesn't have to.
    """
    from pydantic import BaseModel

    if not (isinstance(response_schema, type) and issubclass(response_schema, BaseModel)):
        raise ValueError(
            "response_schema must be a subclass of pydantic.BaseModel"
        )
    schema = response_schema.model_json_schema()
    _make_schema_openai_strict(schema)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": response_schema.__name__,
            "description": response_schema.__doc__ or f"A {response_schema.__name__}.",
            "schema": schema,
            "strict": True,
        },
    }


def _make_schema_openai_strict(schema: Dict[str, Any]) -> None:
    """In-place: enforce OpenAI strict-mode rules on a JSON schema —
    additionalProperties=False on every object, every property listed in
    required."""
    if not isinstance(schema, dict):
        return
    if schema.get("type") == "object":
        properties = schema.get("properties") or {}
        schema["additionalProperties"] = False
        schema["required"] = list(properties.keys())
        for child in properties.values():
            _make_schema_openai_strict(child)
    # $defs (Pydantic-generated definitions) also need the strict treatment
    for child in (schema.get("$defs") or {}).values():
        _make_schema_openai_strict(child)
    # Array items
    items = schema.get("items")
    if isinstance(items, dict):
        _make_schema_openai_strict(items)
    elif isinstance(items, list):
        for item in items:
            _make_schema_openai_strict(item)


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
            logger.info(f"{self.plugin_id}: API key '{self._api_key_name}' not found. Plugin will be disabled.")
            self._client = None
            return

        try:
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=cfg.get("openai_api_base"),
                organization=cfg.get("openai_organization"),
            )
            logger.info(f"{self.plugin_id}: Initialized OpenAI client for model '{self._model_name}'.")
        except Exception as e:
            logger.error(f"{self.plugin_id}: Failed to initialize OpenAI client: {e}", exc_info=True)
            self._client = None

    def _parse_openai_response_message(self, message: OpenAIChatMessage) -> ChatMessage:
        genie_tool_calls: Optional[List[GenieToolCall]] = None
        if message.tool_calls:
            genie_tool_calls = []
            for tc in message.tool_calls:
                if tc.type == "function":
                    genie_tool_calls.append({"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}})
        return {"role": cast(Any, message.role), "content": message.content, "tool_calls": genie_tool_calls}

    def _parse_openai_usage(self, usage_data: Optional[OpenAIUsage]) -> Optional[LLMUsageInfo]:
        if not usage_data:
            return None
        return {"prompt_tokens": usage_data.prompt_tokens, "completion_tokens": usage_data.completion_tokens, "total_tokens": usage_data.total_tokens}

    async def generate(self, prompt: str, **kwargs: Any) -> LLMCompletionResponse:
        if not self._client:
            raise RuntimeError(f"{self.plugin_id}: Client not initialized.")
        messages: List[ChatMessage] = [{"role": "user", "content": prompt}]
        model_to_use = kwargs.pop("model", self._model_name)
        common_params = {"temperature": kwargs.get("temperature"), "top_p": kwargs.get("top_p"), "max_tokens": kwargs.get("max_tokens"), "stop": kwargs.get("stop_sequences"), "presence_penalty": kwargs.get("presence_penalty"), "frequency_penalty": kwargs.get("frequency_penalty")}
        request_params = {k: v for k, v in common_params.items() if v is not None}
        try:
            response = await self._client.chat.completions.create(model=model_to_use, messages=cast(Any, _to_openai_messages(messages)), **request_params)
            text_content = ""
            finish_reason = None
            if response.choices:
                text_content = response.choices[0].message.content or ""
                finish_reason = response.choices[0].finish_reason
            usage = self._parse_openai_usage(response.usage)
            return {"text": text_content, "finish_reason": finish_reason, "usage": usage, "raw_response": response.model_dump(exclude_none=True)}
        except APIError as e:
            logger.error(f"{self.plugin_id} OpenAI API Error during generate: {e.status_code} - {e.message}", exc_info=True)
            raise RuntimeError(f"OpenAI API Error: {e.status_code} - {e.message}") from e
        except Exception as e:
            logger.error(f"{self.plugin_id} Unexpected error during generate: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error in OpenAI generate: {e!s}") from e

    async def chat(self, messages: List[ChatMessage], **kwargs: Any) -> LLMChatResponse:
        if not self._client:
            raise RuntimeError(f"{self.plugin_id}: Client not initialized.")
        model_to_use = kwargs.pop("model", self._model_name)
        tools_for_api = kwargs.get("tools")
        tool_choice_for_api = kwargs.get("tool_choice")
        # M4: native structured outputs via OpenAI response_format=json_schema.
        # When the caller supplies a Pydantic model class, translate it into
        # OpenAI's json_schema format. The model is guaranteed to return JSON
        # conforming to the schema (provider-validated, no client-side retry
        # loop needed).
        response_schema = kwargs.pop("response_schema", None)
        common_params = {"temperature": kwargs.get("temperature"), "top_p": kwargs.get("top_p"), "max_tokens": kwargs.get("max_tokens"), "stop": kwargs.get("stop_sequences"), "presence_penalty": kwargs.get("presence_penalty"), "frequency_penalty": kwargs.get("frequency_penalty")}
        request_params = {k: v for k, v in common_params.items() if v is not None}
        if tools_for_api:
            request_params["tools"] = tools_for_api
        if tool_choice_for_api:
            request_params["tool_choice"] = tool_choice_for_api
        if response_schema is not None:
            request_params["response_format"] = _build_openai_response_format(response_schema)
        try:
            response = await self._client.chat.completions.create(model=model_to_use, messages=cast(Any, _to_openai_messages(messages)), **request_params)
            genie_message: ChatMessage = {"role": "assistant", "content": None}
            finish_reason = None
            if response.choices:
                openai_msg = response.choices[0].message
                genie_message = self._parse_openai_response_message(openai_msg)
                finish_reason = response.choices[0].finish_reason
            usage = self._parse_openai_usage(response.usage)
            return {"message": genie_message, "finish_reason": finish_reason, "usage": usage, "raw_response": response.model_dump(exclude_none=True)}
        except APIError as e:
            logger.error(f"{self.plugin_id} OpenAI API Error during chat: {e.status_code} - {e.message}", exc_info=True)
            raise RuntimeError(f"OpenAI API Error: {e.status_code} - {e.message}") from e
        except Exception as e:
            logger.error(f"{self.plugin_id} Unexpected error during chat: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error in OpenAI chat: {e!s}") from e

    async def get_model_info(self) -> Dict[str, Any]:
        return {"provider": "OpenAI", "configured_model_name": self._model_name, "notes": "Detailed model info (token limits, etc.) typically found in OpenAI documentation for the specified model."}

    async def teardown(self) -> None:
        if self._client:
            await self._client.close()
        self._client = None
        self._key_provider = None
        logger.info(f"{self.plugin_id}: Teardown complete.")
        await super().teardown()
