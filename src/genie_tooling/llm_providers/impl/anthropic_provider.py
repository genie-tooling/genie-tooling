"""AnthropicLLMProviderPlugin: LLMProvider implementation against the
Anthropic Messages API (Claude).

Wire-format differences from OpenAI that this plugin handles:
  * **System messages are a separate `system=` parameter**, not a role in
    the messages list. We extract any role="system" entries from the
    Genie ChatMessage stream and pass them as the `system` argument.
  * **Tool use is structured-block-based**, not OpenAI's `tool_calls`
    field. Assistant messages contain content blocks like
    ``{"type": "tool_use", "id": "...", "name": "...", "input": {...}}``,
    and tool results come back as user messages with
    ``{"type": "tool_result", "tool_use_id": "...", "content": "..."}``.
  * **`max_tokens` is required**, not optional. We default to 4096 when
    the caller doesn't specify.
  * **Streaming uses an async iterator** of `MessageStreamEvent` types,
    not OpenAI-style chunked completions.

Structured outputs (Phase 5 M4) are handled by translating the requested
``response_schema`` into a tool-use round-trip: we register a single
synthetic tool whose schema matches the Pydantic model, set the model to
``tool_choice={"type": "tool", "name": "..."}``, then return the tool's
input dict as a JSON string in the response. The PydanticOutputParser
then validates it client-side. This is Anthropic's recommended pattern
for guaranteed-shape outputs.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncIterable, Dict, List, Optional, Tuple, Union

from genie_tooling.llm_providers.abc import LLMProviderPlugin
from genie_tooling.llm_providers.types import (
    ChatMessage,
    LLMChatChunk,
    LLMChatChunkDeltaMessage,
    LLMChatResponse,
    LLMCompletionChunk,
    LLMCompletionResponse,
    LLMUsageInfo,
    ToolCall,
    ToolCallFunction,
)
from genie_tooling.security.key_provider import KeyProvider

logger = logging.getLogger(__name__)

try:
    from anthropic import AsyncAnthropic
    from anthropic.types import (
        ContentBlockDeltaEvent,
        MessageDeltaEvent,
        MessageStartEvent,
        MessageStopEvent,
    )
    ANTHROPIC_AVAILABLE = True
except ImportError:  # pragma: no cover
    AsyncAnthropic = None  # type: ignore
    ANTHROPIC_AVAILABLE = False


_DEFAULT_MODEL = "claude-sonnet-4-6"
_DEFAULT_MAX_TOKENS = 4096


class AnthropicLLMProviderPlugin(LLMProviderPlugin):
    plugin_id: str = "anthropic_llm_provider_v1"
    description: str = (
        "LLM provider for the Anthropic Messages API (Claude). Supports "
        "tool use, streaming, and structured outputs via tool-use rounds."
    )

    _client: Optional[Any] = None
    _default_model: str = _DEFAULT_MODEL
    _default_max_tokens: int = _DEFAULT_MAX_TOKENS
    _api_key_name: str = "ANTHROPIC_API_KEY"

    async def setup(self, config: Optional[Dict[str, Any]]) -> None:
        if not ANTHROPIC_AVAILABLE:
            logger.error(
                f"{self.plugin_id}: `anthropic` SDK not installed. "
                "Install with the 'anthropic' extra: `pip install genie-tooling[anthropic]`."
            )
            return
        cfg = config or {}
        self._default_model = cfg.get("model_name", _DEFAULT_MODEL)
        self._default_max_tokens = int(cfg.get("max_tokens", _DEFAULT_MAX_TOKENS))
        self._api_key_name = cfg.get("api_key_name", "ANTHROPIC_API_KEY")

        api_key: Optional[str] = cfg.get("api_key")
        if not api_key:
            key_provider: Optional[KeyProvider] = cfg.get("key_provider")
            if key_provider is not None:
                api_key = await key_provider.get_key(self._api_key_name)
        if not api_key:
            logger.error(
                f"{self.plugin_id}: no API key resolved (looked up via key_provider "
                f"under name {self._api_key_name!r}). Plugin will not be functional."
            )
            return

        self._client = AsyncAnthropic(api_key=api_key)
        logger.info(
            f"{self.plugin_id}: Initialized. Default model: {self._default_model!r}, "
            f"default max_tokens: {self._default_max_tokens}."
        )

    # ------------------------------------------------------------------
    # Message-shape translation
    # ------------------------------------------------------------------

    @staticmethod
    def _split_system_from_messages(
        messages: List[ChatMessage],
    ) -> Tuple[Optional[str], List[ChatMessage]]:
        """Anthropic puts system instruction in a separate `system=`
        argument. Concatenate all role='system' entries (in order) into
        a single string and remove them from the messages list."""
        system_parts: List[str] = []
        remaining: List[ChatMessage] = []
        for m in messages:
            if m.get("role") == "system":
                content = m.get("content")
                if isinstance(content, str) and content:
                    system_parts.append(content)
            else:
                remaining.append(m)
        system = "\n\n".join(system_parts) if system_parts else None
        return system, remaining

    @classmethod
    def _to_anthropic_messages(
        cls, messages: List[ChatMessage]
    ) -> List[Dict[str, Any]]:
        """Convert Genie ChatMessages to Anthropic message dicts.

        Genie shape (OpenAI-aligned):
          * role: system | user | assistant | tool
          * content: str (or list of content blocks per M5)
          * tool_calls: list of ToolCall (for assistant)
          * tool_call_id: str (for role=tool)

        Anthropic shape:
          * role: user | assistant
          * content: list of blocks (text / tool_use / tool_result / image)
          * tool calls live inside the assistant's content as
            {"type": "tool_use", "id": ..., "name": ..., "input": ...}
          * tool results are user messages with
            {"type": "tool_result", "tool_use_id": ..., "content": ...}
        """
        out: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            content = m.get("content")
            tool_calls = m.get("tool_calls") or []

            if role == "tool":
                # Genie tool messages become Anthropic user messages
                # containing a tool_result block.
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.get("tool_call_id", ""),
                                "content": _stringify_tool_content(content),
                            }
                        ],
                    }
                )
                continue

            if role not in ("user", "assistant"):
                # Defensive: skip anything we don't recognise (system was
                # already filtered upstream).
                continue

            anth_content = cls._content_to_anthropic_blocks(content)
            if role == "assistant" and tool_calls:
                # Append tool_use blocks for any assistant message that's
                # requesting tool execution.
                for tc in tool_calls:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    raw_args = fn.get("arguments", "{}")
                    try:
                        args_obj = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except (TypeError, json.JSONDecodeError):
                        args_obj = {"_raw": str(raw_args)}
                    anth_content.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": fn.get("name", ""),
                            "input": args_obj or {},
                        }
                    )
            if not anth_content:
                # Anthropic rejects messages with empty content lists.
                anth_content = [{"type": "text", "text": ""}]
            out.append({"role": role, "content": anth_content})
        return out

    @staticmethod
    def _content_to_anthropic_blocks(content: Any) -> List[Dict[str, Any]]:
        """Translate Genie ChatMessage.content into Anthropic content blocks.
        Supports plain strings and the Phase 5 M5 list-of-ContentBlock form.
        """
        if content is None:
            return []
        if isinstance(content, str):
            if not content:
                return []
            return [{"type": "text", "text": content}]
        if isinstance(content, list):
            blocks: List[Dict[str, Any]] = []
            for part in content:
                if not isinstance(part, dict):
                    blocks.append({"type": "text", "text": str(part)})
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    blocks.append({"type": "text", "text": part.get("text", "")})
                elif ptype == "image":
                    # Genie ImageBlock shape (per M5):
                    #   {"type": "image", "source": "url" | "base64",
                    #    "url": "...", "data": "...", "media_type": "image/png"}
                    src = part.get("source", "base64")
                    if src == "url":
                        blocks.append(
                            {
                                "type": "image",
                                "source": {"type": "url", "url": part.get("url", "")},
                            }
                        )
                    else:
                        blocks.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": part.get("media_type", "image/png"),
                                    "data": part.get("data", ""),
                                },
                            }
                        )
                else:
                    # Unknown block — degrade to text repr
                    blocks.append({"type": "text", "text": json.dumps(part)})
            return blocks
        # Last-ditch fallback
        return [{"type": "text", "text": str(content)}]

    # ------------------------------------------------------------------
    # Response shape translation
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_content_and_tool_calls(
        anthropic_response: Any,
    ) -> Tuple[Optional[str], List[ToolCall]]:
        """Pull the assistant's text content + any tool_use blocks out of
        an Anthropic message response."""
        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []
        for block in getattr(anthropic_response, "content", None) or []:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(getattr(block, "text", ""))
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=getattr(block, "id", ""),
                        type="function",
                        function=ToolCallFunction(
                            name=getattr(block, "name", ""),
                            arguments=json.dumps(getattr(block, "input", {}) or {}),
                        ),
                    )
                )
        text = "".join(text_parts) if text_parts else None
        return text, tool_calls

    @staticmethod
    def _usage_from(anthropic_response: Any) -> Optional[LLMUsageInfo]:
        usage = getattr(anthropic_response, "usage", None)
        if usage is None:
            return None
        prompt = getattr(usage, "input_tokens", None)
        completion = getattr(usage, "output_tokens", None)
        total = None
        if prompt is not None and completion is not None:
            total = prompt + completion
        return LLMUsageInfo(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
        )

    # ------------------------------------------------------------------
    # Public API: chat / completion / generate
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: List[ChatMessage],
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[LLMChatResponse, AsyncIterable[LLMChatChunk]]:
        if not self._client:
            raise RuntimeError(
                f"{self.plugin_id}: client not initialized (missing SDK or API key)."
            )

        # M4: structured outputs via tool-use round-trip.
        response_schema = kwargs.pop("response_schema", None)
        # The "tools" kwarg is the Genie-side tool definitions list (OpenAI
        # function-spec shape). Translate to Anthropic tools shape.
        genie_tools = kwargs.pop("tools", None)
        tool_choice = kwargs.pop("tool_choice", None)

        system, remaining = self._split_system_from_messages(messages)
        anth_messages = self._to_anthropic_messages(remaining)

        request: Dict[str, Any] = {
            "model": kwargs.pop("model", self._default_model),
            "max_tokens": kwargs.pop("max_tokens", self._default_max_tokens),
            "messages": anth_messages,
        }
        if system is not None:
            request["system"] = system

        # Translate optional stop sequences (Genie uses 'stop')
        stop = kwargs.pop("stop", None)
        if stop:
            request["stop_sequences"] = stop if isinstance(stop, list) else [stop]

        temperature = kwargs.pop("temperature", None)
        if temperature is not None:
            request["temperature"] = float(temperature)

        if response_schema is not None:
            # Force the model to emit a single tool call whose input matches
            # the requested schema. The PydanticOutputParser will then
            # validate client-side from the arguments JSON.
            from pydantic import BaseModel
            if not (isinstance(response_schema, type) and issubclass(response_schema, BaseModel)):
                raise ValueError(
                    "response_schema must be a subclass of pydantic.BaseModel"
                )
            tool_name = response_schema.__name__
            tool_schema = response_schema.model_json_schema()
            request["tools"] = [
                {
                    "name": tool_name,
                    "description": response_schema.__doc__ or f"Return a {tool_name}.",
                    "input_schema": tool_schema,
                }
            ]
            request["tool_choice"] = {"type": "tool", "name": tool_name}
        elif genie_tools is not None:
            request["tools"] = _to_anthropic_tools(genie_tools)
            if tool_choice is not None:
                request["tool_choice"] = tool_choice

        if stream:
            return self._chat_stream(request)
        return await self._chat_non_stream(request, response_schema)

    async def _chat_non_stream(
        self, request: Dict[str, Any], response_schema: Optional[Any]
    ) -> LLMChatResponse:
        try:
            response = await self._client.messages.create(**request)
        except Exception as e:
            logger.error(f"{self.plugin_id}: Anthropic API error: {e}", exc_info=True)
            raise RuntimeError(f"Anthropic request failed: {e}") from e

        text, tool_calls = self._extract_content_and_tool_calls(response)

        # M4: if a response_schema was set, the model emits one tool_use with
        # the structured input. Surface it as the message content (JSON string)
        # so PydanticOutputParser can validate it without bespoke routing.
        if response_schema is not None and tool_calls:
            text = tool_calls[0]["function"]["arguments"]
            tool_calls = []  # don't surface as a tool call to the caller

        chat_message: ChatMessage = {"role": "assistant", "content": text}
        if tool_calls:
            chat_message["tool_calls"] = tool_calls

        return LLMChatResponse(
            message=chat_message,
            finish_reason=getattr(response, "stop_reason", None),
            usage=self._usage_from(response),
            raw_response=response,
        )

    async def _chat_stream(
        self, request: Dict[str, Any]
    ) -> AsyncIterable[LLMChatChunk]:
        try:
            async with self._client.messages.stream(**request) as stream_ctx:
                final_usage: Optional[LLMUsageInfo] = None
                async for event in stream_ctx:
                    chunk: Optional[LLMChatChunk] = None
                    etype = type(event).__name__
                    if etype == "ContentBlockDeltaEvent":
                        delta = getattr(event, "delta", None)
                        text_delta = getattr(delta, "text", None) if delta else None
                        if text_delta:
                            chunk = LLMChatChunk(
                                message_delta=LLMChatChunkDeltaMessage(
                                    role="assistant", content=text_delta
                                ),
                                finish_reason=None,
                                usage_delta=None,
                                raw_chunk=event,
                            )
                    elif etype == "MessageDeltaEvent":
                        usage = getattr(event, "usage", None)
                        if usage is not None:
                            final_usage = LLMUsageInfo(
                                prompt_tokens=None,
                                completion_tokens=getattr(usage, "output_tokens", None),
                                total_tokens=None,
                            )
                    elif etype == "MessageStopEvent":
                        # Yield a final chunk carrying finish reason + usage
                        msg = await stream_ctx.get_final_message()
                        chunk = LLMChatChunk(
                            message_delta=LLMChatChunkDeltaMessage(role="assistant"),
                            finish_reason=getattr(msg, "stop_reason", None),
                            usage_delta=final_usage or self._usage_from(msg),
                            raw_chunk=event,
                        )
                    if chunk is not None:
                        yield chunk
        except Exception as e:
            logger.error(f"{self.plugin_id}: Anthropic stream error: {e}", exc_info=True)
            raise RuntimeError(f"Anthropic stream failed: {e}") from e

    async def generate(
        self, prompt: str, stream: bool = False, **kwargs: Any
    ) -> Union[LLMCompletionResponse, AsyncIterable[LLMCompletionChunk]]:
        """Provided for protocol parity. Anthropic's Messages API is
        chat-only; `generate` is implemented by wrapping the prompt in a
        single user message and adapting the response shape."""
        messages: List[ChatMessage] = [{"role": "user", "content": prompt}]
        chat_result = await self.chat(messages=messages, stream=stream, **kwargs)
        if stream:
            return _completion_chunks_from_chat_chunks(chat_result)  # type: ignore[arg-type]
        # Non-stream
        chat_resp = chat_result  # type: ignore[assignment]
        text = chat_resp["message"].get("content") or ""
        return LLMCompletionResponse(
            text=text,
            finish_reason=chat_resp.get("finish_reason"),
            usage=chat_resp.get("usage"),
            raw_response=chat_resp.get("raw_response"),
        )

    async def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "Anthropic",
            "default_model": self._default_model,
            "default_max_tokens": self._default_max_tokens,
            "sdk_available": ANTHROPIC_AVAILABLE,
        }

    async def teardown(self) -> None:
        # AsyncAnthropic doesn't require explicit cleanup; httpx pool
        # cleans up on GC. Nothing to do.
        self._client = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stringify_tool_content(content: Any) -> str:
    """Anthropic expects tool_result content as a string or list of blocks.
    Genie's tool results are arbitrary Python; coerce to a JSON string
    when possible."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content)
    except (TypeError, ValueError):
        return str(content)


def _to_anthropic_tools(genie_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Genie's tool definitions follow the OpenAI function-spec shape
    (a list of {"type": "function", "function": {name, description, parameters}}).
    Anthropic's shape is flatter."""
    out: List[Dict[str, Any]] = []
    for t in genie_tools or []:
        if "function" in t:
            fn = t["function"]
            out.append(
                {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                }
            )
        elif "name" in t and "input_schema" in t:
            # Already in Anthropic shape
            out.append(t)
    return out


async def _completion_chunks_from_chat_chunks(
    chat_stream: AsyncIterable[LLMChatChunk],
) -> AsyncIterable[LLMCompletionChunk]:
    async for chunk in chat_stream:
        delta = chunk.get("message_delta") or {}
        yield LLMCompletionChunk(
            text_delta=delta.get("content"),
            finish_reason=chunk.get("finish_reason"),
            usage_delta=chunk.get("usage_delta"),
            raw_chunk=chunk.get("raw_chunk"),
        )
