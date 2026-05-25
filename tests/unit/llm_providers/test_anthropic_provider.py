"""Unit tests for AnthropicLLMProviderPlugin (Phase 5 — M1).

The Anthropic API is mocked at the AsyncAnthropic.messages.create boundary;
these tests verify the Genie ↔ Anthropic translation layer (message-shape
conversion, tool_use ↔ tool_calls round-trip, structured-output coercion,
system-message extraction, usage parsing) without hitting the real API.

Live integration tests against Anthropic require an ANTHROPIC_API_KEY and
are gated separately.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest
from genie_tooling.llm_providers.impl.anthropic_provider import (
    ANTHROPIC_AVAILABLE,
    AnthropicLLMProviderPlugin,
    _to_anthropic_tools,
)
from genie_tooling.llm_providers.types import ChatMessage
from pydantic import BaseModel

pytestmark = pytest.mark.skipif(
    not ANTHROPIC_AVAILABLE, reason="anthropic SDK not installed"
)


# ---------------------------------------------------------------------------
# Helpers — fake Anthropic response objects
# ---------------------------------------------------------------------------


def _text_block(text: str):
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(id: str, name: str, input: Dict[str, Any]):
    return SimpleNamespace(type="tool_use", id=id, name=name, input=input)


def _fake_response(content_blocks, stop_reason="end_turn",
                    input_tokens=10, output_tokens=5):
    return SimpleNamespace(
        content=content_blocks,
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=input_tokens, output_tokens=output_tokens
        ),
    )


async def _make_plugin(model="claude-sonnet-4-6", max_tokens=4096) -> AnthropicLLMProviderPlugin:
    plugin = AnthropicLLMProviderPlugin()
    key_provider = MagicMock()
    key_provider.get_key = AsyncMock(return_value="test-api-key")
    await plugin.setup(
        config={
            "key_provider": key_provider,
            "model_name": model,
            "max_tokens": max_tokens,
        }
    )
    return plugin


# ---------------------------------------------------------------------------
# Message-shape translation
# ---------------------------------------------------------------------------


def test_split_system_extracts_and_concatenates_system_messages():
    msgs: List[ChatMessage] = [
        {"role": "system", "content": "First system instruction."},
        {"role": "user", "content": "Hi"},
        {"role": "system", "content": "Second one."},
        {"role": "assistant", "content": "Hello"},
    ]
    system, remaining = AnthropicLLMProviderPlugin._split_system_from_messages(msgs)
    assert system == "First system instruction.\n\nSecond one."
    assert len(remaining) == 2
    assert all(m["role"] != "system" for m in remaining)


def test_split_system_returns_none_when_no_system():
    msgs: List[ChatMessage] = [{"role": "user", "content": "hi"}]
    system, remaining = AnthropicLLMProviderPlugin._split_system_from_messages(msgs)
    assert system is None
    assert remaining == msgs


def test_to_anthropic_messages_basic_user_message():
    msgs: List[ChatMessage] = [{"role": "user", "content": "hello"}]
    out = AnthropicLLMProviderPlugin._to_anthropic_messages(msgs)
    assert out == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]


def test_to_anthropic_messages_assistant_with_tool_calls():
    """Genie assistant message with tool_calls → Anthropic assistant
    message with text + tool_use blocks."""
    msgs: List[ChatMessage] = [
        {
            "role": "assistant",
            "content": "let me calculate that",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "calculator_tool",
                        "arguments": json.dumps({"num1": 5, "num2": 3, "operation": "add"}),
                    },
                }
            ],
        }
    ]
    out = AnthropicLLMProviderPlugin._to_anthropic_messages(msgs)
    assert out[0]["role"] == "assistant"
    blocks = out[0]["content"]
    assert blocks[0] == {"type": "text", "text": "let me calculate that"}
    assert blocks[1] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "calculator_tool",
        "input": {"num1": 5, "num2": 3, "operation": "add"},
    }


def test_to_anthropic_messages_tool_role_becomes_user_tool_result():
    msgs: List[ChatMessage] = [
        {
            "role": "tool",
            "content": json.dumps({"result": 8}),
            "tool_call_id": "call_1",
        }
    ]
    out = AnthropicLLMProviderPlugin._to_anthropic_messages(msgs)
    assert out[0] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "call_1",
                "content": json.dumps({"result": 8}),
            }
        ],
    }


def test_content_blocks_image_url_translation():
    """Phase 5 M5: Genie ImageBlock translates to Anthropic image block."""
    blocks = AnthropicLLMProviderPlugin._content_to_anthropic_blocks(
        [
            {"type": "text", "text": "what's in this image?"},
            {"type": "image", "source": "url", "url": "https://x/y.png"},
        ]
    )
    assert blocks == [
        {"type": "text", "text": "what's in this image?"},
        {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}},
    ]


def test_content_blocks_image_base64_translation():
    blocks = AnthropicLLMProviderPlugin._content_to_anthropic_blocks(
        [
            {
                "type": "image",
                "source": "base64",
                "media_type": "image/jpeg",
                "data": "AQID",
            }
        ]
    )
    assert blocks == [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": "AQID",
            },
        }
    ]


def test_content_blocks_empty_string_yields_no_blocks():
    assert AnthropicLLMProviderPlugin._content_to_anthropic_blocks("") == []
    assert AnthropicLLMProviderPlugin._content_to_anthropic_blocks(None) == []


# ---------------------------------------------------------------------------
# Response translation
# ---------------------------------------------------------------------------


def test_extract_content_and_tool_calls_text_only():
    resp = _fake_response([_text_block("hello world")])
    text, tcs = AnthropicLLMProviderPlugin._extract_content_and_tool_calls(resp)
    assert text == "hello world"
    assert tcs == []


def test_extract_content_and_tool_calls_with_tool_use():
    resp = _fake_response(
        [
            _text_block("calling tool now"),
            _tool_use_block("call_1", "calculator_tool", {"x": 5}),
        ]
    )
    text, tcs = AnthropicLLMProviderPlugin._extract_content_and_tool_calls(resp)
    assert text == "calling tool now"
    assert len(tcs) == 1
    assert tcs[0]["id"] == "call_1"
    assert tcs[0]["function"]["name"] == "calculator_tool"
    assert json.loads(tcs[0]["function"]["arguments"]) == {"x": 5}


def test_usage_from_anthropic_response():
    resp = _fake_response([_text_block("hi")], input_tokens=10, output_tokens=20)
    usage = AnthropicLLMProviderPlugin._usage_from(resp)
    assert usage == {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
    }


# ---------------------------------------------------------------------------
# chat() end-to-end with mocked Anthropic client
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_chat_non_stream_basic_round_trip():
    plugin = await _make_plugin()
    mock_messages = MagicMock()
    mock_messages.create = AsyncMock(
        return_value=_fake_response([_text_block("hello, human")])
    )
    plugin._client = MagicMock()
    plugin._client.messages = mock_messages

    resp = await plugin.chat([{"role": "user", "content": "hi"}])
    assert resp["message"]["content"] == "hello, human"
    assert resp["finish_reason"] == "end_turn"
    assert resp["usage"]["total_tokens"] == 15

    # The Anthropic API was called with the expected shape.
    call_kwargs = mock_messages.create.await_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-4-6"
    assert call_kwargs["max_tokens"] == 4096
    assert call_kwargs["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    ]
    # No system was extracted
    assert "system" not in call_kwargs


@pytest.mark.asyncio()
async def test_chat_extracts_system_to_separate_param():
    plugin = await _make_plugin()
    mock_messages = MagicMock()
    mock_messages.create = AsyncMock(return_value=_fake_response([_text_block("ok")]))
    plugin._client = MagicMock()
    plugin._client.messages = mock_messages

    await plugin.chat(
        [
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": "hi"},
        ]
    )
    call_kwargs = mock_messages.create.await_args.kwargs
    assert call_kwargs["system"] == "you are a helpful assistant"
    # System message must NOT appear in the messages list (Anthropic rejects)
    assert all(m["role"] in ("user", "assistant") for m in call_kwargs["messages"])


@pytest.mark.asyncio()
async def test_chat_surfaces_tool_calls_in_response():
    plugin = await _make_plugin()
    mock_messages = MagicMock()
    mock_messages.create = AsyncMock(
        return_value=_fake_response(
            [
                _text_block("about to call calculator"),
                _tool_use_block("toolu_1", "calculator_tool", {"x": 7}),
            ],
            stop_reason="tool_use",
        )
    )
    plugin._client = MagicMock()
    plugin._client.messages = mock_messages

    resp = await plugin.chat([{"role": "user", "content": "what's 5+2?"}])
    assert resp["finish_reason"] == "tool_use"
    assert resp["message"]["content"] == "about to call calculator"
    assert resp["message"]["tool_calls"]
    tc = resp["message"]["tool_calls"][0]
    assert tc["id"] == "toolu_1"
    assert tc["function"]["name"] == "calculator_tool"


@pytest.mark.asyncio()
async def test_chat_with_response_schema_forces_tool_use_round_trip():
    """M4: when caller passes response_schema, the plugin registers a
    single synthetic tool, forces tool_choice to that tool, and returns
    the tool's JSON arguments as the message content for the
    PydanticOutputParser to validate."""

    class Person(BaseModel):
        name: str
        age: int

    plugin = await _make_plugin()
    mock_messages = MagicMock()
    mock_messages.create = AsyncMock(
        return_value=_fake_response(
            [_tool_use_block("toolu_x", "Person", {"name": "Alice", "age": 30})],
            stop_reason="tool_use",
        )
    )
    plugin._client = MagicMock()
    plugin._client.messages = mock_messages

    resp = await plugin.chat(
        [{"role": "user", "content": "Make a person"}],
        response_schema=Person,
    )
    # The structured output is delivered as a JSON string in message.content,
    # NOT as a tool_calls entry (since the caller wanted a model, not a call).
    content = resp["message"]["content"]
    assert content is not None
    parsed = json.loads(content)
    assert parsed == {"name": "Alice", "age": 30}
    assert not resp["message"].get("tool_calls")

    # The API was called with tools=[<single synthetic tool>] and
    # tool_choice={type: tool, name: Person}.
    call_kwargs = mock_messages.create.await_args.kwargs
    tools = call_kwargs.get("tools")
    assert tools and len(tools) == 1
    assert tools[0]["name"] == "Person"
    assert "name" in tools[0]["input_schema"]["properties"]
    assert call_kwargs["tool_choice"] == {"type": "tool", "name": "Person"}


@pytest.mark.asyncio()
async def test_chat_translates_genie_tools_to_anthropic_shape():
    plugin = await _make_plugin()
    mock_messages = MagicMock()
    mock_messages.create = AsyncMock(return_value=_fake_response([_text_block("ok")]))
    plugin._client = MagicMock()
    plugin._client.messages = mock_messages

    genie_tools = [
        {
            "type": "function",
            "function": {
                "name": "calculator_tool",
                "description": "Does math.",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                },
            },
        }
    ]
    await plugin.chat(
        [{"role": "user", "content": "math please"}],
        tools=genie_tools,
    )
    call_kwargs = mock_messages.create.await_args.kwargs
    assert call_kwargs["tools"] == [
        {
            "name": "calculator_tool",
            "description": "Does math.",
            "input_schema": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            },
        }
    ]


@pytest.mark.asyncio()
async def test_chat_raises_runtime_error_on_api_failure():
    plugin = await _make_plugin()
    mock_messages = MagicMock()
    mock_messages.create = AsyncMock(side_effect=Exception("network down"))
    plugin._client = MagicMock()
    plugin._client.messages = mock_messages

    with pytest.raises(RuntimeError, match="Anthropic request failed"):
        await plugin.chat([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio()
async def test_chat_with_no_client_raises():
    plugin = AnthropicLLMProviderPlugin()
    # Skip setup → _client stays None
    with pytest.raises(RuntimeError, match="client not initialized"):
        await plugin.chat([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio()
async def test_generate_wraps_prompt_as_user_message():
    plugin = await _make_plugin()
    mock_messages = MagicMock()
    mock_messages.create = AsyncMock(
        return_value=_fake_response([_text_block("response text")])
    )
    plugin._client = MagicMock()
    plugin._client.messages = mock_messages

    resp = await plugin.generate("just a prompt")
    assert resp["text"] == "response text"
    # The chat was called with a single user message
    call_kwargs = mock_messages.create.await_args.kwargs
    assert call_kwargs["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "just a prompt"}]}
    ]


# ---------------------------------------------------------------------------
# Tools translation helper
# ---------------------------------------------------------------------------


def test_to_anthropic_tools_openai_shape():
    out = _to_anthropic_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "x",
                    "description": "d",
                    "parameters": {"type": "object"},
                },
            }
        ]
    )
    assert out == [{"name": "x", "description": "d", "input_schema": {"type": "object"}}]


def test_to_anthropic_tools_pre_translated_passes_through():
    """If the caller already supplied Anthropic-shape tools, accept them."""
    pre = [{"name": "x", "description": "d", "input_schema": {"type": "object"}}]
    out = _to_anthropic_tools(pre)
    assert out == pre


def test_to_anthropic_tools_handles_missing_parameters_field():
    out = _to_anthropic_tools(
        [{"type": "function", "function": {"name": "x", "description": "d"}}]
    )
    assert out[0]["input_schema"] == {"type": "object", "properties": {}}


# ---------------------------------------------------------------------------
# Plugin metadata
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
async def test_get_model_info():
    plugin = await _make_plugin(model="claude-opus-4-7", max_tokens=8192)
    info = await plugin.get_model_info()
    assert info["provider"] == "Anthropic"
    assert info["default_model"] == "claude-opus-4-7"
    assert info["default_max_tokens"] == 8192


@pytest.mark.asyncio()
async def test_setup_warns_if_no_api_key(caplog):
    import logging
    caplog.set_level(logging.ERROR)
    plugin = AnthropicLLMProviderPlugin()
    await plugin.setup(config={"key_provider": None})
    assert plugin._client is None
    assert any("no API key resolved" in r.message for r in caplog.records)
