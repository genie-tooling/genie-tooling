"""M5 vision/multimodal ChatMessage tests.

ChatMessage.content can now be either a plain string (legacy, text-only)
or a list of ContentBlocks (M5). Each provider has its own translation:

  * **OpenAI**:    ``image_url`` / ``image_url.url`` (data URL or http URL)
  * **Anthropic**: ``image`` / ``source: url|base64``
  * **Ollama**:    text-only; multimodal collapses to text + placeholder
"""
from __future__ import annotations

from genie_tooling.llm_providers.impl.anthropic_provider import (
    AnthropicLLMProviderPlugin,
)
from genie_tooling.llm_providers.impl.ollama_provider import (
    _collapse_multimodal_content,
)
from genie_tooling.llm_providers.impl.openai_provider import _to_openai_messages
from genie_tooling.llm_providers.types import ChatMessage

# ---------------------------------------------------------------------------
# OpenAI translation
# ---------------------------------------------------------------------------


def test_openai_text_only_message_passes_through():
    msgs: list[ChatMessage] = [{"role": "user", "content": "hi"}]
    out = _to_openai_messages(msgs)
    assert out == [{"role": "user", "content": "hi"}]


def test_openai_multimodal_url_image_translates_correctly():
    msgs: list[ChatMessage] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what's in this picture?"},
                {"type": "image", "source": "url", "url": "https://example.com/cat.jpg"},
            ],
        }
    ]
    out = _to_openai_messages(msgs)
    assert out[0]["role"] == "user"
    parts = out[0]["content"]
    assert parts == [
        {"type": "text", "text": "what's in this picture?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}},
    ]


def test_openai_multimodal_base64_image_becomes_data_url():
    msgs: list[ChatMessage] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": "base64",
                    "data": "AQID",  # bytes 0x01 0x02 0x03 base64
                    "media_type": "image/png",
                }
            ],
        }
    ]
    out = _to_openai_messages(msgs)
    parts = out[0]["content"]
    assert parts == [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AQID"},
        }
    ]


def test_openai_preserves_non_content_fields():
    msgs: list[ChatMessage] = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "let me call a tool"}],
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "x", "arguments": "{}"}}
            ],
        }
    ]
    out = _to_openai_messages(msgs)
    assert out[0]["role"] == "assistant"
    assert out[0]["tool_calls"] == [
        {"id": "call_1", "type": "function", "function": {"name": "x", "arguments": "{}"}}
    ]


def test_openai_unknown_block_type_falls_back_to_json_text():
    msgs: list[ChatMessage] = [
        {
            "role": "user",
            "content": [{"type": "weird", "data": "anything"}],
        }
    ]
    out = _to_openai_messages(msgs)
    parts = out[0]["content"]
    assert parts[0]["type"] == "text"
    assert "weird" in parts[0]["text"]


# ---------------------------------------------------------------------------
# Anthropic translation (already covered in test_anthropic_provider.py,
# but re-asserting M5-specific behavior here for completeness)
# ---------------------------------------------------------------------------


def test_anthropic_multimodal_round_trip():
    msgs: list[ChatMessage] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {
                    "type": "image",
                    "source": "base64",
                    "data": "AQID",
                    "media_type": "image/jpeg",
                },
            ],
        }
    ]
    out = AnthropicLLMProviderPlugin._to_anthropic_messages(msgs)
    blocks = out[0]["content"]
    assert blocks[0] == {"type": "text", "text": "describe"}
    assert blocks[1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": "AQID"},
    }


# ---------------------------------------------------------------------------
# Ollama text-only collapse
# ---------------------------------------------------------------------------


def test_ollama_collapses_multimodal_to_text_with_placeholder():
    msgs: list[ChatMessage] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what's in this?"},
                {"type": "image", "source": "url", "url": "https://x/y.png"},
                {"type": "text", "text": "be brief"},
            ],
        }
    ]
    out = _collapse_multimodal_content(msgs)
    assert out[0]["role"] == "user"
    content = out[0]["content"]
    assert isinstance(content, str)
    assert "what's in this?" in content
    assert "image elided" in content
    assert "be brief" in content


def test_ollama_collapse_preserves_text_only_messages_unchanged():
    msgs: list[ChatMessage] = [{"role": "user", "content": "plain text"}]
    out = _collapse_multimodal_content(msgs)
    assert out == msgs


def test_ollama_collapse_preserves_extra_fields():
    msgs: list[ChatMessage] = [
        {
            "role": "tool",
            "content": [{"type": "text", "text": "result"}],
            "tool_call_id": "call_1",
        }
    ]
    out = _collapse_multimodal_content(msgs)
    assert out[0]["role"] == "tool"
    assert out[0]["tool_call_id"] == "call_1"
    assert out[0]["content"] == "result"
