"""Unit tests for M4 native structured outputs across providers."""
from __future__ import annotations

from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, Field

from genie_tooling.llm_providers.impl.openai_provider import (
    _build_openai_response_format,
    _make_schema_openai_strict,
)


class Person(BaseModel):
    name: str
    age: int
    favorite_color: Optional[str] = None


class Address(BaseModel):
    street: str
    city: str


class PersonWithAddress(BaseModel):
    name: str
    address: Address


def test_make_schema_openai_strict_adds_additional_properties_false():
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    }
    _make_schema_openai_strict(schema)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"name", "age"}


def test_make_schema_openai_strict_recurses_into_nested_objects():
    schema = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {"inner": {"type": "string"}},
            }
        },
    }
    _make_schema_openai_strict(schema)
    inner_schema = schema["properties"]["outer"]
    assert inner_schema["additionalProperties"] is False
    assert inner_schema["required"] == ["inner"]


def test_make_schema_openai_strict_handles_array_items():
    schema = {
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            }
        },
    }
    _make_schema_openai_strict(schema)
    assert schema["properties"]["tags"]["items"]["additionalProperties"] is False


def test_make_schema_openai_strict_handles_pydantic_defs():
    """Pydantic generates `$defs` for nested models; strict-mode must reach
    in there too."""
    schema = {
        "type": "object",
        "properties": {"x": {"$ref": "#/$defs/Inner"}},
        "$defs": {
            "Inner": {
                "type": "object",
                "properties": {"y": {"type": "string"}},
            }
        },
    }
    _make_schema_openai_strict(schema)
    assert schema["$defs"]["Inner"]["additionalProperties"] is False
    assert schema["$defs"]["Inner"]["required"] == ["y"]


def test_build_openai_response_format_for_simple_model():
    fmt = _build_openai_response_format(Person)
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "Person"
    assert fmt["json_schema"]["strict"] is True
    schema = fmt["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    # All three fields should be in required (strict mode mandates this)
    assert set(schema["required"]) == {"name", "age", "favorite_color"}


def test_build_openai_response_format_for_nested_model():
    fmt = _build_openai_response_format(PersonWithAddress)
    schema = fmt["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    # The nested Address must also be strict
    defs = schema.get("$defs", {})
    if defs:
        assert defs["Address"]["additionalProperties"] is False


def test_build_openai_response_format_rejects_non_basemodel():
    with pytest.raises(ValueError, match="subclass of pydantic.BaseModel"):
        _build_openai_response_format(dict)


@pytest.mark.asyncio
async def test_openai_chat_includes_response_format_when_schema_supplied():
    """When the caller passes response_schema=PersonModel, the OpenAI
    provider must construct the response_format kwarg and pass it to the
    SDK."""
    from openai.types.chat import ChatCompletion
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    from genie_tooling.llm_providers.impl.openai_provider import (
        OpenAILLMProviderPlugin,
    )

    plugin = OpenAILLMProviderPlugin()
    plugin._model_name = "gpt-4o"
    plugin._client = MagicMock()
    plugin._client.chat = MagicMock()
    plugin._client.chat.completions = MagicMock()

    # Mock the OpenAI response object
    mock_msg = MagicMock(spec=ChatCompletionMessage)
    mock_msg.content = '{"name": "Alice", "age": 30, "favorite_color": "blue"}'
    mock_msg.tool_calls = None
    mock_msg.role = "assistant"
    mock_choice = MagicMock(spec=Choice)
    mock_choice.message = mock_msg
    mock_choice.finish_reason = "stop"
    mock_resp = MagicMock(spec=ChatCompletion)
    mock_resp.choices = [mock_choice]
    mock_resp.usage = None
    mock_resp.model_dump = MagicMock(return_value={})

    plugin._client.chat.completions.create = AsyncMock(return_value=mock_resp)

    await plugin.chat(
        messages=[{"role": "user", "content": "Make a person"}],
        response_schema=Person,
    )

    call_kwargs = plugin._client.chat.completions.create.await_args.kwargs
    assert "response_format" in call_kwargs
    rf = call_kwargs["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "Person"
    assert rf["json_schema"]["strict"] is True


@pytest.mark.asyncio
async def test_openai_chat_omits_response_format_when_no_schema():
    """No response_schema supplied -> no response_format key in the API call."""
    from genie_tooling.llm_providers.impl.openai_provider import (
        OpenAILLMProviderPlugin,
    )

    plugin = OpenAILLMProviderPlugin()
    plugin._model_name = "gpt-4o"
    plugin._client = MagicMock()
    plugin._client.chat = MagicMock()
    plugin._client.chat.completions = MagicMock()
    mock_msg = MagicMock()
    mock_msg.content = "hi"
    mock_msg.tool_calls = None
    mock_msg.role = "assistant"
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_choice.finish_reason = "stop"
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.usage = None
    mock_resp.model_dump = MagicMock(return_value={})
    plugin._client.chat.completions.create = AsyncMock(return_value=mock_resp)

    await plugin.chat(messages=[{"role": "user", "content": "hi"}])
    call_kwargs = plugin._client.chat.completions.create.await_args.kwargs
    assert "response_format" not in call_kwargs
