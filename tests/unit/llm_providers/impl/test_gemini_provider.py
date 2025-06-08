import json
import logging
from typing import Any, AsyncIterable, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genie_tooling.llm_providers.impl.gemini_provider import (
    GEMINI_ROLE_MAP,
    GeminiLLMProviderPlugin,
)
from genie_tooling.llm_providers.types import ChatMessage, LLMChatResponse
from genie_tooling.security.key_provider import KeyProvider

PROVIDER_LOGGER_NAME = "genie_tooling.llm_providers.impl.gemini_provider"


@pytest.fixture
def mock_genai_lib():
    """Mocks the entire google.generativeai library."""
    with patch("genie_tooling.llm_providers.impl.gemini_provider.genai") as mock_lib:
        # Mock the classes and functions used by the provider
        mock_lib.GenerativeModel = MagicMock(name="MockGenerativeModelClass")
        mock_lib.configure = MagicMock(name="MockGenaiConfigure")
        mock_lib.get_model = MagicMock(name="MockGenaiGetModel")
        # Mock the response types for isinstance checks and attribute access
        mock_lib.types.GenerateContentResponse = MagicMock(
            name="MockGenerateContentResponse"
        )
        mock_lib.types.AsyncGenerateContentResponse = MagicMock(
            name="MockAsyncGenerateContentResponse"
        )
        yield mock_lib


@pytest.fixture
async def gemini_provider(
    mock_genai_lib: MagicMock, mock_key_provider: KeyProvider
) -> GeminiLLMProviderPlugin:
    """Provides an initialized GeminiLLMProviderPlugin with mocked dependencies."""
    provider = GeminiLLMProviderPlugin()
    kp = await mock_key_provider
    # Mock the model client instance that will be created inside setup
    mock_model_client_instance = AsyncMock(name="MockGenerativeModelInstance")
    mock_model_client_instance.generate_content_async = AsyncMock()
    mock_genai_lib.GenerativeModel.return_value = mock_model_client_instance
    await provider.setup(config={"key_provider": kp})
    return provider


@pytest.mark.asyncio
async def test_gemini_setup_no_api_key(
    mock_genai_lib: MagicMock,
    caplog: pytest.LogCaptureFixture,
    mock_key_provider: KeyProvider,
):
    """Test setup fails gracefully when the API key is not found."""
    caplog.set_level(logging.INFO, logger=PROVIDER_LOGGER_NAME)
    provider = GeminiLLMProviderPlugin()
    actual_kp = await mock_key_provider
    actual_kp.get_key = AsyncMock(return_value=None)  # type: ignore

    await provider.setup(
        config={"api_key_name": "ANY_KEY_NAME_HERE", "key_provider": actual_kp}
    )
    assert provider._model_client is None
    assert (
        "API key 'ANY_KEY_NAME_HERE' not found. Plugin will be disabled."
        in caplog.text
    )


@pytest.mark.asyncio
async def test_gemini_setup_client_init_fails(
    mock_genai_lib: MagicMock,
    caplog: pytest.LogCaptureFixture,
    mock_key_provider: KeyProvider,
):
    """Test setup fails gracefully when the Gemini client constructor raises an error."""
    caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
    provider = GeminiLLMProviderPlugin()
    kp = await mock_key_provider
    mock_genai_lib.GenerativeModel.side_effect = ValueError("Invalid model name")

    await provider.setup(config={"key_provider": kp})
    assert provider._model_client is None
    assert "Failed to initialize Gemini client: Invalid model name" in caplog.text


@pytest.mark.asyncio
class TestGeminiMessageConversion:
    """Tests for the _convert_messages_to_gemini method."""

    async def test_convert_user_and_assistant_roles(
        self, gemini_provider: GeminiLLMProviderPlugin
    ):
        provider = await gemini_provider
        messages: List[ChatMessage] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        gemini_msgs = provider._convert_messages_to_gemini(messages)
        assert len(gemini_msgs) == 2
        assert gemini_msgs[0]["role"] == "user"
        assert gemini_msgs[0]["parts"][0]["text"] == "Hello"
        assert gemini_msgs[1]["role"] == "model"
        assert gemini_msgs[1]["parts"][0]["text"] == "Hi there!"

    async def test_convert_system_role(self, gemini_provider: GeminiLLMProviderPlugin):
        provider = await gemini_provider
        messages: List[ChatMessage] = [
            {"role": "system", "content": "You are a helpful bot."}
        ]
        gemini_msgs = provider._convert_messages_to_gemini(messages)
        assert len(gemini_msgs) == 1
        assert gemini_msgs[0]["role"] == "user"  # System role is mapped to user
        assert gemini_msgs[0]["parts"][0]["text"] == "You are a helpful bot."

    async def test_convert_tool_call_request(
        self, gemini_provider: GeminiLLMProviderPlugin
    ):
        provider = await gemini_provider
        messages: List[ChatMessage] = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "London"}',
                        },
                    }
                ],
            }
        ]
        gemini_msgs = provider._convert_messages_to_gemini(messages)
        assert len(gemini_msgs) == 1
        assert gemini_msgs[0]["role"] == "model"
        assert "function_call" in gemini_msgs[0]["parts"][0]
        assert gemini_msgs[0]["parts"][0]["function_call"]["name"] == "get_weather"
        assert gemini_msgs[0]["parts"][0]["function_call"]["args"] == {"city": "London"}

    async def test_convert_tool_response(
        self, gemini_provider: GeminiLLMProviderPlugin
    ):
        provider = await gemini_provider
        messages: List[ChatMessage] = [
            {
                "role": "tool",
                "tool_call_id": "tc1",
                "name": "get_weather",
                "content": '{"temperature": 15, "unit": "celsius"}',
            }
        ]
        gemini_msgs = provider._convert_messages_to_gemini(messages)
        assert len(gemini_msgs) == 1
        assert gemini_msgs[0]["role"] == "tool"
        assert "function_response" in gemini_msgs[0]["parts"][0]
        assert gemini_msgs[0]["parts"][0]["function_response"]["name"] == "get_weather"
        assert gemini_msgs[0]["parts"][0]["function_response"]["response"] == {
            "temperature": 15,
            "unit": "celsius",
        }

    async def test_convert_tool_message_no_content(
        self, gemini_provider: GeminiLLMProviderPlugin, caplog: pytest.LogCaptureFixture
    ):
        """Test that a tool message with no 'content' key is handled correctly."""
        provider = await gemini_provider
        caplog.set_level(logging.DEBUG, logger=PROVIDER_LOGGER_NAME)
        messages: List[ChatMessage] = [
            {"role": "tool", "tool_call_id": "tc1", "name": "tool_name"}
        ]
        gemini_msgs = provider._convert_messages_to_gemini(messages)
        assert len(gemini_msgs) == 1
        assert gemini_msgs[0]["role"] == "tool"
        assert "function_response" in gemini_msgs[0]["parts"][0]
        # The provider should create a default response payload
        assert gemini_msgs[0]["parts"][0]["function_response"]["response"] == {
            "output": None
        }
        assert "Processed tool message" in caplog.text

    async def test_convert_tool_response_non_json_content(
        self, gemini_provider: GeminiLLMProviderPlugin
    ):
        """Test that a tool message with non-JSON string content is wrapped."""
        provider = await gemini_provider
        messages: List[ChatMessage] = [
            {
                "role": "tool",
                "tool_call_id": "tc1",
                "name": "get_status",
                "content": "OK",
            }
        ]
        gemini_msgs = provider._convert_messages_to_gemini(messages)
        assert len(gemini_msgs) == 1
        assert gemini_msgs[0]["role"] == "tool"
        assert "function_response" in gemini_msgs[0]["parts"][0]
        assert gemini_msgs[0]["parts"][0]["function_response"]["response"] == {
            "output": "OK"
        }

    async def test_convert_assistant_message_no_content_no_tool_calls(
        self, gemini_provider: GeminiLLMProviderPlugin, caplog: pytest.LogCaptureFixture
    ):
        """Test that an assistant message with no content or tool calls is skipped."""
        provider = await gemini_provider
        caplog.set_level(logging.WARNING, logger=PROVIDER_LOGGER_NAME)
        messages: List[ChatMessage] = [{"role": "assistant"}]
        gemini_msgs = provider._convert_messages_to_gemini(messages)
        assert len(gemini_msgs) == 0
        assert "resulted in empty parts. Skipping." in caplog.text


@pytest.mark.asyncio
async def test_generate_api_error(gemini_provider: GeminiLLMProviderPlugin):
    """Test that API errors during generate are caught and re-raised."""
    provider = await gemini_provider
    provider._model_client.generate_content_async.side_effect = Exception(  # type: ignore
        "Gemini API is down"
    )
    with pytest.raises(RuntimeError, match="Gemini API call failed: Gemini API is down"):
        await provider.generate("test prompt")


@pytest.mark.asyncio
async def test_chat_api_error(gemini_provider: GeminiLLMProviderPlugin):
    """Test that API errors during chat are caught and re-raised."""
    provider = await gemini_provider
    provider._model_client.generate_content_async.side_effect = Exception(  # type: ignore
        "Gemini Chat API is down"
    )
    with pytest.raises(
        RuntimeError, match="Gemini API call failed: Gemini Chat API is down"
    ):
        await provider.chat([{"role": "user", "content": "test"}])


@pytest.mark.asyncio
async def test_chat_streaming_success(gemini_provider: GeminiLLMProviderPlugin):
    """Test successful streaming chat response with simple text."""
    provider = await gemini_provider

    async def mock_stream():
        # Mock Gemini's async response stream for a simple text chat
        mock_chunk1 = MagicMock()
        mock_part1 = MagicMock(text="Hello ")
        # **FIX**: Ensure the mock part does NOT have a function_call attribute
        del mock_part1.function_call
        type(mock_chunk1).candidates = [MagicMock(content=MagicMock(parts=[mock_part1]))]
        yield mock_chunk1

        mock_chunk2 = MagicMock()
        mock_part2 = MagicMock(text="World!")
        del mock_part2.function_call
        type(mock_chunk2).candidates = [MagicMock(content=MagicMock(parts=[mock_part2]))]
        yield mock_chunk2

        mock_chunk3 = MagicMock()
        mock_candidate3 = MagicMock()
        mock_candidate3.finish_reason.value = 1  # Corresponds to "stop"
        type(mock_chunk3).candidates = [mock_candidate3]
        type(mock_chunk3).usage_metadata = MagicMock(
            prompt_token_count=10, candidates_token_count=5, total_token_count=15
        )
        yield mock_chunk3

    provider._model_client.generate_content_async.return_value = mock_stream()  # type: ignore

    result_stream = await provider.chat(
        [{"role": "user", "content": "test"}], stream=True
    )
    chunks = [chunk async for chunk in result_stream]

    assert len(chunks) == 3
    assert chunks[0]["message_delta"]["content"] == "Hello "
    assert chunks[1]["message_delta"]["content"] == "World!"
    assert chunks[2]["finish_reason"] == "stop"
    assert chunks[2]["usage_delta"]["total_tokens"] == 15


@pytest.mark.asyncio
async def test_chat_streaming_with_tool_calls(gemini_provider: GeminiLLMProviderPlugin):
    """Test streaming chat response that includes tool calls."""
    provider = await gemini_provider

    async def mock_tool_call_stream():
        # Chunk 1: Start of a tool call
        mock_fc1 = MagicMock()
        type(mock_fc1).name = "get_weather"
        # **FIX**: fc.args must be a dict, not a MagicMock, to be JSON serializable
        type(mock_fc1).args = {"city": "Lon"}
        mock_part1 = MagicMock(function_call=mock_fc1)
        del mock_part1.text  # A tool call part won't have a text attribute
        mock_chunk1 = MagicMock()
        type(mock_chunk1).candidates = [MagicMock(content=MagicMock(parts=[mock_part1]))]
        yield mock_chunk1

        # Chunk 2: Continuation of the tool call arguments
        mock_fc2 = MagicMock()
        type(mock_fc2).name = None  # Name might not be in subsequent chunks
        type(mock_fc2).args = {"city": "London"}  # Simulate args being built up
        mock_part2 = MagicMock(function_call=mock_fc2)
        del mock_part2.text
        mock_chunk2 = MagicMock()
        type(mock_chunk2).candidates = [MagicMock(content=MagicMock(parts=[mock_part2]))]
        yield mock_chunk2

        # Chunk 3: Finish reason
        mock_chunk3 = MagicMock()
        mock_candidate3 = MagicMock()
        mock_candidate3.finish_reason.value = 6  # Corresponds to "tool_calls"
        type(mock_chunk3).candidates = [mock_candidate3]
        yield mock_chunk3

    provider._model_client.generate_content_async.return_value = mock_tool_call_stream()  # type: ignore

    result_stream = await provider.chat(
        [{"role": "user", "content": "test"}], stream=True
    )
    chunks = [chunk async for chunk in result_stream]

    assert len(chunks) == 3
    assert chunks[0]["message_delta"]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert '"city": "Lon"' in chunks[0]["message_delta"]["tool_calls"][0]["function"]["arguments"]
    assert '"city": "London"' in chunks[1]["message_delta"]["tool_calls"][0]["function"]["arguments"]
    assert chunks[2]["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_chat_blocked_response(gemini_provider: GeminiLLMProviderPlugin):
    """Test handling of a response blocked for safety reasons."""
    provider = await gemini_provider
    mock_response = MagicMock()
    type(mock_response).candidates = []  # No candidates
    mock_prompt_feedback = MagicMock()
    mock_prompt_feedback.block_reason.name = "SAFETY"
    type(mock_response).prompt_feedback = mock_prompt_feedback
    provider._model_client.generate_content_async.return_value = mock_response  # type: ignore

    result: LLMChatResponse = await provider.chat([{"role": "user", "content": "risky"}])

    assert result["finish_reason"] == "blocked: SAFETY"
    assert "[Chat blocked: SAFETY]" in result["message"]["content"]


@pytest.mark.asyncio
async def test_get_model_info_success(
    gemini_provider: GeminiLLMProviderPlugin, mock_genai_lib: MagicMock
):
    """Test successful retrieval of model info."""
    provider = await gemini_provider
    mock_model_info = MagicMock()
    type(mock_model_info).display_name = "Gemini 1.5 Flash"
    type(mock_model_info).version = "1.0"
    type(mock_model_info).input_token_limit = 1048576
    mock_genai_lib.get_model.return_value = mock_model_info

    info = await provider.get_model_info()

    assert info["provider"] == "Google Gemini"
    assert info["display_name"] == "Gemini 1.5 Flash"
    assert info["version"] == "1.0"
    assert info["input_token_limit"] == 1048576


@pytest.mark.asyncio
async def test_get_model_info_api_fails(
    gemini_provider: GeminiLLMProviderPlugin, mock_genai_lib: MagicMock
):
    """Test handling of API failure during model info retrieval."""
    provider = await gemini_provider
    mock_genai_lib.get_model.side_effect = Exception("API call failed")

    info = await provider.get_model_info()

    assert "model_info_error" in info
    assert "API call failed" in info["model_info_error"]