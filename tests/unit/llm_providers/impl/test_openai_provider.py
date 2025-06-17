### tests/unit/llm_providers/impl/test_openai_provider.py
import logging
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genie_tooling.llm_providers.impl.openai_provider import (
    APIError,
    AsyncOpenAI,
    OpenAIChatMessage,
    OpenAIChoice,
    OpenAILLMProviderPlugin,
    OpenAIToolCall,
)
from genie_tooling.llm_providers.types import ChatMessage
from genie_tooling.security.key_provider import KeyProvider

PROVIDER_LOGGER_NAME = "genie_tooling.llm_providers.impl.openai_provider"


@pytest.fixture()
def mock_key_provider_openai() -> AsyncMock:
    """Provides a mock KeyProvider that successfully returns a key."""
    kp = AsyncMock(spec=KeyProvider)
    kp.get_key = AsyncMock(return_value="fake-openai-api-key")
    return kp


@pytest.fixture()
def mock_openai_client() -> AsyncMock:
    """Provides a mock AsyncOpenAI client instance."""
    client = AsyncMock(spec=AsyncOpenAI)
    client.chat = AsyncMock()
    client.chat.completions = AsyncMock()
    client.chat.completions.create = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture()
async def openai_provider(
    mock_key_provider_openai: AsyncMock, mock_openai_client: AsyncMock
) -> OpenAILLMProviderPlugin:
    """Provides a fully initialized OpenAILLMProviderPlugin with mocks."""
    provider = OpenAILLMProviderPlugin()
    with patch(
        "genie_tooling.llm_providers.impl.openai_provider.AsyncOpenAI",
        return_value=mock_openai_client,
    ):
        await provider.setup(config={"key_provider": mock_key_provider_openai})
    return provider


@pytest.mark.asyncio()
class TestOpenAILLMProvider:
    async def test_setup_no_api_key(
        self, mock_key_provider: KeyProvider, caplog: pytest.LogCaptureFixture
    ):
        caplog.set_level(logging.INFO, logger=PROVIDER_LOGGER_NAME)
        provider = OpenAILLMProviderPlugin()

        actual_kp = await mock_key_provider

        actual_kp.get_key = AsyncMock(return_value=None)  # type: ignore

        with patch("genie_tooling.llm_providers.impl.openai_provider.AsyncOpenAI"):
            await provider.setup(
                config={"key_provider": actual_kp, "api_key_name": "MISSING_KEY"}
            )
        assert provider._client is None
        assert (
            "API key 'MISSING_KEY' not found. Plugin will be disabled." in caplog.text
        )

    async def test_setup_client_init_fails(
        self, mock_key_provider_openai: AsyncMock, caplog: pytest.LogCaptureFixture
    ):
        caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
        provider = OpenAILLMProviderPlugin()
        with patch(
            "genie_tooling.llm_providers.impl.openai_provider.AsyncOpenAI",
            side_effect=Exception("Connection error"),
        ):
            await provider.setup(config={"key_provider": mock_key_provider_openai})
        assert provider._client is None
        assert "Failed to initialize OpenAI client: Connection error" in caplog.text

    async def test_chat_success(
        self, openai_provider: OpenAILLMProviderPlugin, mock_openai_client: AsyncMock
    ):
        provider = await openai_provider
        mock_openai_response = MagicMock()
        mock_openai_message = MagicMock(spec=OpenAIChatMessage)
        mock_openai_message.content = "Hello from OpenAI"
        mock_openai_message.tool_calls = None
        mock_openai_message.role = "assistant"
        mock_openai_response.choices = [
            MagicMock(spec=OpenAIChoice, message=mock_openai_message, finish_reason="stop")
        ]
        mock_openai_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_openai_client.chat.completions.create.return_value = mock_openai_response

        messages: List[ChatMessage] = [{"role": "user", "content": "Hi"}]
        result = await provider.chat(messages=messages, temperature=0.7)

        mock_openai_client.chat.completions.create.assert_awaited_once()
        assert result["message"]["content"] == "Hello from OpenAI"
        assert result["finish_reason"] == "stop"
        assert result["usage"]["total_tokens"] == 15  # type: ignore

    async def test_chat_with_tool_calls(
        self, openai_provider: OpenAILLMProviderPlugin, mock_openai_client: AsyncMock
    ):
        provider = await openai_provider
        mock_tool_call = MagicMock(spec=OpenAIToolCall)
        mock_tool_call.id = "tc1"
        mock_tool_call.type = "function"

        mock_tool_call.function = MagicMock()
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = '{"city": "Tokyo"}'

        mock_openai_message = MagicMock(spec=OpenAIChatMessage)
        mock_openai_message.content = None
        mock_openai_message.tool_calls = [mock_tool_call]
        mock_openai_message.role = "assistant"

        mock_openai_response = MagicMock()
        mock_openai_response.choices = [
            MagicMock(
                spec=OpenAIChoice,
                message=mock_openai_message,
                finish_reason="tool_calls",
            )
        ]
        mock_openai_response.usage = None
        mock_openai_client.chat.completions.create.return_value = mock_openai_response

        result = await provider.chat(messages=[])

        assert result["finish_reason"] == "tool_calls"
        assert result["message"]["content"] is None
        assert result["message"]["tool_calls"] is not None
        assert len(result["message"]["tool_calls"]) == 1
        assert result["message"]["tool_calls"][0]["function"]["name"] == "get_weather"

    async def test_chat_api_error(
        self, openai_provider: OpenAILLMProviderPlugin, mock_openai_client: AsyncMock
    ):
        provider = await openai_provider

        error_instance = APIError(message="Invalid API key", request=MagicMock(), body=None)
        error_instance.status_code = 401  # type: ignore
        mock_openai_client.chat.completions.create.side_effect = error_instance

        with pytest.raises(RuntimeError, match="OpenAI API Error: 401 - Invalid API key"):
            await provider.chat(messages=[])

    async def test_generate_api_error(
        self, openai_provider: OpenAILLMProviderPlugin, mock_openai_client: AsyncMock
    ):
        provider = await openai_provider

        error_instance = APIError(message="Invalid API key", request=MagicMock(), body=None)
        error_instance.status_code = 401  # type: ignore
        mock_openai_client.chat.completions.create.side_effect = error_instance

        with pytest.raises(RuntimeError, match="OpenAI API Error: 401 - Invalid API key"):
            await provider.generate(prompt="test")

    async def test_teardown(self, openai_provider: OpenAILLMProviderPlugin):
        provider = await openai_provider
        client_mock = provider._client
        assert client_mock is not None
        await provider.teardown()
        client_mock.close.assert_awaited_once()
        assert provider._client is None
