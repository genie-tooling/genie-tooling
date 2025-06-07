import logging
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genie_tooling.llm_providers.impl.gemini_provider import GeminiLLMProviderPlugin
from genie_tooling.llm_providers.types import ChatMessage
from genie_tooling.security.key_provider import KeyProvider

PROVIDER_LOGGER_NAME = "genie_tooling.llm_providers.impl.gemini_provider"

@pytest.fixture
def mock_genai_lib():
    with patch("genie_tooling.llm_providers.impl.gemini_provider.genai") as mock_lib:
        yield mock_lib

@pytest.fixture
async def gemini_provider(mock_genai_lib: MagicMock, mock_key_provider: KeyProvider) -> GeminiLLMProviderPlugin:
    provider = GeminiLLMProviderPlugin()
    kp = await mock_key_provider
    await provider.setup(config={"key_provider": kp})
    return provider

@pytest.mark.asyncio
async def test_gemini_setup_no_api_key(mock_genai_lib: MagicMock, caplog: pytest.LogCaptureFixture, mock_key_provider: KeyProvider):
    caplog.set_level(logging.INFO, logger=PROVIDER_LOGGER_NAME)
    provider = GeminiLLMProviderPlugin()
    actual_kp = await mock_key_provider
    actual_kp.get_key = AsyncMock(return_value=None) # type: ignore

    await provider.setup(config={"api_key_name": "ANY_KEY_NAME_HERE", "key_provider": actual_kp})
    assert provider._model_client is None
    assert "API key 'ANY_KEY_NAME_HERE' not found. Plugin will be disabled." in caplog.text

@pytest.mark.asyncio
async def test_gemini_convert_messages_tool_message_no_content(gemini_provider: GeminiLLMProviderPlugin, caplog: pytest.LogCaptureFixture):
    provider = await gemini_provider
    caplog.set_level(logging.DEBUG)
    messages: List[ChatMessage] = [
        {"role": "tool", "tool_call_id": "tc1", "name": "tool_name"}
    ]
    gemini_msgs = provider._convert_messages_to_gemini(messages)
    assert len(gemini_msgs) == 1
    assert gemini_msgs[0]["role"] == "tool"
    assert gemini_msgs[0]["parts"][0]["function_response"]["response"] == {"output": None}
    assert "Processed tool message" in caplog.text
