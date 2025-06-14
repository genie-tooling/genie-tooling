import logging
from unittest.mock import AsyncMock, patch

import pytest
from genie_tooling.llm_providers.impl.openai_provider import OpenAILLMProviderPlugin
from genie_tooling.security.key_provider import KeyProvider

PROVIDER_LOGGER_NAME = "genie_tooling.llm_providers.impl.openai_provider"

@pytest.mark.asyncio()
async def test_openai_setup_no_api_key(mock_key_provider: KeyProvider, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger=PROVIDER_LOGGER_NAME)
    provider = OpenAILLMProviderPlugin()
    actual_kp = await mock_key_provider
    actual_kp.get_key = AsyncMock(return_value=None) # type: ignore

    with patch("genie_tooling.llm_providers.impl.openai_provider.AsyncOpenAI"):
        await provider.setup(config={"key_provider": actual_kp, "api_key_name": "MISSING_KEY"})
    assert provider._client is None
    assert "API key 'MISSING_KEY' not found. Plugin will be disabled." in caplog.text
