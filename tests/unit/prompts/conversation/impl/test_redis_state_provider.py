import logging
from unittest.mock import AsyncMock, patch

import pytest
from genie_tooling.prompts.conversation.impl.redis_state_provider import (
    REDIS_AVAILABLE,
    RedisError,
    RedisStateProviderPlugin,
)

PROVIDER_LOGGER_NAME = "genie_tooling.prompts.conversation.impl.redis_state_provider"

@pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis library not installed")
@pytest.mark.asyncio
async def test_setup_connection_error(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
    mock_redis_client = AsyncMock()
    mock_redis_client.ping.side_effect = RedisError("Connection failed")
    with patch("genie_tooling.prompts.conversation.impl.redis_state_provider.aioredis.from_url", return_value=mock_redis_client):
        provider = RedisStateProviderPlugin()
        await provider.setup(config={"redis_url": "redis://mock"})
    assert provider._redis_client is None
    assert "Failed to connect to Redis: Connection failed" in caplog.text
