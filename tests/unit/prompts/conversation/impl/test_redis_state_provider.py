import json
import logging
from unittest.mock import AsyncMock, patch

import pytest
from genie_tooling.prompts.conversation.impl.redis_state_provider import (
    REDIS_AVAILABLE,
    RedisError,  # Assuming this is imported or mocked if redis lib not present
    RedisStateProviderPlugin,
)
from genie_tooling.prompts.conversation.types import ConversationState

PROVIDER_LOGGER_NAME = "genie_tooling.prompts.conversation.impl.redis_state_provider"

@pytest.fixture
def mock_aioredis_client() -> AsyncMock:
    client = AsyncMock(name="MockAioRedisClient")
    client.ping = AsyncMock(return_value=True)
    client.get = AsyncMock(return_value=None)
    client.set = AsyncMock(return_value=True)
    client.delete = AsyncMock(return_value=0)
    client.close = AsyncMock()
    return client

@pytest.fixture
async def redis_provider(mock_aioredis_client: AsyncMock) -> RedisStateProviderPlugin:
    with patch("genie_tooling.prompts.conversation.impl.redis_state_provider.aioredis.from_url", return_value=mock_aioredis_client):
        provider = RedisStateProviderPlugin()
        await provider.setup(config={"redis_url": "redis://mock"})
        # Reset mocks for each test after setup
        mock_aioredis_client.get.reset_mock(return_value=None)
        mock_aioredis_client.set.reset_mock(return_value=True)
        mock_aioredis_client.delete.reset_mock(return_value=0)
        return provider

@pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis library not installed")
@pytest.mark.asyncio
async def test_setup_success(mock_aioredis_client: AsyncMock):
    with patch("genie_tooling.prompts.conversation.impl.redis_state_provider.aioredis.from_url", return_value=mock_aioredis_client) as mock_from_url:
        provider = RedisStateProviderPlugin()
        await provider.setup(config={"redis_url": "redis://testhost:1234/1", "key_prefix": "test_cs:", "default_ttl_seconds": 3600})
        mock_from_url.assert_called_once_with("redis://testhost:1234/1")
        mock_aioredis_client.ping.assert_awaited_once()
        assert provider._redis_client is mock_aioredis_client
        assert provider._key_prefix == "test_cs:"
        assert provider._default_ttl_seconds == 3600

@pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis library not installed")
@pytest.mark.asyncio
async def test_setup_connection_error(mock_aioredis_client: AsyncMock, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
    mock_aioredis_client.ping.side_effect = RedisError("Connection failed")
    with patch("genie_tooling.prompts.conversation.impl.redis_state_provider.aioredis.from_url", return_value=mock_aioredis_client):
        provider = RedisStateProviderPlugin()
        await provider.setup()
    assert provider._redis_client is None
    assert "Failed to connect to Redis: Connection failed" in caplog.text

@pytest.mark.asyncio
async def test_redis_not_available(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
    with patch("genie_tooling.prompts.conversation.impl.redis_state_provider.REDIS_AVAILABLE", False):
        provider_no_redis = RedisStateProviderPlugin()
        await provider_no_redis.setup()
    assert provider_no_redis._redis_client is None
    assert "'redis' library (>=4.2) not installed. This plugin will not function." in caplog.text

@pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis library not installed")
@pytest.mark.asyncio
async def test_load_state_success(redis_provider: RedisStateProviderPlugin, mock_aioredis_client: AsyncMock):
    provider = await redis_provider
    session_id = "s1"
    expected_state: ConversationState = {"session_id": session_id, "history": [{"role":"user", "content":"Hi"}], "metadata": {"k":"v"}}
    mock_aioredis_client.get.return_value = json.dumps(expected_state).encode("utf-8")

    loaded_state = await provider.load_state(session_id)
    mock_aioredis_client.get.assert_awaited_once_with(provider._get_redis_key(session_id))
    assert loaded_state == expected_state

@pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis library not installed")
@pytest.mark.asyncio
async def test_load_state_not_found(redis_provider: RedisStateProviderPlugin, mock_aioredis_client: AsyncMock):
    provider = await redis_provider
    mock_aioredis_client.get.return_value = None
    loaded_state = await provider.load_state("s_not_found")
    assert loaded_state is None

@pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis library not installed")
@pytest.mark.asyncio
async def test_load_state_json_decode_error(redis_provider: RedisStateProviderPlugin, mock_aioredis_client: AsyncMock, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
    provider = await redis_provider
    mock_aioredis_client.get.return_value = b"not valid json"
    loaded_state = await provider.load_state("s_json_err")
    assert loaded_state is None
    assert "Failed to JSON decode state from Redis" in caplog.text

@pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis library not installed")
@pytest.mark.asyncio
async def test_save_state_success(redis_provider: RedisStateProviderPlugin, mock_aioredis_client: AsyncMock):
    provider = await redis_provider
    state_to_save: ConversationState = {"session_id": "s_save", "history": [], "metadata": {}}
    await provider.save_state(state_to_save)
    mock_aioredis_client.set.assert_awaited_once_with(
        provider._get_redis_key("s_save"),
        json.dumps(state_to_save).encode("utf-8"),
        ex=provider._default_ttl_seconds
    )

@pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis library not installed")
@pytest.mark.asyncio
async def test_save_state_redis_error(redis_provider: RedisStateProviderPlugin, mock_aioredis_client: AsyncMock, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
    provider = await redis_provider
    mock_aioredis_client.set.side_effect = RedisError("SET failed")
    await provider.save_state({"session_id": "s_save_err", "history": []})
    assert "Redis SET error" in caplog.text

@pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis library not installed")
@pytest.mark.asyncio
async def test_delete_state_success(redis_provider: RedisStateProviderPlugin, mock_aioredis_client: AsyncMock):
    provider = await redis_provider
    mock_aioredis_client.delete.return_value = 1 # Simulate one key deleted
    result = await provider.delete_state("s_del")
    assert result is True
    mock_aioredis_client.delete.assert_awaited_once_with(provider._get_redis_key("s_del"))

@pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis library not installed")
@pytest.mark.asyncio
async def test_operations_if_client_none(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
    provider_no_client = RedisStateProviderPlugin() # Setup will fail to init client
    await provider_no_client.setup(config={"redis_url": "redis://nonexistent"}) # Force setup fail

    assert await provider_no_client.load_state("s1") is None
    assert "Redis client not available" in caplog.text
    caplog.clear()

    await provider_no_client.save_state({"session_id": "s1", "history": []})
    assert "Redis client not available" in caplog.text
    caplog.clear()

    assert await provider_no_client.delete_state("s1") is False
    assert "Redis client not available" in caplog.text

@pytest.mark.skipif(not REDIS_AVAILABLE, reason="redis library not installed")
@pytest.mark.asyncio
async def test_teardown(redis_provider: RedisStateProviderPlugin, mock_aioredis_client: AsyncMock):
    provider = await redis_provider
    await provider.teardown()
    mock_aioredis_client.close.assert_awaited_once()
    assert provider._redis_client is None
