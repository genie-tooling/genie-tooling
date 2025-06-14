### tests/unit/caching/impl/test_redis_cache.py
import json
import logging
from unittest.mock import AsyncMock, patch

import pytest
from genie_tooling.cache_providers.impl.redis_cache import (
    RedisCacheProvider,
    RedisConnectionError,
    RedisError,
)

PROVIDER_LOGGER_NAME = "genie_tooling.cache_providers.impl.redis_cache"


@pytest.fixture()
def mock_redis_client() -> AsyncMock:
    client = AsyncMock(spec=["ping", "get", "set", "delete", "exists", "flushdb", "close"])
    client.ping = AsyncMock(return_value=True)
    client.get = AsyncMock(return_value=None)
    client.set = AsyncMock(return_value=True)
    client.delete = AsyncMock(return_value=0)
    client.exists = AsyncMock(return_value=0)
    client.flushdb = AsyncMock(return_value=True)
    client.close = AsyncMock()
    return client


@pytest.fixture()
async def redis_cache_provider(mock_redis_client: AsyncMock) -> RedisCacheProvider:
    provider = RedisCacheProvider()
    with patch("genie_tooling.cache_providers.impl.redis_cache.aioredis.from_url", return_value=mock_redis_client):
        await provider.setup(config={"redis_url": "redis://mock"})
    # Reset mocks after setup to ensure clean state for each test
    mock_redis_client.get.reset_mock()
    mock_redis_client.set.reset_mock()
    mock_redis_client.delete.reset_mock()
    mock_redis_client.exists.reset_mock()
    mock_redis_client.flushdb.reset_mock()
    return provider


@pytest.mark.asyncio()
async def test_redis_cache_get_hit_json(redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    provider = await redis_cache_provider
    key = "test_key_json"
    expected_value = {"data": "value", "count": 1}
    mock_redis_client.get.return_value = json.dumps(expected_value)
    retrieved_value = await provider.get(key)
    mock_redis_client.get.assert_awaited_once_with(key)
    assert retrieved_value == expected_value


@pytest.mark.asyncio()
async def test_redis_cache_get_hit_string(redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    provider = await redis_cache_provider
    provider._json_serialization = False
    key = "test_key_string"
    expected_value = "simple string value"
    mock_redis_client.get.return_value = expected_value
    retrieved_value = await provider.get(key)
    mock_redis_client.get.assert_awaited_once_with(key)
    assert retrieved_value == expected_value


@pytest.mark.asyncio()
async def test_redis_cache_get_json_decode_error(redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
    provider = await redis_cache_provider
    key = "decode_error_key"
    malformed_json_string = '{"data": "value", "count": 1'
    mock_redis_client.get.return_value = malformed_json_string
    retrieved_value = await provider.get(key)
    assert retrieved_value == malformed_json_string
    assert "Failed to JSON decode for key 'decode_error_key'" in caplog.text


@pytest.mark.asyncio()
async def test_redis_cache_get_raises_redis_error(redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
    provider = await redis_cache_provider
    mock_redis_client.get.side_effect = RedisError("GET command failed")
    result = await provider.get("any_key")
    assert result is None
    assert "Redis GET error for 'any_key'" in caplog.text


@pytest.mark.asyncio()
async def test_redis_cache_set_json_value(redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    provider = await redis_cache_provider
    key = "set_json_key"
    value = {"complex": True, "items": [1, "a"]}
    expected_stored_value = json.dumps(value)
    await provider.set(key, value)
    mock_redis_client.set.assert_awaited_once_with(key, expected_stored_value, ex=None)


@pytest.mark.asyncio()
async def test_redis_cache_set_string_value_no_json_serialize(redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    provider = await redis_cache_provider
    provider._json_serialization = False
    key = "set_string_key_no_json"
    value = "plain string"
    await provider.set(key, value)
    mock_redis_client.set.assert_awaited_once_with(key, value, ex=None)


@pytest.mark.asyncio()
async def test_redis_cache_set_string_value_with_json_serialize(redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    provider = await redis_cache_provider
    key = "set_string_key_json_true"
    value = "plain string to store"
    await provider.set(key, value)
    mock_redis_client.set.assert_awaited_once_with(key, value, ex=None)


@pytest.mark.asyncio()
async def test_redis_cache_set_with_ttl(redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    provider = await redis_cache_provider
    key = "set_ttl_key"
    value = "data_with_ttl"
    ttl = 300
    await provider.set(key, value, ttl_seconds=ttl)
    mock_redis_client.set.assert_awaited_once_with(key, value, ex=ttl)


@pytest.mark.asyncio()
async def test_redis_cache_set_raises_redis_error(redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
    provider = await redis_cache_provider
    mock_redis_client.set.side_effect = RedisError("SET command failed")
    await provider.set("any_key", "any_value")
    assert "Redis SET error for 'any_key'" in caplog.text


@pytest.mark.asyncio()
async def test_redis_cache_delete_existing(redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    provider = await redis_cache_provider
    key = "delete_me"
    mock_redis_client.delete.return_value = 1
    result = await provider.delete(key)
    mock_redis_client.delete.assert_awaited_once_with(key)
    assert result is True


@pytest.mark.asyncio()
async def test_redis_cache_delete_non_existing(redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    provider = await redis_cache_provider
    key = "i_dont_exist"
    mock_redis_client.delete.return_value = 0
    result = await provider.delete(key)
    assert result is False


@pytest.mark.asyncio()
async def test_redis_cache_delete_raises_redis_error(redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
    provider = await redis_cache_provider
    mock_redis_client.delete.side_effect = RedisError("DEL command failed")
    result = await provider.delete("any_key")
    assert result is False
    assert "Redis DELETE error for 'any_key'" in caplog.text


@pytest.mark.asyncio()
async def test_redis_cache_exists_true(redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    provider = await redis_cache_provider
    key = "existing_key"
    mock_redis_client.exists.return_value = 1
    result = await provider.exists(key)
    mock_redis_client.exists.assert_awaited_once_with(key)
    assert result is True


@pytest.mark.asyncio()
async def test_redis_cache_exists_false(redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    provider = await redis_cache_provider
    key = "non_existing_key"
    mock_redis_client.exists.return_value = 0
    result = await provider.exists(key)
    assert result is False


@pytest.mark.asyncio()
async def test_redis_cache_exists_raises_redis_error(redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
    provider = await redis_cache_provider
    mock_redis_client.exists.side_effect = RedisError("EXISTS command failed")
    result = await provider.exists("any_key")
    assert result is False
    assert "Redis EXISTS error for 'any_key'" in caplog.text


@pytest.mark.asyncio()
async def test_redis_cache_clear_all(redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    provider = await redis_cache_provider
    result = await provider.clear_all()
    mock_redis_client.flushdb.assert_awaited_once()
    assert result is True


@pytest.mark.asyncio()
async def test_redis_cache_clear_all_raises_redis_error(redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
    provider = await redis_cache_provider
    mock_redis_client.flushdb.side_effect = RedisError("FLUSHDB command failed")
    result = await provider.clear_all()
    assert result is False
    assert "Redis FLUSHDB error" in caplog.text


@pytest.mark.asyncio()
async def test_setup_no_redis_url(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger=PROVIDER_LOGGER_NAME)
    provider = RedisCacheProvider()
    with patch("genie_tooling.cache_providers.impl.redis_cache.aioredis") as mock_aioredis:
        await provider.setup(config={})  # No redis_url
        assert provider._redis_client is None
        mock_aioredis.from_url.assert_not_called()
        assert "'redis_url' not configured. Plugin will be disabled" in caplog.text


@pytest.mark.asyncio()
async def test_setup_connection_error(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
    provider = RedisCacheProvider()
    with patch("genie_tooling.cache_providers.impl.redis_cache.aioredis") as mock_aioredis:
        mock_aioredis.from_url.return_value.ping.side_effect = RedisConnectionError("Connection refused")
        await provider.setup(config={"redis_url": "redis://bad-host"})
        assert provider._redis_client is None
        assert "Failed to connect to Redis: Connection refused" in caplog.text


@pytest.mark.asyncio()
async def test_teardown_closes_client(redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    provider = await redis_cache_provider
    await provider.teardown()
    mock_redis_client.close.assert_awaited_once()
    assert provider._redis_client is None


@pytest.mark.asyncio()
async def test_teardown_handles_close_error(redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
    provider = await redis_cache_provider
    mock_redis_client.close.side_effect = RedisError("Failed to close gracefully")
    await provider.teardown()
    assert provider._redis_client is None
    assert "Error closing Redis client: Failed to close gracefully" in caplog.text
