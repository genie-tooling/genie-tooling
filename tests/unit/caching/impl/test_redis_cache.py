"""Unit tests for RedisCacheProvider."""
import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from genie_tooling.cache_providers.impl.redis_cache import (
    RedisCacheProvider,
    RedisConnectionError,
)

@pytest.fixture
def mock_redis_client() -> AsyncMock:
    client = AsyncMock(spec=["ping", "get", "set", "delete", "exists", "flushdb", "close"])
    client.ping = AsyncMock(return_value=True)
    client.get = AsyncMock(return_value=None)
    client.set = AsyncMock(return_value=True)
    client.delete = AsyncMock(return_value=0)
    client.exists = AsyncMock(return_value=0)
    client.flushdb = AsyncMock(return_value=True)
    client.close = AsyncMock(return_value=None)
    return client

@pytest.fixture
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

@pytest.mark.asyncio
async def test_redis_cache_get_hit_json(redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    provider = await redis_cache_provider
    key = "test_key_json"
    expected_value = {"data": "value", "count": 1}
    mock_redis_client.get.return_value = json.dumps(expected_value)
    retrieved_value = await provider.get(key)
    mock_redis_client.get.assert_awaited_once_with(key)
    assert retrieved_value == expected_value

@pytest.mark.asyncio
async def test_redis_cache_get_hit_string(redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    provider = await redis_cache_provider
    provider._json_serialization = False
    key = "test_key_string"
    expected_value = "simple string value"
    mock_redis_client.get.return_value = expected_value
    retrieved_value = await provider.get(key)
    mock_redis_client.get.assert_awaited_once_with(key)
    assert retrieved_value == expected_value

@pytest.mark.asyncio
async def test_redis_cache_get_json_decode_error(redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    provider = await redis_cache_provider
    key = "decode_error_key"
    malformed_json_string = '{"data": "value", "count": 1'
    mock_redis_client.get.return_value = malformed_json_string
    retrieved_value = await provider.get(key)
    assert retrieved_value == malformed_json_string

@pytest.mark.asyncio
async def test_redis_cache_set_json_value(redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    provider = await redis_cache_provider
    key = "set_json_key"
    value = {"complex": True, "items": [1, "a"]}
    expected_stored_value = json.dumps(value)
    await provider.set(key, value)
    mock_redis_client.set.assert_awaited_once_with(key, expected_stored_value, ex=None)

@pytest.mark.asyncio
async def test_redis_cache_set_string_value_no_json_serialize(redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    provider = await redis_cache_provider
    provider._json_serialization = False
    key = "set_string_key_no_json"
    value = "plain string"
    await provider.set(key, value)
    mock_redis_client.set.assert_awaited_once_with(key, value, ex=None)

@pytest.mark.asyncio
async def test_redis_cache_set_string_value_with_json_serialize(redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    provider = await redis_cache_provider
    key = "set_string_key_json_true"
    value = "plain string to store"
    await provider.set(key, value)
    mock_redis_client.set.assert_awaited_once_with(key, value, ex=None)

@pytest.mark.asyncio
async def test_redis_cache_set_with_ttl(redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    provider = await redis_cache_provider
    key = "set_ttl_key"
    value = "data_with_ttl"
    ttl = 300
    await provider.set(key, value, ttl_seconds=ttl)
    mock_redis_client.set.assert_awaited_once_with(key, value, ex=ttl)

@pytest.mark.asyncio
async def test_redis_cache_delete_existing(redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    provider = await redis_cache_provider
    key = "delete_me"
    mock_redis_client.delete.return_value = 1
    result = await provider.delete(key)
    mock_redis_client.delete.assert_awaited_once_with(key)
    assert result is True

@pytest.mark.asyncio
async def test_redis_cache_exists_true(redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    provider = await redis_cache_provider
    key = "existing_key"
    mock_redis_client.exists.return_value = 1
    result = await provider.exists(key)
    mock_redis_client.exists.assert_awaited_once_with(key)
    assert result is True

@pytest.mark.asyncio
async def test_redis_cache_clear_all(redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    provider = await redis_cache_provider
    result = await provider.clear_all()
    mock_redis_client.flushdb.assert_awaited_once()
    assert result is True
