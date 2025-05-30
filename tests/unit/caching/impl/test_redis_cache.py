### tests/unit/caching/impl/test_redis_cache.py
"""Unit tests for RedisCacheProvider."""
import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from genie_tooling.cache_providers.impl.redis_cache import (
    RedisCacheProvider,
    RedisConnectionError,  # Assuming this is correctly imported or mocked if redis lib not present
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
def sync_redis_cache_provider(request) -> RedisCacheProvider:
    provider = RedisCacheProvider()
    async def async_finalizer():
        if getattr(provider, "_redis_client", None):
            await provider.teardown()
    def finalizer_wrapper():
        try: asyncio.get_running_loop().run_until_complete(async_finalizer())
        except RuntimeError: asyncio.run(async_finalizer())
    request.addfinalizer(finalizer_wrapper)
    return provider

@pytest.mark.asyncio
async def test_redis_cache_setup_success(sync_redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    with patch("genie_tooling.cache_providers.impl.redis_cache.aioredis.from_url", return_value=mock_redis_client) as mock_from_url:
        await sync_redis_cache_provider.setup(config={"redis_url": "redis://testhost:1234/1"})
        mock_from_url.assert_called_once_with("redis://testhost:1234/1", decode_responses=True)
        mock_redis_client.ping.assert_awaited_once()
        assert sync_redis_cache_provider._redis_client is mock_redis_client

@pytest.mark.asyncio
async def test_redis_cache_setup_connection_error(sync_redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    mock_redis_client.ping = AsyncMock(side_effect=RedisConnectionError("Ping failed"))
    with patch("genie_tooling.cache_providers.impl.redis_cache.aioredis.from_url", return_value=mock_redis_client):
        await sync_redis_cache_provider.setup(config={"redis_url": "redis://failhost:6379/0"})
        assert sync_redis_cache_provider._redis_client is None

@pytest.mark.asyncio
async def test_redis_cache_get_hit_json(sync_redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    with patch("genie_tooling.cache_providers.impl.redis_cache.aioredis.from_url", return_value=mock_redis_client):
        await sync_redis_cache_provider.setup(config={"json_serialization": True})
    key = "test_key_json"; expected_value = {"data": "value", "count": 1}
    mock_redis_client.get.return_value = json.dumps(expected_value)
    retrieved_value = await sync_redis_cache_provider.get(key)
    mock_redis_client.get.assert_awaited_once_with(key)
    assert retrieved_value == expected_value

@pytest.mark.asyncio
async def test_redis_cache_get_hit_string(sync_redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    with patch("genie_tooling.cache_providers.impl.redis_cache.aioredis.from_url", return_value=mock_redis_client):
        await sync_redis_cache_provider.setup(config={"json_serialization": False})
    key = "test_key_string"; expected_value = "simple string value"
    mock_redis_client.get.return_value = expected_value
    retrieved_value = await sync_redis_cache_provider.get(key)
    mock_redis_client.get.assert_awaited_once_with(key)
    assert retrieved_value == expected_value

@pytest.mark.asyncio
async def test_redis_cache_get_miss(sync_redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    with patch("genie_tooling.cache_providers.impl.redis_cache.aioredis.from_url", return_value=mock_redis_client):
        await sync_redis_cache_provider.setup()
    mock_redis_client.get.return_value = None; key = "miss_key"
    retrieved_value = await sync_redis_cache_provider.get(key)
    mock_redis_client.get.assert_awaited_once_with(key)
    assert retrieved_value is None

@pytest.mark.asyncio
async def test_redis_cache_get_json_decode_error(sync_redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    with patch("genie_tooling.cache_providers.impl.redis_cache.aioredis.from_url", return_value=mock_redis_client):
        await sync_redis_cache_provider.setup(config={"json_serialization": True})
    key = "decode_error_key"; malformed_json_string = '{"data": "value", "count": 1'
    mock_redis_client.get.return_value = malformed_json_string
    retrieved_value = await sync_redis_cache_provider.get(key)
    assert retrieved_value == malformed_json_string

@pytest.mark.asyncio
async def test_redis_cache_set_json_value(sync_redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    with patch("genie_tooling.cache_providers.impl.redis_cache.aioredis.from_url", return_value=mock_redis_client):
        await sync_redis_cache_provider.setup(config={"json_serialization": True})
    key = "set_json_key"; value = {"complex": True, "items": [1, "a"]}
    expected_stored_value = json.dumps(value)
    mock_redis_client.set.reset_mock()
    await sync_redis_cache_provider.set(key, value)
    mock_redis_client.set.assert_awaited_once_with(key, expected_stored_value, ex=None)

@pytest.mark.asyncio
async def test_redis_cache_set_string_value_no_json_serialize(sync_redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    with patch("genie_tooling.cache_providers.impl.redis_cache.aioredis.from_url", return_value=mock_redis_client):
        await sync_redis_cache_provider.setup(config={"json_serialization": False})
    key = "set_string_key_no_json"; value = "plain string"
    mock_redis_client.set.reset_mock()
    await sync_redis_cache_provider.set(key, value)
    mock_redis_client.set.assert_awaited_once_with(key, value, ex=None)

    complex_value = {"data": "test"}
    mock_redis_client.set.reset_mock()
    await sync_redis_cache_provider.set("complex_key_no_json", complex_value)
    mock_redis_client.set.assert_awaited_with("complex_key_no_json", str(complex_value), ex=None)

@pytest.mark.asyncio
async def test_redis_cache_set_string_value_with_json_serialize(sync_redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    # Test case where json_serialization is True, but value is already a string
    with patch("genie_tooling.cache_providers.impl.redis_cache.aioredis.from_url", return_value=mock_redis_client):
        await sync_redis_cache_provider.setup(config={"json_serialization": True})
    key = "set_string_key_json_true"; value = "plain string to store"
    mock_redis_client.set.reset_mock()
    await sync_redis_cache_provider.set(key, value)
    # If value is already string, json.dumps() is NOT called, it's passed as is.
    mock_redis_client.set.assert_awaited_once_with(key, value, ex=None)


@pytest.mark.asyncio
async def test_redis_cache_set_with_ttl(sync_redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    with patch("genie_tooling.cache_providers.impl.redis_cache.aioredis.from_url", return_value=mock_redis_client):
        # Ensure json_serialization is False for this test to avoid double quoting if value is string
        await sync_redis_cache_provider.setup(config={"json_serialization": False})
    key = "set_ttl_key"; value = "data_with_ttl"; ttl = 300
    mock_redis_client.set.reset_mock()
    await sync_redis_cache_provider.set(key, value, ttl_seconds=ttl)
    mock_redis_client.set.assert_awaited_once_with(key, value, ex=ttl)

@pytest.mark.asyncio
async def test_redis_cache_delete_existing(sync_redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    with patch("genie_tooling.cache_providers.impl.redis_cache.aioredis.from_url", return_value=mock_redis_client):
        await sync_redis_cache_provider.setup()
    key = "delete_me"; mock_redis_client.delete.return_value = 1
    result = await sync_redis_cache_provider.delete(key)
    mock_redis_client.delete.assert_awaited_once_with(key)
    assert result is True

@pytest.mark.asyncio
async def test_redis_cache_delete_non_existing(sync_redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    with patch("genie_tooling.cache_providers.impl.redis_cache.aioredis.from_url", return_value=mock_redis_client):
        await sync_redis_cache_provider.setup()
    key = "i_dont_exist"; mock_redis_client.delete.return_value = 0
    result = await sync_redis_cache_provider.delete(key)
    assert result is False

@pytest.mark.asyncio
async def test_redis_cache_exists_true(sync_redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    with patch("genie_tooling.cache_providers.impl.redis_cache.aioredis.from_url", return_value=mock_redis_client):
        await sync_redis_cache_provider.setup()
    key = "existing_key"; mock_redis_client.exists.return_value = 1
    assert await sync_redis_cache_provider.exists(key) is True
    mock_redis_client.exists.assert_awaited_once_with(key)

@pytest.mark.asyncio
async def test_redis_cache_exists_false(sync_redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    with patch("genie_tooling.cache_providers.impl.redis_cache.aioredis.from_url", return_value=mock_redis_client):
        await sync_redis_cache_provider.setup()
    key = "non_existing_key_for_exists"; mock_redis_client.exists.return_value = 0
    assert await sync_redis_cache_provider.exists(key) is False

@pytest.mark.asyncio
async def test_redis_cache_clear_all(sync_redis_cache_provider: RedisCacheProvider, mock_redis_client: AsyncMock):
    with patch("genie_tooling.cache_providers.impl.redis_cache.aioredis.from_url", return_value=mock_redis_client):
        await sync_redis_cache_provider.setup()
    assert await sync_redis_cache_provider.clear_all() is True
    mock_redis_client.flushdb.assert_awaited_once()

@pytest.mark.asyncio
async def test_redis_cache_teardown_if_client_exists(mock_redis_client: AsyncMock):
    provider = RedisCacheProvider()
    with patch("genie_tooling.cache_providers.impl.redis_cache.aioredis.from_url", return_value=mock_redis_client):
        await provider.setup(config={"redis_url": "redis://mock"})
    assert provider._redis_client is not None
    await provider.teardown()
    mock_redis_client.close.assert_awaited_once()
    assert provider._redis_client is None

@pytest.mark.asyncio
async def test_redis_cache_teardown_if_client_is_none(mock_redis_client: AsyncMock):
    provider = RedisCacheProvider()
    with patch("genie_tooling.cache_providers.impl.redis_cache.aioredis.from_url", side_effect=RedisConnectionError("Setup fail")):
        await provider.setup(config={"redis_url": "redis://mock"})
    assert provider._redis_client is None
    await provider.teardown()
    mock_redis_client.close.assert_not_called()

@pytest.mark.asyncio
async def test_redis_cache_operations_fail_if_client_not_available(mock_redis_client: AsyncMock):
    provider = RedisCacheProvider()
    with patch("genie_tooling.cache_providers.impl.redis_cache.aioredis.from_url") as mock_from_url:
        mock_from_url.side_effect = RedisConnectionError("Initial connection failed")
        await provider.setup(config={"redis_url": "redis://bad_url"})
    assert provider._redis_client is None
    assert await provider.get("key") is None
    await provider.set("key", "value") # Should do nothing and not error
    assert await provider.delete("key") is False
    assert await provider.exists("key") is False
    assert await provider.clear_all() is False
    mock_redis_client.set.assert_not_called() # Verify no actual redis calls attempted
