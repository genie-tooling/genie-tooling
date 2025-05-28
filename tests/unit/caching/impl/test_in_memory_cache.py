### tests/unit/caching/impl/test_in_memory_cache.py
"""Unit tests for InMemoryCacheProvider."""
import asyncio
from unittest.mock import AsyncMock as StdLibAsyncMock  # Use standard library AsyncMock
from unittest.mock import MagicMock

import pytest
from genie_tooling.caching.impl.in_memory import InMemoryCacheProvider


@pytest.fixture
async def mem_cache() -> InMemoryCacheProvider:
    cache = InMemoryCacheProvider()
    await cache.setup(config={"cleanup_interval_seconds": 10000})
    await cache.clear_all()
    return cache

@pytest.mark.asyncio
async def test_in_memory_cache_set_get_exists(mem_cache: InMemoryCacheProvider):
    actual_cache = await mem_cache
    key, value = "test_key", "test_value"
    assert await actual_cache.get(key) is None
    assert await actual_cache.exists(key) is False
    await actual_cache.set(key, value)
    assert await actual_cache.get(key) == value
    assert await actual_cache.exists(key) is True

@pytest.mark.asyncio
async def test_in_memory_cache_delete(mem_cache: InMemoryCacheProvider):
    actual_cache = await mem_cache
    key, value = "delete_key", {"data": 123}
    await actual_cache.set(key, value)
    assert await actual_cache.get(key) == value
    assert await actual_cache.delete(key) is True
    assert await actual_cache.get(key) is None
    assert await actual_cache.exists(key) is False
    assert await actual_cache.delete(key) is False

@pytest.mark.asyncio
async def test_in_memory_cache_ttl_expiration(mem_cache: InMemoryCacheProvider):
    actual_cache = await mem_cache
    key, value = "ttl_key", "expiring_value"
    ttl = 0.05
    await actual_cache.set(key, value, ttl_seconds=ttl)
    assert await actual_cache.get(key) == value
    await asyncio.sleep(ttl + 0.03)
    assert await actual_cache.get(key) is None
    assert await actual_cache.exists(key) is False

@pytest.mark.asyncio
async def test_in_memory_cache_no_ttl(mem_cache: InMemoryCacheProvider):
    actual_cache = await mem_cache
    key, value = "no_ttl_key", "persistent_value"
    await actual_cache.set(key, value, ttl_seconds=None)
    await asyncio.sleep(0.05)
    assert await actual_cache.get(key) == value

@pytest.mark.asyncio
async def test_in_memory_cache_default_ttl_from_setup(mocker):
    cache_with_default_ttl = InMemoryCacheProvider()
    await cache_with_default_ttl.setup(config={"default_ttl_seconds": 0.05, "cleanup_interval_seconds": 10000})
    key, value = "default_ttl_applies", "data"
    await cache_with_default_ttl.set(key, value)
    assert await cache_with_default_ttl.get(key) == value
    await asyncio.sleep(0.08)
    assert await cache_with_default_ttl.get(key) is None
    await cache_with_default_ttl.teardown()

@pytest.mark.asyncio
async def test_in_memory_cache_clear_all(mem_cache: InMemoryCacheProvider):
    actual_cache = await mem_cache
    await actual_cache.set("key1", "val1")
    await actual_cache.set("key2", "val2", ttl_seconds=10)
    assert await actual_cache.exists("key1") is True
    assert await actual_cache.clear_all() is True
    assert await actual_cache.exists("key1") is False
    assert await actual_cache.exists("key2") is False
    assert len(actual_cache._cache) == 0 # type: ignore

@pytest.mark.asyncio
async def test_in_memory_cache_max_size_eviction_basic(mocker):
    cache_with_size_limit = InMemoryCacheProvider()
    await cache_with_size_limit.setup(config={"max_size": 2, "cleanup_interval_seconds": 10000})
    await cache_with_size_limit.set("k1", "v1")
    await cache_with_size_limit.set("k2", "v2")
    assert len(cache_with_size_limit._cache) == 2 # type: ignore
    await cache_with_size_limit.set("k3", "v3")
    assert len(cache_with_size_limit._cache) == 2 # type: ignore
    assert await cache_with_size_limit.get("k1") is None
    assert await cache_with_size_limit.get("k2") == "v2"
    assert await cache_with_size_limit.get("k3") == "v3"
    await cache_with_size_limit.teardown()

@pytest.mark.asyncio
async def test_in_memory_cache_periodic_cleanup_task(mocker):
    cleanup_interval = 0.05
    cache_with_cleanup = InMemoryCacheProvider()

    # Use standard library AsyncMock explicitly
    mock_task_instance = StdLibAsyncMock(spec=asyncio.Task)

    # When `await mock_task_instance` is called, it should raise CancelledError.
    # The most direct way for an AsyncMock is to set its side_effect to the exception.
    mock_task_instance.side_effect = asyncio.CancelledError("Task was cancelled by mock side_effect")

    # Mock the cancel method just to record the call.
    mock_task_instance.cancel = MagicMock(return_value=True)
    mock_task_instance.done.return_value = False

    patched_create_task = mocker.patch("asyncio.create_task", return_value=mock_task_instance)

    await cache_with_cleanup.setup(config={"cleanup_interval_seconds": cleanup_interval})
    patched_create_task.assert_called_once()

    # This will call mock_task_instance.cancel() and then `await mock_task_instance`.
    # The `await mock_task_instance` should now raise CancelledError due to the side_effect.
    await cache_with_cleanup.teardown()

    mock_task_instance.cancel.assert_called_once()
    # Successful teardown implies the CancelledError was caught and handled as expected.
###<END-OF-FILE>###
