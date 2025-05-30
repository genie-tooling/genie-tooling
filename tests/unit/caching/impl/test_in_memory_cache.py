### tests/unit/caching/impl/test_in_memory_cache.py
"""Unit tests for InMemoryCacheProvider."""
import asyncio
from unittest.mock import AsyncMock as StdLibAsyncMock
from unittest.mock import MagicMock

import pytest
from genie_tooling.cache_providers.impl.in_memory import InMemoryCacheProvider


@pytest.fixture
async def mem_cache() -> InMemoryCacheProvider:
    cache = InMemoryCacheProvider()
    # Reduce cleanup interval for faster test execution if a test relies on it,
    # but for most tests, a long interval is fine to prevent interference.
    await cache.setup(config={"cleanup_interval_seconds": 10000})
    await cache.clear_all() # Ensure clean state for each test
    return cache

@pytest.mark.asyncio
async def test_in_memory_cache_set_get_exists(mem_cache: InMemoryCacheProvider):
    actual_cache = await mem_cache # Fixture is async
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
    assert await actual_cache.delete(key) is False # Deleting non-existent key

@pytest.mark.asyncio
async def test_in_memory_cache_ttl_expiration(mem_cache: InMemoryCacheProvider):
    actual_cache = await mem_cache
    key, value = "ttl_key", "expiring_value"
    ttl = 0.05 # Short TTL for testing
    await actual_cache.set(key, value, ttl_seconds=ttl)
    assert await actual_cache.get(key) == value # Should be present immediately
    await asyncio.sleep(ttl + 0.03) # Wait for slightly longer than TTL
    assert await actual_cache.get(key) is None # Should be expired and removed
    assert await actual_cache.exists(key) is False

@pytest.mark.asyncio
async def test_in_memory_cache_no_ttl(mem_cache: InMemoryCacheProvider):
    actual_cache = await mem_cache
    key, value = "no_ttl_key", "persistent_value"
    await actual_cache.set(key, value, ttl_seconds=None) # Explicitly no TTL
    await asyncio.sleep(0.05) # Wait a bit
    assert await actual_cache.get(key) == value # Should still be there

@pytest.mark.asyncio
async def test_in_memory_cache_default_ttl_from_setup(mocker): # No mem_cache fixture here
    cache_with_default_ttl = InMemoryCacheProvider()
    default_ttl_val = 0.05
    # Setup with a default TTL and a long cleanup interval to not interfere
    await cache_with_default_ttl.setup(config={"default_ttl_seconds": default_ttl_val, "cleanup_interval_seconds": 10000})
    key, value = "default_ttl_applies", "data"
    await cache_with_default_ttl.set(key, value) # Uses default TTL
    assert await cache_with_default_ttl.get(key) == value
    await asyncio.sleep(default_ttl_val + 0.03)
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
    assert len(actual_cache._cache) == 0

@pytest.mark.asyncio
async def test_in_memory_cache_max_size_eviction_basic(mocker): # No mem_cache fixture
    cache_with_size_limit = InMemoryCacheProvider()
    await cache_with_size_limit.setup(config={"max_size": 2, "cleanup_interval_seconds": 10000})
    await cache_with_size_limit.set("k1", "v1") # k1 is oldest
    await cache_with_size_limit.set("k2", "v2") # k2 is newer
    assert len(cache_with_size_limit._cache) == 2
    await cache_with_size_limit.set("k3", "v3") # k3 added, k1 should be evicted
    assert len(cache_with_size_limit._cache) == 2
    assert await cache_with_size_limit.get("k1") is None # k1 evicted
    assert await cache_with_size_limit.get("k2") == "v2" # k2 remains
    assert await cache_with_size_limit.get("k3") == "v3" # k3 present
    await cache_with_size_limit.teardown()

@pytest.mark.asyncio
async def test_in_memory_cache_periodic_cleanup_task(mocker):
    cleanup_interval = 0.05
    cache_with_cleanup = InMemoryCacheProvider()
    mock_task_instance = StdLibAsyncMock(spec=asyncio.Task)
    mock_task_instance.side_effect = asyncio.CancelledError("Task was cancelled by mock side_effect")
    mock_task_instance.cancel = MagicMock(return_value=True)
    mock_task_instance.done.return_value = False

    # Patch where 'create_task' is looked up by the InMemoryCacheProvider module
    patched_create_task = mocker.patch(
        "genie_tooling.cache_providers.impl.in_memory.asyncio.create_task",
        return_value=mock_task_instance
    )

    await cache_with_cleanup.setup(config={"cleanup_interval_seconds": cleanup_interval})
    patched_create_task.assert_called_once() # This should now pass

    await cache_with_cleanup.teardown()
    mock_task_instance.cancel.assert_called_once()
