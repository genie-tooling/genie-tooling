### tests/unit/caching/impl/test_in_memory_cache.py
"""Unit tests for InMemoryCacheProvider."""
import asyncio
from unittest.mock import AsyncMock as StdLibAsyncMock
from unittest.mock import MagicMock

import pytest
from genie_tooling.cache_providers.impl.in_memory import InMemoryCacheProvider


@pytest.fixture()
def mem_cache(
    request: pytest.FixtureRequest, event_loop: asyncio.AbstractEventLoop
) -> InMemoryCacheProvider:
    """
    Provides a managed InMemoryCacheProvider instance for tests.

    This fixture is a regular 'def' fixture to avoid issues with how some
    pytest environments handle 'async def' generator fixtures. It manually
    handles the async setup and teardown using the event loop and a finalizer,
    which is a more robust pattern.
    """
    cache = InMemoryCacheProvider()

    # Run the async setup using the event loop provided by pytest-asyncio
    event_loop.run_until_complete(
        cache.setup(config={"cleanup_interval_seconds": 10000})
    )

    def finalizer() -> None:
        """The teardown logic to be run after the test completes."""
        # This is a sync function, so we use the event loop to run the async teardown.
        event_loop.run_until_complete(cache.teardown())

    # Register the finalizer to be called when the fixture goes out of scope
    request.addfinalizer(finalizer)

    return cache


@pytest.mark.asyncio()
async def test_in_memory_cache_set_get_exists(mem_cache: InMemoryCacheProvider):
    # The fixture now directly returns the setup instance.
    actual_cache = mem_cache
    key, value = "test_key", "test_value"
    assert await actual_cache.get(key) is None
    assert await actual_cache.exists(key) is False
    await actual_cache.set(key, value)
    assert await actual_cache.get(key) == value
    assert await actual_cache.exists(key) is True


@pytest.mark.asyncio()
async def test_in_memory_cache_delete(mem_cache: InMemoryCacheProvider):
    actual_cache = mem_cache
    key, value = "delete_key", {"data": 123}
    await actual_cache.set(key, value)
    assert await actual_cache.get(key) == value
    assert await actual_cache.delete(key) is True
    assert await actual_cache.get(key) is None
    assert await actual_cache.exists(key) is False
    assert await actual_cache.delete(key) is False  # Deleting non-existent key


@pytest.mark.asyncio()
async def test_in_memory_cache_ttl_expiration(mem_cache: InMemoryCacheProvider):
    actual_cache = mem_cache
    key, value = "ttl_key", "expiring_value"
    ttl = 0.05  # Short TTL for testing
    await actual_cache.set(key, value, ttl_seconds=ttl)
    assert await actual_cache.get(key) == value  # Should be present immediately
    await asyncio.sleep(ttl + 0.03)  # Wait for slightly longer than TTL
    assert await actual_cache.get(key) is None  # Should be expired and removed
    assert await actual_cache.exists(key) is False


@pytest.mark.asyncio()
async def test_in_memory_cache_no_ttl(mem_cache: InMemoryCacheProvider):
    actual_cache = mem_cache
    key, value = "no_ttl_key", "persistent_value"
    await actual_cache.set(key, value, ttl_seconds=None)  # Explicitly no TTL
    await asyncio.sleep(0.05)  # Wait a bit
    assert await actual_cache.get(key) == value  # Should still be there


@pytest.mark.asyncio()
async def test_in_memory_cache_default_ttl_from_setup():
    cache_with_default_ttl = InMemoryCacheProvider()
    try:
        default_ttl_val = 0.05
        # Setup with a default TTL and a long cleanup interval to not interfere
        await cache_with_default_ttl.setup(
            config={
                "default_ttl_seconds": default_ttl_val,
                "cleanup_interval_seconds": 10000,
            }
        )
        key, value = "default_ttl_applies", "data"
        await cache_with_default_ttl.set(key, value)  # Uses default TTL
        assert await cache_with_default_ttl.get(key) == value
        await asyncio.sleep(default_ttl_val + 0.03)
        assert await cache_with_default_ttl.get(key) is None
    finally:
        await cache_with_default_ttl.teardown()


@pytest.mark.asyncio()
async def test_in_memory_cache_clear_all(mem_cache: InMemoryCacheProvider):
    actual_cache = mem_cache
    await actual_cache.set("key1", "val1")
    await actual_cache.set("key2", "val2", ttl_seconds=10)
    assert await actual_cache.exists("key1") is True
    assert await actual_cache.clear_all() is True
    assert await actual_cache.exists("key1") is False
    assert await actual_cache.exists("key2") is False
    assert len(actual_cache._cache) == 0


@pytest.mark.asyncio()
async def test_in_memory_cache_max_size_eviction_basic():
    cache_with_size_limit = InMemoryCacheProvider()
    try:
        await cache_with_size_limit.setup(
            config={"max_size": 2, "cleanup_interval_seconds": 10000}
        )
        await cache_with_size_limit.set("k1", "v1")  # k1 is oldest
        await cache_with_size_limit.set("k2", "v2")  # k2 is newer
        assert len(cache_with_size_limit._cache) == 2
        await cache_with_size_limit.set("k3", "v3")  # k3 added, k1 should be evicted
        assert len(cache_with_size_limit._cache) == 2
        assert await cache_with_size_limit.get("k1") is None  # k1 evicted
        assert await cache_with_size_limit.get("k2") == "v2"  # k2 remains
        assert await cache_with_size_limit.get("k3") == "v3"  # k3 present
    finally:
        await cache_with_size_limit.teardown()


@pytest.mark.asyncio()
async def test_in_memory_cache_periodic_cleanup_task(mocker):
    cleanup_interval = 0.05
    cache_with_cleanup = InMemoryCacheProvider()
    try:
        patched_create_task = mocker.patch(
            "genie_tooling.cache_providers.impl.in_memory.asyncio.create_task",
            wraps=asyncio.create_task,
        )

        await cache_with_cleanup.setup(
            config={"cleanup_interval_seconds": cleanup_interval}
        )
        patched_create_task.assert_called_once()
        assert cache_with_cleanup._cleanup_task is not None
        assert not cache_with_cleanup._cleanup_task.done()

    finally:
        await cache_with_cleanup.teardown()

    assert (
        cache_with_cleanup._cleanup_task is None
    ), "Teardown should nullify the _cleanup_task attribute"