"""Unit tests for the default implementations in caching.abc.CacheProvider."""
import logging
from typing import Any, Dict, Optional

import pytest
from genie_tooling.caching.abc import CacheProvider
from genie_tooling.core.types import Plugin  # For concrete implementation


# A minimal concrete implementation of CacheProvider for testing defaults
class DefaultImplCacheProvider(CacheProvider, Plugin):
    plugin_id: str = "default_impl_cache_provider_v1"
    description: str = "A cache provider using only default implementations from CacheProvider protocol."

    # setup and teardown are part of Plugin, not CacheProvider itself directly with defaults
    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        await super().setup(config) # Call Plugin's default setup

    async def teardown(self) -> None:
        await super().teardown() # Call Plugin's default teardown

@pytest.fixture
async def default_cache_provider_fixture() -> DefaultImplCacheProvider: # Renamed for clarity
    provider = DefaultImplCacheProvider()
    await provider.setup() # Call setup for completeness
    return provider

@pytest.mark.asyncio
async def test_cache_provider_default_get(default_cache_provider_fixture: DefaultImplCacheProvider, caplog: pytest.LogCaptureFixture):
    """Test default CacheProvider.get() logs warning and returns None."""
    default_cache_provider = await default_cache_provider_fixture # Await the fixture
    caplog.set_level(logging.WARNING)
    key = "test_get_key"
    result = await default_cache_provider.get(key)

    assert result is None
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert f"CacheProvider '{default_cache_provider.plugin_id}' get method not fully implemented." in caplog.text

@pytest.mark.asyncio
async def test_cache_provider_default_set(default_cache_provider_fixture: DefaultImplCacheProvider, caplog: pytest.LogCaptureFixture):
    """Test default CacheProvider.set() logs warning and does nothing."""
    default_cache_provider = await default_cache_provider_fixture # Await the fixture
    caplog.set_level(logging.WARNING)
    key = "test_set_key"
    value = "test_value"
    await default_cache_provider.set(key, value) # Should not raise error

    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert f"CacheProvider '{default_cache_provider.plugin_id}' set method not fully implemented." in caplog.text
    # Verify it didn't actually set anything (by calling default get)
    assert await default_cache_provider.get(key) is None

@pytest.mark.asyncio
async def test_cache_provider_default_delete(default_cache_provider_fixture: DefaultImplCacheProvider, caplog: pytest.LogCaptureFixture):
    """Test default CacheProvider.delete() logs warning and returns False."""
    default_cache_provider = await default_cache_provider_fixture # Await the fixture
    caplog.set_level(logging.WARNING)
    key = "test_delete_key"
    result = await default_cache_provider.delete(key)

    assert result is False
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert f"CacheProvider '{default_cache_provider.plugin_id}' delete method not fully implemented." in caplog.text

@pytest.mark.asyncio
async def test_cache_provider_default_exists(default_cache_provider_fixture: DefaultImplCacheProvider, caplog: pytest.LogCaptureFixture):
    """Test default CacheProvider.exists() uses get() and logs debug + warning."""
    default_cache_provider = await default_cache_provider_fixture # Await the fixture
    caplog.set_level(logging.DEBUG) # To catch the debug log from default exists
    key = "test_exists_key"
    result = await default_cache_provider.exists(key)

    assert result is False
    # Expect one DEBUG log from exists() itself, and one WARNING log from the underlying get()
    assert any(record.levelname == "DEBUG" and f"CacheProvider '{default_cache_provider.plugin_id}' exists method using default get() check." in record.message for record in caplog.records)
    assert any(record.levelname == "WARNING" and f"CacheProvider '{default_cache_provider.plugin_id}' get method not fully implemented." in record.message for record in caplog.records)

@pytest.mark.asyncio
async def test_cache_provider_default_clear_all(default_cache_provider_fixture: DefaultImplCacheProvider, caplog: pytest.LogCaptureFixture):
    """Test default CacheProvider.clear_all() logs warning and returns False."""
    default_cache_provider = await default_cache_provider_fixture # Await the fixture
    caplog.set_level(logging.WARNING)
    result = await default_cache_provider.clear_all()

    assert result is False
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert f"CacheProvider '{default_cache_provider.plugin_id}' clear_all method not fully implemented." in caplog.text
