"""Unit tests for the default implementations in caching.abc.CacheProvider."""
import logging
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock

import pytest
from genie_tooling.cache_providers.abc import CacheProvider
from genie_tooling.core.types import Plugin


class DefaultImplCacheProvider(CacheProvider, Plugin):
    plugin_id: str = "default_impl_cache_provider_v1"
    description: str = "A cache provider using only default implementations from CacheProvider protocol."
    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        await super().setup(config)
    async def teardown(self) -> None:
        await super().teardown()

@pytest.fixture
async def default_cache_provider_fixture() -> DefaultImplCacheProvider:
    provider = DefaultImplCacheProvider()
    await provider.setup()
    return provider

@pytest.mark.asyncio
async def test_cache_provider_default_exists(default_cache_provider_fixture: DefaultImplCacheProvider, caplog: pytest.LogCaptureFixture):
    """Test default CacheProvider.exists() uses get() and logs debug + warning."""
    default_cache_provider = await default_cache_provider_fixture
    # Mock the get method on the instance to control its behavior
    default_cache_provider.get = AsyncMock(return_value=None)
    caplog.set_level(logging.DEBUG)
    key = "test_exists_key"
    result = await default_cache_provider.exists(key)

    assert result is False
    default_cache_provider.get.assert_awaited_once_with(key)
    assert any(record.levelname == "DEBUG" and f"CacheProvider '{default_cache_provider.plugin_id}' exists method using default get() check." in record.message for record in caplog.records)
