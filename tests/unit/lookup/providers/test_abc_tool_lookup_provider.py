"""Unit tests for the default implementations in lookup.providers.abc.ToolLookupProvider."""
import logging
from typing import Any, Dict, List, Optional

import pytest

from genie_tooling.core.types import Plugin  # For concrete implementation

# Updated import path for ToolLookupProvider
from genie_tooling.tool_lookup_providers.abc import ToolLookupProvider

# RankedToolResult is imported by ToolLookupProvider, but not used directly in this test file.
# No change needed for RankedToolResult here.


# Minimal concrete implementation of ToolLookupProvider
class DefaultImplToolLookupProvider(ToolLookupProvider, Plugin):
    plugin_id: str = "default_impl_lookup_provider_v1"
    description: str = "A lookup provider using only default implementations."

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        await super().setup(config)

    async def teardown(self) -> None:
        await super().teardown()

@pytest.fixture
async def default_lookup_provider_fixture() -> DefaultImplToolLookupProvider: # Renamed for clarity
    provider = DefaultImplToolLookupProvider()
    await provider.setup()
    return provider

@pytest.mark.asyncio
async def test_tool_lookup_provider_default_index_tools(default_lookup_provider_fixture: DefaultImplToolLookupProvider, caplog: pytest.LogCaptureFixture):
    """Test default ToolLookupProvider.index_tools() logs warning."""
    default_lookup_provider = await default_lookup_provider_fixture # Await the fixture
    caplog.set_level(logging.WARNING)
    tools_data: List[Dict[str, Any]] = [{"id": "tool1", "desc": "Test tool"}]
    await default_lookup_provider.index_tools(tools_data) # Should not raise error

    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert f"ToolLookupProvider '{default_lookup_provider.plugin_id}' index_tools method not fully implemented." in caplog.text

@pytest.mark.asyncio
async def test_tool_lookup_provider_default_find_tools(default_lookup_provider_fixture: DefaultImplToolLookupProvider, caplog: pytest.LogCaptureFixture):
    """Test default ToolLookupProvider.find_tools() logs warning and returns empty list."""
    default_lookup_provider = await default_lookup_provider_fixture # Await the fixture
    caplog.set_level(logging.WARNING)
    query = "find a tool"
    results = await default_lookup_provider.find_tools(query)

    assert isinstance(results, list)
    assert len(results) == 0
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
    assert f"ToolLookupProvider '{default_lookup_provider.plugin_id}' find_tools method not fully implemented." in caplog.text
