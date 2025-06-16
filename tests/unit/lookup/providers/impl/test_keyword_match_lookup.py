### tests/unit/lookup/providers/impl/test_keyword_match_lookup.py
"""Unit tests for KeywordMatchLookupProvider."""
import logging

import pytest
from genie_tooling.tool_lookup_providers.impl.keyword_match import (
    KeywordMatchLookupProvider,
)

MODULE_LOGGER_NAME = "genie_tooling.tool_lookup_providers.impl.keyword_match"
module_logger_instance = logging.getLogger(MODULE_LOGGER_NAME)


@pytest.fixture()
async def keyword_lookup_provider() -> KeywordMatchLookupProvider:
    provider = KeywordMatchLookupProvider()
    return provider

# --- Test Cases ---

@pytest.mark.asyncio()
async def test_km_index_tools_stores_data(keyword_lookup_provider: KeywordMatchLookupProvider):
    provider = await keyword_lookup_provider
    tools_data = [
        {"identifier": "tool_alpha", "name": "Alpha Tool", "lookup_text_representation": "Performs alpha operations."},
        {"identifier": "tool_beta", "name": "Beta Utility", "lookup_text_representation": "Handles beta tasks and processes."}
    ]
    await provider.index_tools(tools_data)
    # CORRECTED: Assert against the dictionary structure
    assert len(provider._indexed_tools_data) == 2
    assert provider._indexed_tools_data["tool_alpha"] == tools_data[0]
    assert provider._indexed_tools_data["tool_beta"] == tools_data[1]

@pytest.mark.asyncio()
async def test_km_find_tools_successful_match(keyword_lookup_provider: KeywordMatchLookupProvider):
    provider = await keyword_lookup_provider
    tools_data = [
        {"identifier": "search_tool",
         "_raw_metadata_snapshot": {"name": "Web Search", "description_llm": "Finds information on the internet.", "tags": ["web", "information"]},
         "lookup_text_representation": "Finds information on the internet. Search web content."
        },
        {"identifier": "calculator",
         "_raw_metadata_snapshot": {"name": "Math Calculator", "description_llm": "Solves arithmetic expressions."}
        }
    ]
    await provider.index_tools(tools_data)

    results = await provider.find_tools("search for internet data", top_k=1)
    assert len(results) == 1
    assert results[0].tool_identifier == "search_tool"
    assert results[0].score > 0
    assert results[0].description_snippet is not None
    assert "Matched keywords: internet, search" in results[0].description_snippet

@pytest.mark.asyncio()
async def test_km_find_tools_empty_query(keyword_lookup_provider: KeywordMatchLookupProvider, caplog: pytest.LogCaptureFixture):
    provider = await keyword_lookup_provider
    query_text = "   "
    expected_log_message = f"{provider.plugin_id}: Empty query."

    with caplog.at_level(logging.DEBUG, logger=MODULE_LOGGER_NAME):
        # CORRECTED: Use "identifier" key in test data
        await provider.index_tools([{"identifier": "t1", "name":"tool"}])
        results = await provider.find_tools(query_text)
        assert len(results) == 0

        assert any(
            rec.name == MODULE_LOGGER_NAME and
            rec.levelname == "DEBUG" and
            expected_log_message in rec.message
            for rec in caplog.records
        ), f"Log '{expected_log_message}' not found or not DEBUG. Caplog: {caplog.text}"

@pytest.mark.asyncio()
async def test_km_teardown_clears_index(keyword_lookup_provider: KeywordMatchLookupProvider):
    provider = await keyword_lookup_provider
    # CORRECTED: Use "identifier" key in test data
    await provider.index_tools([{"identifier":"t1", "name":"test"}])
    assert len(provider._indexed_tools_data) == 1

    await provider.teardown()
    assert len(provider._indexed_tools_data) == 0
