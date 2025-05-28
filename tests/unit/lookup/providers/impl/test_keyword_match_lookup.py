### tests/unit/lookup/providers/impl/test_keyword_match_lookup.py
"""Unit tests for KeywordMatchLookupProvider."""
import logging

import pytest
from genie_tooling.lookup.providers.impl.keyword_match import KeywordMatchLookupProvider

MODULE_LOGGER_NAME = "genie_tooling.lookup.providers.impl.keyword_match"
module_logger_instance = logging.getLogger(MODULE_LOGGER_NAME)


@pytest.fixture
async def keyword_lookup_provider() -> KeywordMatchLookupProvider:
    provider = KeywordMatchLookupProvider()
    return provider

# --- Test Cases ---

@pytest.mark.asyncio
async def test_km_index_tools_stores_data(keyword_lookup_provider: KeywordMatchLookupProvider):
    provider = await keyword_lookup_provider
    tools_data = [
        {"identifier": "tool_alpha", "name": "Alpha Tool", "lookup_text_representation": "Performs alpha operations."},
        {"identifier": "tool_beta", "name": "Beta Utility", "lookup_text_representation": "Handles beta tasks and processes."}
    ]
    await provider.index_tools(tools_data)
    assert provider._indexed_tools_data == tools_data

@pytest.mark.asyncio
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

@pytest.mark.asyncio
async def test_km_find_tools_no_match(keyword_lookup_provider: KeywordMatchLookupProvider):
    provider = await keyword_lookup_provider
    tools_data = [{"identifier": "tool1", "name": "Specific Tool", "lookup_text_representation": "Does one specific thing."}]
    await provider.index_tools(tools_data)

    results = await provider.find_tools("completely unrelated query about unicorns")
    assert len(results) == 0

@pytest.mark.asyncio
async def test_km_find_tools_empty_index(keyword_lookup_provider: KeywordMatchLookupProvider, caplog: pytest.LogCaptureFixture):
    provider = await keyword_lookup_provider

    original_level = module_logger_instance.level
    original_handlers = list(module_logger_instance.handlers)
    original_propagate = module_logger_instance.propagate
    module_logger_instance.setLevel(logging.DEBUG)
    module_logger_instance.propagate = True
    for handler in original_handlers: module_logger_instance.removeHandler(handler)

    try:
        with caplog.at_level(logging.DEBUG, logger=MODULE_LOGGER_NAME):
            await provider.index_tools([])
            results = await provider.find_tools("any query")
            assert len(results) == 0
            assert any(
                rec.name == MODULE_LOGGER_NAME and
                rec.levelname == "DEBUG" and
                f"{provider.plugin_id}: No tools indexed." in rec.message
                for rec in caplog.records
            ), f"Log 'No tools indexed.' not found or not DEBUG. Caplog: {caplog.text}"
    finally:
        module_logger_instance.setLevel(original_level)
        module_logger_instance.propagate = original_propagate
        for handler_in_finally in list(module_logger_instance.handlers): # Iterate over a copy
            module_logger_instance.removeHandler(handler_in_finally)
        for handler in original_handlers:
             module_logger_instance.addHandler(handler)


@pytest.mark.asyncio
async def test_km_find_tools_empty_query(keyword_lookup_provider: KeywordMatchLookupProvider, caplog: pytest.LogCaptureFixture):
    provider = await keyword_lookup_provider

    original_level = module_logger_instance.level
    original_handlers = list(module_logger_instance.handlers)
    original_propagate = module_logger_instance.propagate
    module_logger_instance.setLevel(logging.DEBUG)
    module_logger_instance.propagate = True
    for handler in original_handlers: module_logger_instance.removeHandler(handler)

    try:
        with caplog.at_level(logging.DEBUG, logger=MODULE_LOGGER_NAME):
            await provider.index_tools([{"id": "t1", "name":"tool"}])
            results = await provider.find_tools("   ")
            assert len(results) == 0
            assert any(
                rec.name == MODULE_LOGGER_NAME and
                rec.levelname == "DEBUG" and
                f"{provider.plugin_id}: Empty query." in rec.message
                for rec in caplog.records
            ), f"Log 'Empty query.' not found or not DEBUG. Caplog: {caplog.text}"
    finally:
        module_logger_instance.setLevel(original_level)
        module_logger_instance.propagate = original_propagate
        for handler_in_finally in list(module_logger_instance.handlers):
            module_logger_instance.removeHandler(handler_in_finally)
        for handler in original_handlers:
            module_logger_instance.addHandler(handler)


@pytest.mark.asyncio
async def test_km_find_tools_query_no_valid_keywords(keyword_lookup_provider: KeywordMatchLookupProvider, caplog: pytest.LogCaptureFixture):
    provider = await keyword_lookup_provider
    query_text = "is of an" # Changed query to ensure no keywords > 2 chars
    expected_log_message = f"{provider.plugin_id}: No valid keywords extracted from query '{query_text}'."

    original_level = module_logger_instance.level
    original_handlers = list(module_logger_instance.handlers)
    original_propagate = module_logger_instance.propagate
    module_logger_instance.setLevel(logging.DEBUG)
    module_logger_instance.propagate = True
    for handler in original_handlers: module_logger_instance.removeHandler(handler)

    try:
        with caplog.at_level(logging.DEBUG, logger=MODULE_LOGGER_NAME):
            await provider.index_tools([{"identifier": "t1", "name":"tool"}]) # Added identifier
            results = await provider.find_tools(query_text)
            assert len(results) == 0

            assert any(
                rec.name == MODULE_LOGGER_NAME and
                rec.levelname == "DEBUG" and
                expected_log_message in rec.message
                for rec in caplog.records
            ), f"Log '{expected_log_message}' not found or not DEBUG. Caplog: {caplog.text}"
    finally:
        module_logger_instance.setLevel(original_level)
        module_logger_instance.propagate = original_propagate
        for handler_in_finally in list(module_logger_instance.handlers):
            module_logger_instance.removeHandler(handler_in_finally)
        for handler in original_handlers:
            module_logger_instance.addHandler(handler)


@pytest.mark.asyncio
async def test_km_find_tools_top_k_behavior(keyword_lookup_provider: KeywordMatchLookupProvider):
    provider = await keyword_lookup_provider
    tools_data = [
        {"identifier": "t1", "name": "Query Match One", "lookup_text_representation": "excellent query text"},
        {"identifier": "t2", "name": "Query Match Two", "lookup_text_representation": "good query words"},
        {"identifier": "t3", "name": "Query Less", "lookup_text_representation": "some query"},
        {"identifier": "t4", "name": "No Match", "lookup_text_representation": "nothing relevant"}
    ]
    await provider.index_tools(tools_data)

    results_top2 = await provider.find_tools("excellent good query", top_k=2)
    assert len(results_top2) == 2
    assert results_top2[0].score >= results_top2[1].score

    results_top5 = await provider.find_tools("excellent good query", top_k=5)
    assert len(results_top5) == 3

@pytest.mark.asyncio
async def test_km_find_tools_fallback_to_metadata_fields(keyword_lookup_provider: KeywordMatchLookupProvider):
    provider = await keyword_lookup_provider
    tools_data = [
        {
            "identifier": "tool_a",
            "lookup_text_representation": "Primary lookup target: apple banana",
            "_raw_metadata_snapshot": {"name": "Alpha Processor", "description_llm": "Some other words"}
        },
        {
            "identifier": "tool_b",
            "_raw_metadata_snapshot": {
                "name": "Beta Processor",
                "description_llm": "Processes cherry and date fruits.",
                "tags": ["fruit", "processing"]
            }
        },
        {
            "identifier": "tool_c",
            "_raw_metadata_snapshot": {}
        }
    ]
    await provider.index_tools(tools_data)

    results_apple = await provider.find_tools("apple", top_k=1)
    assert len(results_apple) == 1
    assert results_apple[0].tool_identifier == "tool_a"

    results_cherry_fruit = await provider.find_tools("cherry fruit", top_k=1)
    assert len(results_cherry_fruit) == 1
    assert results_cherry_fruit[0].tool_identifier == "tool_b"

    results_processor_words = await provider.find_tools("processor words", top_k=2)
    assert len(results_processor_words) == 2
    assert results_processor_words[0].tool_identifier == "tool_a"
    assert results_processor_words[1].tool_identifier == "tool_b"


@pytest.mark.asyncio
async def test_km_find_tools_no_searchable_text_for_tool( # Renamed test for clarity
    keyword_lookup_provider: KeywordMatchLookupProvider, caplog: pytest.LogCaptureFixture
):
    provider = await keyword_lookup_provider
    tool_id_short_keywords = "id" # This identifier itself yields no keywords
    tools_data = [{"identifier": tool_id_short_keywords, "_raw_metadata_snapshot": {"name": "a"}}] # name "a" also yields no keywords

    # This is the *new* log message we expect due to the provider changes
    expected_log_message = f"{provider.plugin_id}: Tool '{tool_id_short_keywords}' yielded no usable keywords from its text fields. Skipping."

    original_level = module_logger_instance.level
    original_handlers = list(module_logger_instance.handlers)
    original_propagate = module_logger_instance.propagate
    module_logger_instance.setLevel(logging.DEBUG)
    module_logger_instance.propagate = True
    for handler in original_handlers: module_logger_instance.removeHandler(handler)

    try:
        with caplog.at_level(logging.DEBUG, logger=MODULE_LOGGER_NAME):
            await provider.index_tools(tools_data)
            results = await provider.find_tools("any_query_will_do")
            assert len(results) == 0

            assert any(
                rec.name == MODULE_LOGGER_NAME and
                rec.levelname == "DEBUG" and
                expected_log_message in rec.message
                for rec in caplog.records
            ), f"Log '{expected_log_message}' not found or not DEBUG. Caplog: {caplog.text}"
    finally:
        module_logger_instance.setLevel(original_level)
        module_logger_instance.propagate = original_propagate
        for handler_in_finally in list(module_logger_instance.handlers):
            module_logger_instance.removeHandler(handler_in_finally)
        for handler in original_handlers:
            module_logger_instance.addHandler(handler)


@pytest.mark.asyncio
async def test_km_extract_keywords_from_text_logic(keyword_lookup_provider: KeywordMatchLookupProvider):
    provider_instance = await keyword_lookup_provider
    assert provider_instance._extract_keywords_from_text("Hello World! This is a test.") == {"hello", "world", "this", "test"}
    assert provider_instance._extract_keywords_from_text("UPPERCASE_and_numbers123") == {"uppercase", "and", "numbers123"}
    assert provider_instance._extract_keywords_from_text("it a of to be") == set()
    assert provider_instance._extract_keywords_from_text("   leading and trailing spaces   ") == {"leading", "and", "trailing", "spaces"}
    assert provider_instance._extract_keywords_from_text("") == set()
    assert provider_instance._extract_keywords_from_text(None) == set() # type: ignore
    assert provider_instance._extract_keywords_from_text("word_with_underscore") == {"word", "with", "underscore"}


@pytest.mark.asyncio
async def test_km_teardown_clears_index(keyword_lookup_provider: KeywordMatchLookupProvider):
    provider = await keyword_lookup_provider
    await provider.index_tools([{"id":"t1", "name":"test"}])
    assert len(provider._indexed_tools_data) == 1

    await provider.teardown()
    assert len(provider._indexed_tools_data) == 0
