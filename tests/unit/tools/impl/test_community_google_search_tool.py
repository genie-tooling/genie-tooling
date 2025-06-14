### tests/unit/tools/impl/test_community_google_search_tool.py
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genie_tooling.tools.impl.community_google_search_tool import (
    community_google_search,
)

# Constants for patching and logging
SEARCH_MODULE_PATH = "genie_tooling.tools.impl.community_google_search_tool"
SEARCH_LOGGER_NAME = SEARCH_MODULE_PATH
GOOGLESEARCH_LIB_AVAILABLE_PATH = f"{SEARCH_MODULE_PATH}.GOOGLESEARCH_LIB_AVAILABLE"
SEARCH_FUNCTION_PATH = f"{SEARCH_MODULE_PATH}.googlesearch_search"


@pytest.fixture()
def mock_googlesearch_function() -> MagicMock:
    """Provides a mock for the `googlesearch.search` function."""
    return MagicMock()


@pytest.fixture()
def mock_context_and_kp() -> tuple[dict, AsyncMock]:
    """Provides mock context and key_provider arguments for the tool signature."""
    mock_context = {}
    mock_kp = AsyncMock()
    return mock_context, mock_kp


@pytest.mark.asyncio()
class TestCommunityGoogleSearchTool:
    """Tests for the community_google_search tool."""

    @patch(GOOGLESEARCH_LIB_AVAILABLE_PATH, True)
    @patch(SEARCH_FUNCTION_PATH)
    async def test_execute_success_basic(
        self, mock_search: MagicMock, mock_context_and_kp: tuple
    ):
        """Test basic search returning a list of URLs."""
        mock_search.return_value = ["http://example.com/1", "http://example.com/2"]
        mock_context, mock_kp = mock_context_and_kp

        result = await community_google_search(
            query="test query", num_results=2, context=mock_context, key_provider=mock_kp
        )

        assert result["error"] is None
        assert result["results"] == ["http://example.com/1", "http://example.com/2"]
        mock_search.assert_called_once()
        # Check that the query and num_results were passed correctly
        call_args, call_kwargs = mock_search.call_args
        assert call_args[0] == "test query"
        assert call_kwargs.get("num_results") == 2
        assert call_kwargs.get("advanced") is False

    @patch(GOOGLESEARCH_LIB_AVAILABLE_PATH, True)
    @patch(SEARCH_FUNCTION_PATH)
    async def test_execute_success_advanced(
        self, mock_search: MagicMock, mock_context_and_kp: tuple
    ):
        """Test advanced search returning structured data."""

        class MockSearchResult:
            def __init__(self, title, url, description):
                self.title = title
                self.url = url
                self.description = description

        mock_search.return_value = [
            MockSearchResult("Title 1", "url1", "Desc 1"),
            MockSearchResult("Title 2", "url2", "Desc 2"),
        ]
        mock_context, mock_kp = mock_context_and_kp

        result = await community_google_search(
            query="advanced query", advanced=True, context=mock_context, key_provider=mock_kp
        )

        assert result["error"] is None
        assert len(result["results"]) == 2
        assert result["results"][0] == {"title": "Title 1", "url": "url1", "description": "Desc 1"}
        mock_search.assert_called_once()
        assert mock_search.call_args.kwargs.get("advanced") is True

    @patch(GOOGLESEARCH_LIB_AVAILABLE_PATH, True)
    @patch(SEARCH_FUNCTION_PATH)
    async def test_execute_search_library_raises_exception(
        self, mock_search: MagicMock, mock_context_and_kp: tuple, caplog: pytest.LogCaptureFixture
    ):
        """Test handling of exceptions from the underlying search library."""
        caplog.set_level(logging.ERROR, logger=SEARCH_LOGGER_NAME)
        mock_search.side_effect = Exception("HTTP 429 Too Many Requests")
        mock_context, mock_kp = mock_context_and_kp

        result = await community_google_search(
            query="error query", context=mock_context, key_provider=mock_kp
        )

        assert "An unexpected error occurred during search" in result["error"]
        assert "HTTP 429 Too Many Requests" in result["error"]
        assert result["results"] == []
        assert "Error during search for 'error query'" in caplog.text

    @patch(GOOGLESEARCH_LIB_AVAILABLE_PATH, False)
    async def test_execute_library_not_available(
        self, mock_context_and_kp: tuple, caplog: pytest.LogCaptureFixture
    ):
        """Test behavior when the googlesearch-python library is not installed."""
        caplog.set_level(logging.ERROR, logger=SEARCH_LOGGER_NAME)
        mock_context, mock_kp = mock_context_and_kp

        result = await community_google_search(
            query="any query", context=mock_context, key_provider=mock_kp
        )

        assert "googlesearch library not installed or available" in result["error"]
        assert "googlesearch library is not available" in caplog.text

    @patch(GOOGLESEARCH_LIB_AVAILABLE_PATH, True)
    @patch(SEARCH_FUNCTION_PATH)
    async def test_execute_with_unique_flag(
        self, mock_search: MagicMock, mock_context_and_kp: tuple
    ):
        """Test that the unique flag correctly filters duplicate URLs."""
        mock_search.return_value = [
            "http://example.com/page1",
            "http://example.com/page2",
            "http://example.com/page1",  # Duplicate
        ]
        mock_context, mock_kp = mock_context_and_kp

        result = await community_google_search(
            query="unique results", unique=True, context=mock_context, key_provider=mock_kp
        )

        assert result["results"] == ["http://example.com/page1", "http://example.com/page2"]
