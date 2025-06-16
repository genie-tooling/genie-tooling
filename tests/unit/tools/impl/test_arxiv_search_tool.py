# tests/unit/tools/impl/test_arxiv_search_tool.py
import logging
from unittest.mock import MagicMock, patch

import pytest
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.impl.arxiv_search_tool import ArxivSearchTool

# Constants for patching and logging
ARXIV_MODULE_PATH = "genie_tooling.tools.impl.arxiv_search_tool"
ARXIV_AVAILABLE_PATH = f"{ARXIV_MODULE_PATH}.ARXIV_AVAILABLE"
ARXIV_CLASS_PATH = f"{ARXIV_MODULE_PATH}.arxiv"
TOOL_LOGGER_NAME = f"{ARXIV_MODULE_PATH}.logger"


@pytest.fixture()
def mock_arxiv_search_class() -> MagicMock:
    """Provides a mock for the `arxiv.Search` class."""
    return MagicMock()


@pytest.fixture()
def mock_key_provider() -> MagicMock:
    """Provides a mock KeyProvider for the tool's execute signature."""
    return MagicMock(spec=KeyProvider)


@pytest.fixture()
async def arxiv_tool() -> ArxivSearchTool:
    """Provides a setup instance of the ArxivSearchTool."""
    tool = ArxivSearchTool()
    await tool.setup()
    return tool


@pytest.mark.asyncio()
class TestArxivSearchTool:
    """Tests for the ArxivSearchTool."""

    async def test_get_metadata(self, arxiv_tool: ArxivSearchTool):
        """Verify the metadata is correctly structured."""
        tool = await arxiv_tool
        metadata = await tool.get_metadata()

        assert metadata["identifier"] == "arxiv_search_tool"
        assert "ArXiv Search" in metadata["name"]
        assert "query" in metadata["input_schema"]["properties"]
        assert "max_results" in metadata["input_schema"]["properties"]
        assert "results" in metadata["output_schema"]["properties"]
        assert "error" in metadata["output_schema"]["properties"]

    @patch(ARXIV_AVAILABLE_PATH, True)
    @patch(ARXIV_CLASS_PATH)
    async def test_execute_success(
        self, mock_arxiv_module: MagicMock, arxiv_tool: ArxivSearchTool, mock_key_provider: MagicMock
    ):
        """Test successful search returning structured results."""
        tool = await arxiv_tool
        mock_search_instance = MagicMock()
        mock_arxiv_module.Search.return_value = mock_search_instance

        # Mock the result object structure
        mock_result = MagicMock()
        mock_result.entry_id = "1234.5678"
        mock_result.title = "Quantum Computing Explained"
        mock_result.summary = "A paper about quantum bits."

        # FIX: Correctly mock the author object and its 'name' attribute
        mock_author = MagicMock()
        mock_author.name = "Dr. Quantum"
        mock_result.authors = [mock_author]

        mock_result.published.isoformat.return_value = "2023-10-27T10:00:00Z"
        mock_result.pdf_url = "http://arxiv.org/pdf/1234.5678"

        # The .results() method should be an iterable
        mock_search_instance.results.return_value = [mock_result]

        params = {"query": "quantum computing", "max_results": 1}
        result = await tool.execute(params, mock_key_provider, context={})

        assert result["error"] is None
        assert len(result["results"]) == 1
        first_res = result["results"][0]
        assert first_res["title"] == "Quantum Computing Explained"
        assert first_res["authors"] == ["Dr. Quantum"]  # This assertion should now pass
        assert first_res["pdf_url"] == "https://arxiv.org/pdf/1234.5678"

        mock_arxiv_module.Search.assert_called_once()
        call_args, call_kwargs = mock_arxiv_module.Search.call_args
        # FIX: The query is the first positional argument
        assert call_args[0] == "quantum computing"
        assert call_kwargs["max_results"] == 1

    @patch(ARXIV_AVAILABLE_PATH, True)
    @patch(ARXIV_CLASS_PATH)
    async def test_execute_search_library_raises_exception(
        self, mock_arxiv_module: MagicMock, arxiv_tool: ArxivSearchTool, mock_key_provider: MagicMock, caplog: pytest.LogCaptureFixture
    ):
        """Test handling of exceptions from the underlying arxiv library."""
        tool = await arxiv_tool
        caplog.set_level(logging.ERROR, logger=TOOL_LOGGER_NAME)
        mock_arxiv_module.Search.side_effect = Exception("ArXiv API is down")

        params = {"query": "error query"}
        result = await tool.execute(params, mock_key_provider, context={})

        assert "An unexpected error occurred during ArXiv search: ArXiv API is down" in result["error"]
        assert result["results"] == []
        assert "Error during ArXiv search for 'error query'" in caplog.text

    @patch(ARXIV_AVAILABLE_PATH, False)
    async def test_execute_library_not_available(
        self, arxiv_tool: ArxivSearchTool, mock_key_provider: MagicMock
    ):
        """Test behavior when the arxiv library is not installed."""
        tool = await arxiv_tool
        result = await tool.execute({"query": "any"}, mock_key_provider, context={})
        assert "arxiv library not installed" in result["error"]

    @patch(ARXIV_AVAILABLE_PATH, True)
    @patch(ARXIV_CLASS_PATH)
    async def test_execute_no_results_found(
        self, mock_arxiv_module: MagicMock, arxiv_tool: ArxivSearchTool, mock_key_provider: MagicMock
    ):
        """Test correct handling when the search yields no results."""
        tool = await arxiv_tool
        mock_search_instance = MagicMock()
        mock_arxiv_module.Search.return_value = mock_search_instance
        mock_search_instance.results.return_value = []  # Empty results

        result = await tool.execute({"query": "obscure topic"}, mock_key_provider, context={})

        assert result["error"] is None
        assert result["results"] == []
