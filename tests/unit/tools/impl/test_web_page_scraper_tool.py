# tests/unit/tools/impl/test_web_page_scraper_tool.py
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.impl.web_page_scraper_tool import WebPageScraperTool

# Added anext for Python < 3.10 compatibility
try:
    from asyncio import anext
except ImportError:
    async def anext(ait):
        return await ait.__anext__()


TOOL_MODULE_PATH = "genie_tooling.tools.impl.web_page_scraper_tool"
TOOL_LOGGER_NAME = TOOL_MODULE_PATH + ".logger"


@pytest.fixture()
async def scraper_tool() -> WebPageScraperTool:
    """Provides a setup instance of the WebPageScraperTool."""
    tool = WebPageScraperTool()
    await tool.setup()
    yield tool
    await tool.teardown()


@pytest.fixture()
def mock_key_provider() -> MagicMock:
    """Provides a mock KeyProvider."""
    return MagicMock(spec=KeyProvider)


@pytest.fixture()
def dummy_request() -> httpx.Request:
    """Provides a dummy httpx.Request object for mock responses."""
    return httpx.Request("GET", "http://dummy-request.com")


@pytest.mark.asyncio()
class TestWebPageScraperTool:
    async def test_setup_and_teardown_initializes_client(self, mocker):
        """Verify that setup creates the client and teardown closes it."""
        mock_async_client_constructor = mocker.patch(
            f"{TOOL_MODULE_PATH}.httpx.AsyncClient"
        )
        mock_client_instance = mock_async_client_constructor.return_value
        mock_client_instance.aclose = AsyncMock()

        tool = WebPageScraperTool()
        await tool.setup()
        assert tool._http_client is mock_client_instance

        await tool.teardown()
        mock_client_instance.aclose.assert_awaited_once()
        assert tool._http_client is None

    async def test_execute_success_with_trafilatura(
        self, scraper_tool, mock_key_provider, dummy_request
    ):
        """Test happy path where Trafilatura successfully extracts content."""
        html_content = "<html><head><title>Test Title</title></head><body>Main content here.</body></html>"
        tool = await anext(scraper_tool)
        mock_response = httpx.Response(200, text=html_content, request=dummy_request)

        with patch(f"{TOOL_MODULE_PATH}.trafilatura", create=True) as mock_trafilatura, \
             patch(f"{TOOL_MODULE_PATH}.BeautifulSoup", create=True) as mock_bs:

            mock_trafilatura.extract.return_value = "Main content here."
            mock_soup_instance = MagicMock()
            mock_soup_instance.title.string = "Test Title"
            mock_bs.return_value = mock_soup_instance
            # Patch the client on the already-setup tool instance
            tool._http_client.get = AsyncMock(return_value=mock_response) # type: ignore

            result = await tool.execute(
                {"url": "http://example.com"}, mock_key_provider, {}
            )

        assert result["error"] is None
        assert result["content"] == "Main content here."
        assert result["title"] == "Test Title"
        mock_trafilatura.extract.assert_called_once()

    async def test_execute_success_with_bs4_fallback(
        self, scraper_tool, mock_key_provider, dummy_request
    ):
        """Test fallback to BeautifulSoup when Trafilatura fails."""
        html_content = "<html><head><title>BS4 Page</title></head><body><p>BS4 content</p><script>bad</script></body></html>"
        tool = await anext(scraper_tool)
        mock_response = httpx.Response(200, text=html_content, request=dummy_request)

        with patch(f"{TOOL_MODULE_PATH}.trafilatura", create=True) as mock_trafilatura, \
             patch(f"{TOOL_MODULE_PATH}.BeautifulSoup") as mock_bs4_constructor, \
             patch.object(tool, "_http_client", new_callable=AsyncMock) as mock_client:
            mock_trafilatura.extract.return_value = None
            mock_soup_instance = MagicMock()
            mock_soup_instance.get_text.return_value = "BS4 content"
            mock_soup_instance.title = MagicMock()
            mock_soup_instance.title.string = "BS4 Page"
            mock_bs4_constructor.return_value = mock_soup_instance
            mock_client.get.return_value = mock_response

            result = await tool.execute(
                {"url": "http://example.com"}, mock_key_provider, {}
            )

        assert result["error"] is None
        assert result["content"] == "BS4 content"
        assert result["title"] == "BS4 Page"

    async def test_execute_http_error(
        self, scraper_tool, mock_key_provider, dummy_request
    ):
        """Test handling of HTTP status errors."""
        tool = await anext(scraper_tool)
        with patch.object(tool, "_http_client", new_callable=AsyncMock) as mock_client:
            mock_client.get.side_effect=httpx.RequestError(
                    "Network request error", request=dummy_request
                )
            result = await tool.execute(
                {"url": "http://example.com/404"}, mock_key_provider, {}
            )
        assert "Network request error" in result["error"]

    async def test_execute_no_url_parameter(
        self, scraper_tool, mock_key_provider
    ):
        """Test error handling when URL parameter is missing."""
        tool = await anext(scraper_tool)
        result = await tool.execute({}, mock_key_provider, {})
        assert "URL must be a non-empty string" in result["error"]

    async def test_execute_client_not_initialized(
        self, scraper_tool, mock_key_provider
    ):
        """Test error handling if execute is called before setup."""
        tool = await anext(scraper_tool)
        tool._http_client = None
        result = await tool.execute(
            {"url": "http://example.com"}, mock_key_provider, {}
        )
        assert "HTTP client not initialized" in result["error"]
