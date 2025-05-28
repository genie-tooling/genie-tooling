"""Unit tests for WebPageLoader."""
import logging
from typing import Any, AsyncGenerator, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlparse

import httpx
import pytest
from genie_tooling.core.types import Document
from genie_tooling.rag.plugins.impl.loaders.web_page import (
    WebPageLoader,
)

# Optional: Add a specific logger for this test file for targeted debugging
# test_logger_wp = logging.getLogger("test_web_page_loader_debug")
# test_logger_wp.setLevel(logging.DEBUG)
# import sys
# if not test_logger_wp.hasHandlers(): # Avoid adding multiple handlers during re-runs
#     handler = logging.StreamHandler(sys.stdout)
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s')
#     handler.setFormatter(formatter)
#     test_logger_wp.addHandler(handler)


async def collect_docs_from_loader(loader_instance: WebPageLoader, url: str, config: Dict[str, Any] = None) -> List[Document]:
    results: List[Document] = []
    # test_logger_wp.debug(f"collect_docs_from_loader: Starting for URL '{url}' with loader ID {id(loader_instance)}")
    async for doc_item in loader_instance.load(url, config=config):
        # test_logger_wp.debug(f"collect_docs_from_loader: Collected doc: {doc_item.id if doc_item else 'None'}")
        if doc_item: # Ensure doc_item is not None before accessing attributes
             results.append(doc_item)
    # test_logger_wp.debug(f"collect_docs_from_loader: Finished for URL '{url}', collected {len(results)} docs.")
    return results

@pytest.fixture
async def web_loader_fixture_obj() -> AsyncGenerator[WebPageLoader, None]:
    loader_instance = WebPageLoader()
    await loader_instance.setup()
    yield loader_instance
    await loader_instance.teardown()


@pytest.fixture
def dummy_request() -> httpx.Request:
    """Provides a dummy httpx.Request object for mock responses."""
    return httpx.Request("GET", "http://dummy-request.com")


@pytest.mark.asyncio
async def test_setup_initializes_client(web_loader_fixture_obj: AsyncGenerator[WebPageLoader, None]):
    actual_loader = await anext(web_loader_fixture_obj)
    assert actual_loader._http_client is not None, "HTTP client was not initialized in setup"
    assert isinstance(actual_loader._http_client, httpx.AsyncClient)

@pytest.mark.asyncio
async def test_load_successful_html(web_loader_fixture_obj: AsyncGenerator[WebPageLoader, None], dummy_request: httpx.Request):
    actual_loader = await anext(web_loader_fixture_obj)
    assert actual_loader._http_client is not None, "_http_client is None on actual_loader instance after anext()"

    html_content = "<html><head><title>Test Page Success</title></head><body><p>Success content</p></body></html>"
    mock_response = httpx.Response(200, text=html_content, headers={"content-type": "text/html; charset=utf-8"}, request=dummy_request)

    mock_get_method = AsyncMock(return_value=mock_response, name="mock_get_for_successful_html")

    with patch.object(actual_loader._http_client, "get", mock_get_method):
        docs = await collect_docs_from_loader(actual_loader, "http://example.com/success.html")

    assert len(docs) == 1, f"Expected 1 doc, got {len(docs)}. Content: {html_content[:100]}"
    if docs:
        doc_item = docs[0]
        # BeautifulSoup's get_text(strip=True) often includes title text if present in <head> and visible.
        # For this specific HTML, it will be "Test Page SuccessSuccess content"
        expected_text_content = "Test Page Success Success content"
        assert doc_item.content.strip() == expected_text_content
        assert doc_item.metadata["title"] == "Test Page Success"
    mock_get_method.assert_awaited_once_with("http://example.com/success.html")


@pytest.mark.asyncio
async def test_load_non_html_content_type(web_loader_fixture_obj: AsyncGenerator[WebPageLoader, None], dummy_request: httpx.Request, caplog: pytest.LogCaptureFixture):
    actual_loader = await anext(web_loader_fixture_obj)
    assert actual_loader._http_client is not None, "_http_client is None on actual_loader instance after anext()"
    caplog.set_level(logging.WARNING)
    raw_content = "This is plain text for non-html test."
    mock_response = httpx.Response(200, text=raw_content, headers={"content-type": "text/plain"}, request=dummy_request) # Added request

    mock_get_method = AsyncMock(return_value=mock_response, name="mock_get_for_non_html_type")
    with patch.object(actual_loader._http_client, "get", mock_get_method):
        docs = await collect_docs_from_loader(actual_loader, "http://example.com/file.txt")

    assert len(docs) == 1, f"Expected 1 doc, got {len(docs)}"
    if docs:
        assert docs[0].content == raw_content
        assert docs[0].metadata["content_type"] == "text/plain"
    assert "Content type for http://example.com/file.txt is 'text/plain', not HTML. Using raw content." in caplog.text
    mock_get_method.assert_awaited_once_with("http://example.com/file.txt")

@pytest.mark.asyncio
async def test_load_bs4_not_available(caplog: pytest.LogCaptureFixture, dummy_request: httpx.Request):
    caplog.set_level(logging.WARNING)
    loader_for_no_bs4_test = WebPageLoader()

    html_content = "<html><head><title>BS4 Test No Lib Title</title></head><body>Raw HTML content</body></html>"
    # Added dummy_request here
    mock_response = httpx.Response(200, text=html_content, headers={"content-type": "text/html; charset=utf-8"}, request=dummy_request)

    mock_client_instance = AsyncMock(spec=httpx.AsyncClient, name="ClientForNoBS4")
    mock_client_instance.get = AsyncMock(return_value=mock_response, name="GetForNoBS4")
    mock_client_instance.aclose = AsyncMock(name="ACloseForNoBS4")

    with patch("genie_tooling.rag.plugins.impl.loaders.web_page.BeautifulSoup", None), \
         patch("httpx.AsyncClient", return_value=mock_client_instance) as mock_constructor:

        await loader_for_no_bs4_test.setup()
        mock_constructor.assert_called_once()
        assert loader_for_no_bs4_test._http_client is mock_client_instance

        docs = await collect_docs_from_loader(loader_for_no_bs4_test, "http://example.com/no_bs4.html")

    assert len(docs) == 1, f"Expected 1 doc, got {len(docs)}"
    if docs:
        assert docs[0].content == html_content
        # When BS4 is None, title falls back to netloc
        assert docs[0].metadata["title"] == urlparse("http://example.com/no_bs4.html").netloc
    assert "'beautifulsoup4' library not installed" in caplog.text

    await loader_for_no_bs4_test.teardown()
    mock_client_instance.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_load_http_status_error(web_loader_fixture_obj: AsyncGenerator[WebPageLoader, None], caplog: pytest.LogCaptureFixture):
    actual_loader = await anext(web_loader_fixture_obj)
    assert actual_loader._http_client is not None
    caplog.set_level(logging.ERROR)
    mock_request_for_error = httpx.Request("GET", "http://example.com/notfound")
    mock_response_obj = httpx.Response(404, text="Not Found", request=mock_request_for_error)

    mock_get_method = AsyncMock(side_effect=httpx.HTTPStatusError(
        message="404 Client Error", request=mock_request_for_error, response=mock_response_obj
    ), name="mock_get_for_status_error")

    with patch.object(actual_loader._http_client, "get", mock_get_method):
        docs = await collect_docs_from_loader(actual_loader, "http://example.com/notfound")

    assert len(docs) == 0
    assert "HTTP status error loading http://example.com/notfound. Status: 404" in caplog.text
    mock_get_method.assert_awaited_once_with("http://example.com/notfound")

@pytest.mark.asyncio
async def test_load_request_error(web_loader_fixture_obj: AsyncGenerator[WebPageLoader, None], caplog: pytest.LogCaptureFixture):
    actual_loader = await anext(web_loader_fixture_obj)
    assert actual_loader._http_client is not None
    caplog.set_level(logging.ERROR)

    mock_request_for_req_error = httpx.Request("GET", "http://example.com/networkfail")
    mock_get_method = AsyncMock(side_effect=httpx.RequestError(
        "Network error", request=mock_request_for_req_error
    ), name="mock_get_for_request_error")

    with patch.object(actual_loader._http_client, "get", mock_get_method):
        docs = await collect_docs_from_loader(actual_loader, "http://example.com/networkfail")

    assert len(docs) == 0
    assert "Request error loading http://example.com/networkfail. Error: Network error" in caplog.text
    mock_get_method.assert_awaited_once_with("http://example.com/networkfail")

@pytest.mark.asyncio
async def test_load_bs4_parse_error(web_loader_fixture_obj: AsyncGenerator[WebPageLoader, None], dummy_request: httpx.Request, caplog: pytest.LogCaptureFixture):
    actual_loader = await anext(web_loader_fixture_obj)
    assert actual_loader._http_client is not None
    caplog.set_level(logging.ERROR)
    html_content = "<html><head><title>BS4 Parse Error Page</title></head><body><p>Content before BS4 error.</p>"
    mock_response = httpx.Response(200, text=html_content, headers={"content-type": "text/html; charset=utf-8"}, request=dummy_request)

    mock_get_method = AsyncMock(return_value=mock_response, name="mock_get_for_bs4_parse_error")

    mock_bs4_constructor = MagicMock(name="MockBS4Constructor")
    mock_soup_instance = MagicMock(name="MockSoupInstance")
    mock_soup_instance.get_text.side_effect = RuntimeError("BS4 custom parsing failure")
    mock_soup_instance_title = MagicMock(name="MockSoupTitle")
    mock_soup_instance_title.string = "BS4 Parse Error Page"
    mock_soup_instance.title = mock_soup_instance_title
    mock_bs4_constructor.return_value = mock_soup_instance

    with patch.object(actual_loader._http_client, "get", mock_get_method), \
         patch("genie_tooling.rag.plugins.impl.loaders.web_page.BeautifulSoup", mock_bs4_constructor):
        docs = await collect_docs_from_loader(actual_loader, "http://example.com/bs4_parse_fail.html")

    assert len(docs) == 1, f"Expected 1 doc, got {len(docs)}"
    if docs:
        assert docs[0].content == html_content
        assert docs[0].metadata["title"] == "BS4 Parse Error Page"
    assert "Error parsing HTML with BeautifulSoup" in caplog.text
    assert "BS4 custom parsing failure" in caplog.text
    mock_get_method.assert_awaited_once_with("http://example.com/bs4_parse_fail.html")

@pytest.mark.asyncio
async def test_load_empty_response_content(web_loader_fixture_obj: AsyncGenerator[WebPageLoader, None], dummy_request: httpx.Request):
    actual_loader = await anext(web_loader_fixture_obj)
    assert actual_loader._http_client is not None
    mock_response = httpx.Response(200, text="", headers={"content-type": "text/html; charset=utf-8"}, request=dummy_request)

    mock_get_method = AsyncMock(return_value=mock_response, name="mock_get_for_empty_resp")

    with patch.object(actual_loader._http_client, "get", mock_get_method):
        docs = await collect_docs_from_loader(actual_loader, "http://example.com/empty_page.html")

    assert len(docs) == 1, f"Expected 1 doc, got {len(docs)}"
    if docs:
        assert docs[0].content == ""
        assert docs[0].metadata["title"] == "example.com"
    mock_get_method.assert_awaited_once_with("http://example.com/empty_page.html")

@pytest.mark.asyncio
async def test_teardown_closes_client():
    loader = WebPageLoader()

    mock_client_instance = AsyncMock(spec=httpx.AsyncClient, name="ClientForTeardownTestExplicit")
    mock_client_instance.aclose = AsyncMock(name="ACloseForTeardownTestExplicit")

    with patch("httpx.AsyncClient", return_value=mock_client_instance) as mock_async_client_constructor:
        await loader.setup()

    mock_async_client_constructor.assert_called_once()
    assert loader._http_client is mock_client_instance

    await loader.teardown()

    mock_client_instance.aclose.assert_awaited_once()
    assert loader._http_client is None


@pytest.mark.asyncio
async def test_load_client_not_initialized(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR)
    loader_no_setup = WebPageLoader()
    docs = await collect_docs_from_loader(loader_no_setup, "http://example.com/client_not_init.html")
    assert len(docs) == 0
    assert "HTTP client not initialized" in caplog.text
