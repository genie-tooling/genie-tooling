"""Unit tests for WebPageLoader."""
import logging
from typing import Any, AsyncGenerator, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlparse

import httpx
import pytest
from genie_tooling.core.types import Document
from genie_tooling.document_loaders.impl.web_page import (
    WebPageLoader,
)


async def collect_docs_from_loader(loader_instance: WebPageLoader, url: str, config: Dict[str, Any] = None) -> List[Document]:
    results: List[Document] = []
    async for doc_item in loader_instance.load(url, config=config):
        if doc_item: results.append(doc_item)
    return results

@pytest.fixture
async def web_loader_fixture_obj() -> AsyncGenerator[WebPageLoader, None]:
    loader_instance = WebPageLoader()
    await loader_instance.setup(config={"use_trafilatura": False})
    yield loader_instance
    await loader_instance.teardown()

@pytest.fixture
def dummy_request() -> httpx.Request:
    return httpx.Request("GET", "http://dummy-request.com")

@pytest.mark.asyncio
async def test_load_bs4_not_available(caplog: pytest.LogCaptureFixture, dummy_request: httpx.Request):
    caplog.set_level(logging.WARNING)
    loader_for_no_bs4_test = WebPageLoader()
    html_content = "<html><head><title>BS4 Test No Lib Title</title></head><body>Raw HTML content</body></html>"
    mock_response = httpx.Response(200, text=html_content, headers={"content-type": "text/html; charset=utf-8"}, request=dummy_request)
    mock_client_instance = AsyncMock(spec=httpx.AsyncClient); mock_client_instance.get = AsyncMock(return_value=mock_response); mock_client_instance.aclose = AsyncMock()

    with patch("genie_tooling.document_loaders.impl.web_page.BeautifulSoup", None), \
         patch("httpx.AsyncClient", return_value=mock_client_instance):
        await loader_for_no_bs4_test.setup(config={"use_trafilatura": False})
        docs = await collect_docs_from_loader(loader_for_no_bs4_test, "http://example.com/no_bs4.html")

    assert len(docs) == 1
    assert docs[0].content == html_content
    assert docs[0].metadata["title"] == urlparse("http://example.com/no_bs4.html").netloc

    # Corrected log message assertion
    expected_log = f"{loader_for_no_bs4_test.plugin_id}: Neither Trafilatura nor BeautifulSoup4 available. Text extraction will be raw HTML."
    assert expected_log in caplog.text
    await loader_for_no_bs4_test.teardown()

@pytest.mark.asyncio
async def test_load_bs4_parse_error(web_loader_fixture_obj: AsyncGenerator[WebPageLoader, None], dummy_request: httpx.Request, caplog: pytest.LogCaptureFixture):
    actual_loader = await anext(web_loader_fixture_obj)

    caplog.set_level(logging.ERROR)
    html_content = "<html><head><title>BS4 Parse Error Page</title></head><body><p>Content before BS4 error.</p></body></html>"
    mock_response = httpx.Response(200, text=html_content, headers={"content-type": "text/html; charset=utf-8"}, request=dummy_request)
    mock_get_method = AsyncMock(return_value=mock_response)

    mock_bs4_constructor = MagicMock(name="MockBS4Constructor")
    mock_soup_instance = MagicMock(name="MockSoupInstance")

    mock_soup_instance_title = MagicMock(name="MockSoupTitle")
    mock_soup_instance_title.string = "BS4 Parse Error Page"
    mock_soup_instance.title = mock_soup_instance_title

    mock_soup_instance.get_text.side_effect = RuntimeError("BS4 custom parsing failure")
    mock_bs4_constructor.return_value = mock_soup_instance

    with patch.object(actual_loader._http_client, "get", mock_get_method), \
         patch("genie_tooling.document_loaders.impl.web_page.BeautifulSoup", mock_bs4_constructor):
        docs = await collect_docs_from_loader(actual_loader, "http://example.com/bs4_parse_fail.html")

    assert len(docs) == 1
    assert docs[0].content == html_content
    assert docs[0].metadata["title"] == "BS4 Parse Error Page"
    assert "Error parsing HTML with BeautifulSoup" in caplog.text
    assert "BS4 custom parsing failure" in caplog.text
