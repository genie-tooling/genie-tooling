### tests/unit/rag/plugins/impl/loaders/test_web_page_loader.py
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

# Logger for the module under test
LOADER_LOGGER_NAME = "genie_tooling.document_loaders.impl.web_page"


async def collect_docs_from_loader(loader_instance: WebPageLoader, url: str, config: Dict[str, Any] = None) -> List[Document]:
    results: List[Document] = []
    async for doc_item in loader_instance.load(url, config=config):
        if doc_item:
            results.append(doc_item)
    return results

@pytest.fixture
async def web_loader() -> AsyncGenerator[WebPageLoader, None]:
    loader = WebPageLoader()
    try:
        yield loader
    finally:
        await loader.teardown()

@pytest.fixture
def dummy_request() -> httpx.Request:
    return httpx.Request("GET", "http://dummy-request.com")


@pytest.mark.asyncio
async def test_load_bs4_not_available(web_loader: AsyncGenerator[WebPageLoader, None], caplog: pytest.LogCaptureFixture, dummy_request: httpx.Request):
    actual_loader = await anext(web_loader)
    caplog.set_level(logging.WARNING, logger=LOADER_LOGGER_NAME)
    html_content = "<html><head><title>BS4 Test No Lib Title</title></head><body>Raw HTML content</body></html>"
    mock_response = httpx.Response(200, text=html_content, headers={"content-type": "text/html; charset=utf-8"}, request=dummy_request)
    mock_client_instance = AsyncMock(spec=httpx.AsyncClient)
    mock_client_instance.get = AsyncMock(return_value=mock_response)
    mock_client_instance.aclose = AsyncMock()

    with patch("genie_tooling.document_loaders.impl.web_page.BeautifulSoup", None), \
         patch("genie_tooling.document_loaders.impl.web_page.trafilatura", None), \
         patch("httpx.AsyncClient", return_value=mock_client_instance):
        await actual_loader.setup(config={"use_trafilatura": False})
        docs = await collect_docs_from_loader(actual_loader, "http://example.com/no_bs4.html")

    assert len(docs) == 1
    assert docs[0].content == html_content
    assert docs[0].metadata["title"] == urlparse("http://example.com/no_bs4.html").netloc

    expected_log = f"{actual_loader.plugin_id}: Neither Trafilatura nor BeautifulSoup4 available. Text extraction will be raw HTML."
    assert any(expected_log in record.message for record in caplog.records)

@pytest.mark.asyncio
async def test_load_bs4_parse_error(web_loader: AsyncGenerator[WebPageLoader, None], dummy_request: httpx.Request, caplog: pytest.LogCaptureFixture):
    actual_loader = await anext(web_loader)
    caplog.set_level(logging.ERROR, logger=LOADER_LOGGER_NAME)
    html_content = "<html><head><title>BS4 Parse Error Page</title></head><body><p>Content before BS4 error.</p></body></html>"
    mock_response = httpx.Response(200, text=html_content, headers={"content-type": "text/html; charset=utf-8"}, request=dummy_request)

    mock_client_instance = AsyncMock(spec=httpx.AsyncClient)
    mock_client_instance.get = AsyncMock(return_value=mock_response)
    mock_client_instance.aclose = AsyncMock()

    mock_bs4_constructor = MagicMock(name="MockBS4Constructor")
    mock_soup_instance = MagicMock(name="MockSoupInstance")
    mock_soup_instance_title = MagicMock(name="MockSoupTitle")
    mock_soup_instance_title.string = "BS4 Parse Error Page"
    mock_soup_instance.title = mock_soup_instance_title
    mock_soup_instance.get_text.side_effect = RuntimeError("BS4 custom parsing failure")
    mock_bs4_constructor.return_value = mock_soup_instance

    with patch("httpx.AsyncClient", return_value=mock_client_instance), \
         patch("genie_tooling.document_loaders.impl.web_page.BeautifulSoup", mock_bs4_constructor):
        await actual_loader.setup(config={"use_trafilatura": False})
        docs = await collect_docs_from_loader(actual_loader, "http://example.com/bs4_parse_fail.html")

    assert len(docs) == 1, f"Docs found: {[d.id for d in docs]}"
    assert docs[0].content == html_content
    assert docs[0].metadata["title"] == "BS4 Parse Error Page"
    assert any("Error parsing HTML with BeautifulSoup" in record.message and "BS4 custom parsing failure" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_load_successful_trafilatura(web_loader: AsyncGenerator[WebPageLoader, None], dummy_request: httpx.Request):
    actual_loader = await anext(web_loader)
    html_content = "<html><body><article><p>Main content by Trafilatura.</p></article><footer>Footer</footer></body></html>"
    trafilatura_extracted_text = "Main content by Trafilatura."
    mock_response = httpx.Response(200, text=html_content, headers={"content-type": "text/html"}, request=dummy_request)

    mock_client_instance = AsyncMock(spec=httpx.AsyncClient)
    mock_client_instance.get = AsyncMock(return_value=mock_response)
    mock_client_instance.aclose = AsyncMock()

    mock_trafilatura_module = MagicMock()
    mock_trafilatura_module.extract = MagicMock(return_value=trafilatura_extracted_text)

    with patch("genie_tooling.document_loaders.impl.web_page.trafilatura", mock_trafilatura_module), \
         patch("genie_tooling.document_loaders.impl.web_page.BeautifulSoup", MagicMock()), \
         patch("httpx.AsyncClient", return_value=mock_client_instance):
        await actual_loader.setup(config={"use_trafilatura": True})
        docs = await collect_docs_from_loader(actual_loader, "http://example.com/trafilatura_success.html")

    assert len(docs) == 1, f"Docs found: {[d.id for d in docs]}"
    assert docs[0].content == trafilatura_extracted_text
    mock_trafilatura_module.extract.assert_called_once()

@pytest.mark.asyncio
async def test_load_trafilatura_fails_fallback_bs4(web_loader: AsyncGenerator[WebPageLoader, None], dummy_request: httpx.Request):
    actual_loader = await anext(web_loader)
    html_content = "<html><head><title>Fallback Test</title></head><body><p>BS4 should get this.</p></body></html>"
    bs4_extracted_text = "BS4 should get this."
    mock_response = httpx.Response(200, text=html_content, headers={"content-type": "text/html"}, request=dummy_request)

    mock_client_instance = AsyncMock(spec=httpx.AsyncClient)
    mock_client_instance.get = AsyncMock(return_value=mock_response)
    mock_client_instance.aclose = AsyncMock()

    mock_trafilatura_module = MagicMock()
    mock_trafilatura_module.extract = MagicMock(return_value=None)

    mock_bs4_soup_instance = MagicMock()
    mock_bs4_soup_instance.get_text.return_value = bs4_extracted_text
    mock_bs4_soup_instance.title.string = "Fallback Test"
    mock_bs4_constructor = MagicMock(return_value=mock_bs4_soup_instance)

    with patch("genie_tooling.document_loaders.impl.web_page.trafilatura", mock_trafilatura_module), \
         patch("genie_tooling.document_loaders.impl.web_page.BeautifulSoup", mock_bs4_constructor), \
         patch("httpx.AsyncClient", return_value=mock_client_instance):
        await actual_loader.setup(config={"use_trafilatura": True})
        docs = await collect_docs_from_loader(actual_loader, "http://example.com/traf_fail_bs4_ok.html")

    assert len(docs) == 1, f"Docs found: {[d.id for d in docs]}"
    assert docs[0].content == bs4_extracted_text
    assert docs[0].metadata["title"] == "Fallback Test"
    mock_trafilatura_module.extract.assert_called_once()
    mock_bs4_constructor.assert_called()

@pytest.mark.asyncio
async def test_load_trafilatura_fails_no_bs4_fallback_raw(web_loader: AsyncGenerator[WebPageLoader, None], dummy_request: httpx.Request, caplog: pytest.LogCaptureFixture):
    actual_loader = await anext(web_loader)
    # Capture INFO logs to check the setup message
    caplog.set_level(logging.INFO, logger=LOADER_LOGGER_NAME)
    html_content = "<html><head><title>Raw Fallback</title></head><body>Raw HTML content here.</body></html>"
    mock_response = httpx.Response(200, text=html_content, headers={"content-type": "text/html"}, request=dummy_request)

    mock_client_instance = AsyncMock(spec=httpx.AsyncClient)
    mock_client_instance.get = AsyncMock(return_value=mock_response)
    mock_client_instance.aclose = AsyncMock()

    mock_trafilatura_module = MagicMock()
    mock_trafilatura_module.extract = MagicMock(return_value=None) # Trafilatura fails to extract

    with patch("genie_tooling.document_loaders.impl.web_page.trafilatura", mock_trafilatura_module), \
         patch("genie_tooling.document_loaders.impl.web_page.BeautifulSoup", None), \
         patch("httpx.AsyncClient", return_value=mock_client_instance):
        # Setup: use_trafilatura=True, trafilatura module is mocked (present), BeautifulSoup is None
        await actual_loader.setup(config={"use_trafilatura": True})
        docs = await collect_docs_from_loader(actual_loader, "http://example.com/traf_fail_no_bs4.html")

    assert len(docs) == 1, f"Docs found: {[d.id for d in docs]}"
    assert docs[0].content == html_content # Correctly falls back to raw HTML
    assert docs[0].metadata["title"] == "example.com" # Correctly defaults title

    # Check the setup log: Trafilatura is configured and available (mocked)
    expected_setup_log = f"{actual_loader.plugin_id}: Trafilatura will be used for content extraction."
    assert any(expected_setup_log in record.message and record.levelno == logging.INFO for record in caplog.records)

    # Ensure the previously asserted WARNING (which was incorrect for this patching) is NOT present
    unexpected_warning_log = f"{actual_loader.plugin_id}: Neither Trafilatura nor BeautifulSoup4 available."
    assert not any(unexpected_warning_log in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_load_successful_bs4_only(web_loader: AsyncGenerator[WebPageLoader, None], dummy_request: httpx.Request):
    actual_loader = await anext(web_loader)
    html_content = "<html><head><title>BS4 Success</title></head><body><p>Content via BS4.</p></body></html>"
    bs4_extracted_text = "Content via BS4."
    mock_response = httpx.Response(200, text=html_content, headers={"content-type": "text/html"}, request=dummy_request)

    mock_client_instance = AsyncMock(spec=httpx.AsyncClient)
    mock_client_instance.get = AsyncMock(return_value=mock_response)
    mock_client_instance.aclose = AsyncMock()

    mock_bs4_soup_instance = MagicMock()
    mock_bs4_soup_instance.get_text.return_value = bs4_extracted_text
    mock_bs4_soup_instance.title.string = "BS4 Success"
    mock_bs4_constructor = MagicMock(return_value=mock_bs4_soup_instance)

    with patch("httpx.AsyncClient", return_value=mock_client_instance), \
         patch("genie_tooling.document_loaders.impl.web_page.BeautifulSoup", mock_bs4_constructor):
        await actual_loader.setup(config={"use_trafilatura": False})
        docs = await collect_docs_from_loader(actual_loader, "http://example.com/bs4_success.html")

    assert len(docs) == 1, f"Docs found: {[d.id for d in docs]}"
    assert docs[0].content == bs4_extracted_text
    assert docs[0].metadata["title"] == "BS4 Success"

@pytest.mark.asyncio
async def test_load_non_html_content_type(web_loader: AsyncGenerator[WebPageLoader, None], dummy_request: httpx.Request, caplog: pytest.LogCaptureFixture):
    actual_loader = await anext(web_loader)
    caplog.set_level(logging.WARNING, logger=LOADER_LOGGER_NAME)
    json_content = '{"key": "value", "data": 123}'
    mock_response = httpx.Response(200, text=json_content, headers={"content-type": "application/json"}, request=dummy_request)

    mock_client_instance = AsyncMock(spec=httpx.AsyncClient)
    mock_client_instance.get = AsyncMock(return_value=mock_response)
    mock_client_instance.aclose = AsyncMock()

    with patch("httpx.AsyncClient", return_value=mock_client_instance):
        await actual_loader.setup()
        docs = await collect_docs_from_loader(actual_loader, "http://example.com/data.json")

    assert len(docs) == 1, f"Docs found: {[d.id for d in docs]}"
    assert docs[0].content == json_content
    assert docs[0].metadata["title"] == "example.com"
    assert any(f"{actual_loader.plugin_id}: Content type for http://example.com/data.json is 'application/json', not HTML. Using raw content." in record.message for record in caplog.records)

@pytest.mark.asyncio
async def test_load_http_status_error_404(web_loader: AsyncGenerator[WebPageLoader, None], dummy_request: httpx.Request, caplog: pytest.LogCaptureFixture):
    actual_loader = await anext(web_loader)
    caplog.set_level(logging.ERROR, logger=LOADER_LOGGER_NAME)
    mock_response = httpx.Response(404, text="Not Found", request=dummy_request)

    mock_client_instance = AsyncMock(spec=httpx.AsyncClient)
    mock_client_instance.get = AsyncMock(side_effect=httpx.HTTPStatusError(message="404 Not Found", request=dummy_request, response=mock_response))
    mock_client_instance.aclose = AsyncMock()

    with patch("httpx.AsyncClient", return_value=mock_client_instance):
        await actual_loader.setup()
        docs = await collect_docs_from_loader(actual_loader, "http://example.com/notfound.html")

    assert len(docs) == 0
    assert any(f"{actual_loader.plugin_id}: HTTP status error loading http://example.com/notfound.html. Status: 404." in record.message for record in caplog.records)

@pytest.mark.asyncio
async def test_load_http_request_error_network(web_loader: AsyncGenerator[WebPageLoader, None], dummy_request: httpx.Request, caplog: pytest.LogCaptureFixture):
    actual_loader = await anext(web_loader)
    caplog.set_level(logging.ERROR, logger=LOADER_LOGGER_NAME)

    mock_client_instance = AsyncMock(spec=httpx.AsyncClient)
    mock_client_instance.get = AsyncMock(side_effect=httpx.RequestError(message="Network error", request=dummy_request))
    mock_client_instance.aclose = AsyncMock()

    with patch("httpx.AsyncClient", return_value=mock_client_instance):
        await actual_loader.setup()
        docs = await collect_docs_from_loader(actual_loader, "http://example.com/network_error.html")

    assert len(docs) == 0
    assert any(f"{actual_loader.plugin_id}: Request error loading http://example.com/network_error.html. Error: Network error" in record.message for record in caplog.records)

@pytest.mark.asyncio
async def test_load_client_not_initialized(web_loader: AsyncGenerator[WebPageLoader, None], caplog: pytest.LogCaptureFixture):
    actual_loader = await anext(web_loader)
    actual_loader._http_client = None

    caplog.set_level(logging.ERROR, logger=LOADER_LOGGER_NAME)
    docs = await collect_docs_from_loader(actual_loader, "http://example.com/no_setup.html")
    assert len(docs) == 0
    assert any(f"{actual_loader.plugin_id}: HTTP client not initialized. Cannot load URL." in record.message for record in caplog.records)

@pytest.mark.asyncio
async def test_load_empty_html_page(web_loader: AsyncGenerator[WebPageLoader, None], dummy_request: httpx.Request):
    actual_loader = await anext(web_loader)
    html_content = "<html><head><title>Empty Page</title></head><body></body></html>"
    mock_response = httpx.Response(200, text=html_content, headers={"content-type": "text/html"}, request=dummy_request)

    mock_client_instance = AsyncMock(spec=httpx.AsyncClient)
    mock_client_instance.get = AsyncMock(return_value=mock_response)
    mock_client_instance.aclose = AsyncMock()

    mock_bs4_soup_instance = MagicMock()
    mock_bs4_soup_instance.get_text.return_value = ""
    mock_bs4_soup_instance.title.string = "Empty Page"
    mock_bs4_constructor = MagicMock(return_value=mock_bs4_soup_instance)

    with patch("httpx.AsyncClient", return_value=mock_client_instance), \
         patch("genie_tooling.document_loaders.impl.web_page.BeautifulSoup", mock_bs4_constructor):
        await actual_loader.setup(config={"use_trafilatura": False})
        docs = await collect_docs_from_loader(actual_loader, "http://example.com/empty.html")

    assert len(docs) == 1, f"Docs found: {[d.id for d in docs]}"
    assert docs[0].content == ""
    assert docs[0].metadata["title"] == "Empty Page"

@pytest.mark.asyncio
async def test_load_title_extraction_no_title_tag(web_loader: AsyncGenerator[WebPageLoader, None], dummy_request: httpx.Request):
    actual_loader = await anext(web_loader)
    html_content = "<html><head></head><body><p>Content here.</p></body></html>"
    mock_response = httpx.Response(200, text=html_content, headers={"content-type": "text/html"}, request=dummy_request)

    mock_client_instance = AsyncMock(spec=httpx.AsyncClient)
    mock_client_instance.get = AsyncMock(return_value=mock_response)
    mock_client_instance.aclose = AsyncMock()

    mock_bs4_soup_instance = MagicMock()
    mock_bs4_soup_instance.get_text.return_value = "Content here."
    mock_bs4_soup_instance.title = None
    mock_bs4_constructor = MagicMock(return_value=mock_bs4_soup_instance)

    with patch("httpx.AsyncClient", return_value=mock_client_instance), \
         patch("genie_tooling.document_loaders.impl.web_page.BeautifulSoup", mock_bs4_constructor):
        await actual_loader.setup(config={"use_trafilatura": False})
        docs = await collect_docs_from_loader(actual_loader, "http://example.com/no_title.html")

    assert len(docs) == 1, f"Docs found: {[d.id for d in docs]}"
    assert docs[0].metadata["title"] == "example.com"

@pytest.mark.asyncio
async def test_load_with_trafilatura_config_options(web_loader: AsyncGenerator[WebPageLoader, None], dummy_request: httpx.Request):
    actual_loader = await anext(web_loader)
    html_content = "<html><body><!-- A comment --><p>Main content.</p><table><tr><td>Table</td></tr></table></body></html>"
    trafilatura_extracted_text_with_comments_tables = "A comment\nMain content.\nTable"
    mock_response = httpx.Response(200, text=html_content, headers={"content-type": "text/html"}, request=dummy_request)

    mock_client_instance = AsyncMock(spec=httpx.AsyncClient)
    mock_client_instance.get = AsyncMock(return_value=mock_response)
    mock_client_instance.aclose = AsyncMock()

    mock_trafilatura_module = MagicMock()
    mock_trafilatura_module.extract = MagicMock(return_value=trafilatura_extracted_text_with_comments_tables)

    trafilatura_options_passed_to_load = {
        "trafilatura_include_comments": True,
        "trafilatura_include_tables": True,
        "trafilatura_no_fallback": True,
    }

    with patch("genie_tooling.document_loaders.impl.web_page.trafilatura", mock_trafilatura_module), \
         patch("genie_tooling.document_loaders.impl.web_page.BeautifulSoup", MagicMock()), \
         patch("httpx.AsyncClient", return_value=mock_client_instance):
        await actual_loader.setup(config={"use_trafilatura": True})
        docs = await collect_docs_from_loader(actual_loader, "http://example.com/traf_options.html", config=trafilatura_options_passed_to_load)

    assert len(docs) == 1, f"Docs found: {[d.id for d in docs]}"
    assert docs[0].content == trafilatura_extracted_text_with_comments_tables
    mock_trafilatura_module.extract.assert_called_once_with(
        html_content,
        include_comments=True,
        include_tables=True,
        no_fallback=True
    )
