### tests/unit/tools/impl/test_arxiv_search_tool.py
import asyncio
import logging
import xml.etree.ElementTree as ET
from typing import Any, AsyncGenerator, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx as real_httpx_module # Import for creating real Response/Request objects
import pytest

from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.impl.arxiv_search_tool import ArxivSearchTool

TOOL_LOGGER_NAME = "genie_tooling.tools.impl.arxiv_search_tool"


class MockArxivKeyProvider(KeyProvider):
    async def get_key(self, key_name: str) -> Optional[str]:
        return None

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        pass

    async def teardown(self) -> None:
        pass


@pytest.fixture
async def arxiv_search_tool() -> AsyncGenerator[ArxivSearchTool, None]:
    tool = ArxivSearchTool()

    # Create an AsyncMock without spec for the client instance
    mock_client_instance_for_tool = AsyncMock()
    mock_client_instance_for_tool.get = AsyncMock()
    mock_client_instance_for_tool.aclose = AsyncMock()

    # Patch the AsyncClient where it's used by the ArxivSearchTool module.
    # When tool.setup() calls httpx.AsyncClient(), it will get our mock_client_instance_for_tool.
    with patch("genie_tooling.tools.impl.arxiv_search_tool.httpx.AsyncClient",
               return_value=mock_client_instance_for_tool):
        await tool.setup()
        yield tool
        await tool.teardown()


@pytest.fixture
def mock_key_provider_arxiv() -> MockArxivKeyProvider:
    return MockArxivKeyProvider()


@pytest.mark.asyncio
async def test_arxiv_get_metadata(arxiv_search_tool: AsyncGenerator[ArxivSearchTool, None]):
    tool = await anext(arxiv_search_tool)
    metadata = await tool.get_metadata()
    assert metadata["identifier"] == "arxiv_search_tool"
    assert metadata["name"] == "ArXiv Search"
    assert "query" in metadata["input_schema"]["properties"]
    assert "max_results" in metadata["input_schema"]["properties"]
    assert metadata["output_schema"]["properties"]["results"]["type"] == "array"
    assert not metadata["key_requirements"]


@pytest.mark.asyncio
async def test_arxiv_setup_and_teardown():
    tool = ArxivSearchTool()
    # Create an AsyncMock without spec for the client instance for this test
    mock_client_instance_for_this_test = AsyncMock()
    mock_client_instance_for_this_test.aclose = AsyncMock()

    with patch("genie_tooling.tools.impl.arxiv_search_tool.httpx.AsyncClient",
               return_value=mock_client_instance_for_this_test) as mock_async_client_constructor_in_test:
        await tool.setup()
        assert tool._http_client is mock_client_instance_for_this_test
        mock_async_client_constructor_in_test.assert_called_once()

    await tool.teardown()
    mock_client_instance_for_this_test.aclose.assert_awaited_once()
    assert tool._http_client is None


@pytest.mark.asyncio
async def test_arxiv_execute_success(
    arxiv_search_tool: AsyncGenerator[ArxivSearchTool, None],
    mock_key_provider_arxiv: MockArxivKeyProvider,
):
    tool = await anext(arxiv_search_tool)
    assert tool._http_client is not None, "HTTP client should be initialized by fixture"

    sample_xml_response = """
    <feed xmlns="http://www.w3.org/2005/Atom"
          xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
          xmlns:arxiv="http://arxiv.org/schemas/atom">
      <opensearch:totalResults>1</opensearch:totalResults>
      <entry>
        <id>http://arxiv.org/abs/2301.00001v1</id>
        <title>Test Paper Title</title>
        <summary>This is a test summary.</summary>
        <author><name>John Doe</name></author>
        <author><name>Jane Smith</name></author>
        <arxiv:primary_category term="cs.AI"/>
        <category term="cs.AI"/>
        <category term="cs.LG"/>
        <published>2023-01-01T00:00:00Z</published>
        <updated>2023-01-02T00:00:00Z</updated>
        <link title="pdf" href="http://arxiv.org/pdf/2301.00001v1" rel="related" type="application/pdf"/>
      </entry>
    </feed>
    """
    mock_response = real_httpx_module.Response(
        200, text=sample_xml_response, request=real_httpx_module.Request("GET", tool.ARXIV_API_BASE_URL)
    )
    tool._http_client.get = AsyncMock(return_value=mock_response)

    params = {"query": "test query", "max_results": 1}
    result = await tool.execute(params, mock_key_provider_arxiv)

    assert result["error"] is None
    assert len(result["results"]) == 1
    paper = result["results"][0]
    assert paper["title"] == "Test Paper Title"
    assert paper["arxiv_id"] == "2301.00001v1"
    assert "John Doe" in paper["authors"]
    assert "cs.AI" in paper["categories"]
    assert "cs.LG" in paper["categories"]
    assert result["total_results_available"] == 1
    tool._http_client.get.assert_awaited_once()


@pytest.mark.asyncio
async def test_arxiv_execute_http_status_error(
    arxiv_search_tool: AsyncGenerator[ArxivSearchTool, None],
    mock_key_provider_arxiv: MockArxivKeyProvider,
    caplog: pytest.LogCaptureFixture,
):
    tool = await anext(arxiv_search_tool)
    assert tool._http_client is not None
    caplog.set_level(logging.ERROR, logger=TOOL_LOGGER_NAME)

    mock_request = real_httpx_module.Request("GET", tool.ARXIV_API_BASE_URL)
    mock_response = real_httpx_module.Response(503, text="Service Unavailable", request=mock_request)
    tool._http_client.get = AsyncMock(
        side_effect=real_httpx_module.HTTPStatusError(
            message="Service Unavailable", request=mock_request, response=mock_response
        )
    )

    result = await tool.execute({"query": "error test"}, mock_key_provider_arxiv)
    assert "HTTP error 503: Service Unavailable" in result["error"]
    assert "HTTP error 503: Service Unavailable for URL" in caplog.text


@pytest.mark.asyncio
async def test_arxiv_execute_request_error(
    arxiv_search_tool: AsyncGenerator[ArxivSearchTool, None],
    mock_key_provider_arxiv: MockArxivKeyProvider,
    caplog: pytest.LogCaptureFixture,
):
    tool = await anext(arxiv_search_tool)
    assert tool._http_client is not None
    caplog.set_level(logging.ERROR, logger=TOOL_LOGGER_NAME)

    tool._http_client.get = AsyncMock(
        side_effect=real_httpx_module.RequestError("Network error", request=real_httpx_module.Request("GET", tool.ARXIV_API_BASE_URL))
    )
    result = await tool.execute({"query": "network error test"}, mock_key_provider_arxiv)
    assert "Request error connecting to ArXiv: Network error" in result["error"]
    assert "Request error connecting to ArXiv: Network error for URL" in caplog.text


@pytest.mark.asyncio
async def test_arxiv_execute_xml_parse_error(
    arxiv_search_tool: AsyncGenerator[ArxivSearchTool, None],
    mock_key_provider_arxiv: MockArxivKeyProvider,
    caplog: pytest.LogCaptureFixture,
):
    tool = await anext(arxiv_search_tool)
    assert tool._http_client is not None
    caplog.set_level(logging.ERROR, logger=TOOL_LOGGER_NAME)

    malformed_xml = "<feed><entry>malformed</entry>"
    mock_response = real_httpx_module.Response(
        200, text=malformed_xml, request=real_httpx_module.Request("GET", tool.ARXIV_API_BASE_URL)
    )
    tool._http_client.get = AsyncMock(return_value=mock_response)

    with patch("xml.etree.ElementTree.fromstring", side_effect=ET.ParseError("mocked XML parse error")):
        result = await tool.execute({"query": "xml error test"}, mock_key_provider_arxiv)

    assert "XML parsing error: mocked XML parse error" in result["error"]
    assert "XML parsing error: mocked XML parse error. Response text: <feed><entry>malformed</entry>" in caplog.text


@pytest.mark.asyncio
async def test_arxiv_execute_empty_response(
    arxiv_search_tool: AsyncGenerator[ArxivSearchTool, None],
    mock_key_provider_arxiv: MockArxivKeyProvider,
    caplog: pytest.LogCaptureFixture,
):
    tool = await anext(arxiv_search_tool)
    assert tool._http_client is not None
    caplog.set_level(logging.WARNING, logger=TOOL_LOGGER_NAME)

    mock_response = real_httpx_module.Response(
        200, text="", request=real_httpx_module.Request("GET", tool.ARXIV_API_BASE_URL)
    )
    tool._http_client.get = AsyncMock(return_value=mock_response)

    result = await tool.execute({"query": "empty response test"}, mock_key_provider_arxiv)
    assert result["error"] == "ArXiv API returned an empty response."
    assert "Received empty response from ArXiv for query 'empty response test'" in caplog.text


@pytest.mark.asyncio
async def test_arxiv_execute_tool_not_initialized(mock_key_provider_arxiv: MockArxivKeyProvider):
    tool_no_setup = ArxivSearchTool()
    result = await tool_no_setup.execute({"query": "test"}, mock_key_provider_arxiv)
    assert result["error"] == "Tool not initialized: HTTP client missing."


@pytest.mark.asyncio
async def test_arxiv_query_construction(
    arxiv_search_tool: AsyncGenerator[ArxivSearchTool, None],
    mock_key_provider_arxiv: MockArxivKeyProvider,
):
    tool = await anext(arxiv_search_tool)
    assert tool._http_client is not None
    tool._http_client.get = AsyncMock(return_value=real_httpx_module.Response(200, text="<feed></feed>", request=real_httpx_module.Request("GET", "")))

    await tool.execute({"query": "simple query"}, mock_key_provider_arxiv)
    call_args, _ = tool._http_client.get.call_args
    assert "search_query=all%3Asimple+query" in call_args[0]

    await tool.execute({"query": "au:Hinton"}, mock_key_provider_arxiv)
    call_args, _ = tool._http_client.get.call_args
    assert "search_query=au%3AHinton" in call_args[0]

    await tool.execute({"query": "test", "max_results": 10, "sort_by": "submittedDate", "sort_order": "ascending"}, mock_key_provider_arxiv)
    call_args, _ = tool._http_client.get.call_args
    assert "max_results=10" in call_args[0]
    assert "sortBy=submittedDate" in call_args[0]
    assert "sortOrder=ascending" in call_args[0]

@pytest.mark.asyncio
async def test_parse_arxiv_entry_missing_fields(arxiv_search_tool: AsyncGenerator[ArxivSearchTool, None]):
    tool = await anext(arxiv_search_tool)
    minimal_entry_xml = """
    <entry xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
      <id>http://arxiv.org/abs/0000.00000</id>
      <title>Minimal Paper</title>
      <summary>Minimal summary.</summary>
      <author><name>Anon</name></author>
      </entry>
    """
    entry_element = ET.fromstring(minimal_entry_xml)
    namespaces = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom"
    }
    parsed = tool._parse_arxiv_entry(entry_element, namespaces)

    assert parsed["arxiv_id"] == "0000.00000"
    assert parsed["title"] == "Minimal Paper"
    assert parsed["summary"] == "Minimal summary."
    assert parsed["authors"] == ["Anon"]
    assert parsed["published_date"] is None
    assert parsed["updated_date"] is None
    assert parsed["pdf_url"] is None
    assert parsed["categories"] == []

    empty_entry_xml = '<entry xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom"></entry>'
    empty_entry_element = ET.fromstring(empty_entry_xml)
    parsed_empty = tool._parse_arxiv_entry(empty_entry_element, namespaces)
    assert parsed_empty["arxiv_id"] == "unknown_id"
    assert parsed_empty["title"] == "N/A Title"
    assert parsed_empty["summary"] == "N/A Summary"
    assert parsed_empty["authors"] == []