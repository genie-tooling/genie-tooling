### tests/unit/tools/impl/test_google_search_tool.py
"""Unit tests for the GoogleSearchTool."""
from typing import Any, AsyncGenerator, Dict, Optional
from unittest.mock import AsyncMock, patch

import httpx as real_httpx_module # Import for creating real Response/Request objects
import pytest

from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.impl.google_search import GoogleSearchTool


class MockGSKeyProvider(KeyProvider):
    plugin_id = "mock_gs_key_provider_for_google_tests"
    _keys: Dict[str, Optional[str]]

    def __init__(self, api_key: Optional[str] = "fake_google_api_key", cse_id: Optional[str] = "fake_cse_id"):
        self._keys = {
            GoogleSearchTool.API_KEY_NAME: api_key,
            GoogleSearchTool.CSE_ID_NAME: cse_id
        }
    async def get_key(self, key_name: str) -> Optional[str]:
        return self._keys.get(key_name)
    async def setup(self,config: Optional[Dict[str, Any]]=None): pass
    async def teardown(self): pass

@pytest.fixture
async def google_search_tool() -> AsyncGenerator[GoogleSearchTool, None]:
    tool = GoogleSearchTool()
    # Create an AsyncMock without spec for the client instance
    mock_client_instance_for_tool = AsyncMock()
    mock_client_instance_for_tool.get = AsyncMock()
    mock_client_instance_for_tool.aclose = AsyncMock()

    with patch("genie_tooling.tools.impl.google_search.httpx.AsyncClient",
               return_value=mock_client_instance_for_tool):
        await tool.setup() # Initializes tool._http_client with our mock
        yield tool
        await tool.teardown()

@pytest.fixture
def mock_gs_key_provider_valid() -> MockGSKeyProvider:
    return MockGSKeyProvider()

@pytest.fixture
def mock_httpx_client_for_gs() -> AsyncMock: # This fixture might not be strictly needed if tool uses its own
    mock_client = AsyncMock() # No spec
    mock_client.get = AsyncMock()
    return mock_client

@pytest.mark.asyncio
async def test_gs_get_metadata(google_search_tool: AsyncGenerator[GoogleSearchTool, None]):
    tool = await anext(google_search_tool)
    metadata = await tool.get_metadata()
    assert metadata["identifier"] == "google_search_tool_v1"
    assert metadata["name"] == "Google Search"
    assert "query" in metadata["input_schema"]["properties"]
    assert "num_results" in metadata["input_schema"]["properties"]
    assert metadata["input_schema"]["properties"]["num_results"]["default"] == 5
    assert metadata["output_schema"]["properties"]["results"]["type"] == "array"
    assert {"name": GoogleSearchTool.API_KEY_NAME, "description": "Google API Key for Custom Search JSON API."} in metadata["key_requirements"]
    assert {"name": GoogleSearchTool.CSE_ID_NAME, "description": "Google Programmable Search Engine ID."} in metadata["key_requirements"]
    assert metadata["cacheable"] is True
    assert metadata["cache_ttl_seconds"] == 3600

@pytest.mark.asyncio
async def test_gs_setup_teardown(mocker):
    # Create an AsyncMock without spec for the client instance for this test
    mock_client_instance_for_this_test = AsyncMock()
    mock_client_instance_for_this_test.aclose = AsyncMock()

    mock_async_client_constructor = mocker.patch(
        "genie_tooling.tools.impl.google_search.httpx.AsyncClient",
        return_value=mock_client_instance_for_this_test
    )
    tool = GoogleSearchTool()
    await tool.setup()
    assert tool._http_client is mock_client_instance_for_this_test
    mock_async_client_constructor.assert_called_once()

    close_mock = tool._http_client.aclose
    await tool.teardown()
    close_mock.assert_awaited_once()
    assert tool._http_client is None

@pytest.mark.asyncio
async def test_gs_execute_success(
    google_search_tool: AsyncGenerator[GoogleSearchTool, None],
    mock_gs_key_provider_valid: MockGSKeyProvider
):
    tool = await anext(google_search_tool)
    # tool._http_client is already mocked by the google_search_tool fixture

    sample_response_data = {
        "items": [
            {"title": "Result 1", "link": "http://example.com/1", "snippet": "Snippet for result 1."},
            {"title": "Result 2", "link": "http://example.com/2", "snippet": "Snippet for result 2."}
        ]
    }
    # Use real_httpx_module to create Response and Request for type consistency if needed by underlying httpx
    mock_request = real_httpx_module.Request("GET", tool.API_BASE_URL)
    tool._http_client.get.return_value = real_httpx_module.Response(
        200, json=sample_response_data, request=mock_request
    )

    params = {"query": "test query", "num_results": 2}
    result = await tool.execute(params, mock_gs_key_provider_valid)

    assert result["error"] is None
    assert len(result["results"]) == 2
    assert result["results"][0]["title"] == "Result 1"
    tool._http_client.get.assert_awaited_once()
    called_args, called_kwargs = tool._http_client.get.call_args
    assert called_kwargs["params"]["q"] == "test query"
    assert called_kwargs["params"]["num"] == 2

@pytest.mark.asyncio
async def test_gs_execute_missing_api_key(google_search_tool: AsyncGenerator[GoogleSearchTool, None]):
    tool = await anext(google_search_tool)
    kp_no_api_key = MockGSKeyProvider(api_key=None)
    result = await tool.execute({"query": "test"}, kp_no_api_key)
    assert result["error"] == f"Missing API key: {GoogleSearchTool.API_KEY_NAME}"
    assert result["results"] == []

@pytest.mark.asyncio
async def test_gs_execute_missing_cse_id(google_search_tool: AsyncGenerator[GoogleSearchTool, None]):
    tool = await anext(google_search_tool)
    kp_no_cse_id = MockGSKeyProvider(cse_id=None)
    result = await tool.execute({"query": "test"}, kp_no_cse_id)
    assert result["error"] == f"Missing CSE ID: {GoogleSearchTool.CSE_ID_NAME}"
    assert result["results"] == []

@pytest.mark.asyncio
async def test_gs_execute_http_error(
    google_search_tool: AsyncGenerator[GoogleSearchTool, None],
    mock_gs_key_provider_valid: MockGSKeyProvider
):
    tool = await anext(google_search_tool)
    # tool._http_client is mocked by the fixture

    mock_request = real_httpx_module.Request("GET", tool.API_BASE_URL)
    mock_response = real_httpx_module.Response(401, text="Unauthorized API key", request=mock_request)
    tool._http_client.get.side_effect = real_httpx_module.HTTPStatusError(
        message="Client error '401 Unauthorized'", request=mock_request, response=mock_response
    )
    result = await tool.execute({"query": "test"}, mock_gs_key_provider_valid)
    assert "HTTP error 401: Unauthorized API key" in result["error"]

@pytest.mark.asyncio
async def test_gs_execute_request_error(
    google_search_tool: AsyncGenerator[GoogleSearchTool, None],
    mock_gs_key_provider_valid: MockGSKeyProvider
):
    tool = await anext(google_search_tool)
    # tool._http_client is mocked by the fixture
    tool._http_client.get.side_effect = real_httpx_module.RequestError(
        "Network connection failed", request=real_httpx_module.Request("GET", tool.API_BASE_URL)
    )

    result = await tool.execute({"query": "test"}, mock_gs_key_provider_valid)
    assert "Unexpected error: Network connection failed" in result["error"]

@pytest.mark.asyncio
async def test_gs_execute_no_items_in_response(
    google_search_tool: AsyncGenerator[GoogleSearchTool, None],
    mock_gs_key_provider_valid: MockGSKeyProvider
):
    tool = await anext(google_search_tool)
    # tool._http_client is mocked by the fixture
    tool._http_client.get.return_value = real_httpx_module.Response(
        200, json={"kind": "search"}, request=real_httpx_module.Request("GET", tool.API_BASE_URL)
    ) # No "items" key

    result = await tool.execute({"query": "test"}, mock_gs_key_provider_valid)
    assert result["results"] == []
    assert result["error"] == "No search results found or API error."

@pytest.mark.asyncio
async def test_gs_execute_tool_not_initialized(mock_gs_key_provider_valid: MockGSKeyProvider):
    tool_no_setup = GoogleSearchTool() # _http_client will be None
    result = await tool_no_setup.execute({"query": "test"}, mock_gs_key_provider_valid)
    assert result["error"] == "Tool not initialized: HTTP client missing."