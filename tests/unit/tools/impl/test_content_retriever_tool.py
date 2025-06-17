### tests/unit/tools/impl/test_content_retriever_tool.py
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from genie_tooling.tools.impl.content_retriever_tool import ContentRetrieverTool


@pytest.fixture()
def mock_genie_for_retriever() -> MagicMock:
    """Provides a mock Genie facade instance."""
    genie = MagicMock(name="MockGenieForContentRetriever")
    genie.execute_tool = AsyncMock()
    return genie


@pytest.fixture()
def mock_context(mock_genie_for_retriever: MagicMock) -> Dict[str, Any]:
    """Provides a mock context dictionary with the Genie instance."""
    return {"genie_framework_instance": mock_genie_for_retriever}


@pytest.fixture()
async def content_retriever() -> ContentRetrieverTool:
    """Provides an initialized ContentRetrieverTool instance."""
    tool = ContentRetrieverTool()
    await tool.setup()
    return tool


@pytest.mark.asyncio()
class TestContentRetrieverTool:
    async def test_setup_configures_sub_tool_ids(self):
        tool = ContentRetrieverTool()
        config = {"pdf_extractor_id": "custom_pdf", "web_scraper_id": "custom_web"}
        await tool.setup(config)
        assert tool._pdf_extractor_id == "custom_pdf"
        assert tool._web_scraper_id == "custom_web"

    @patch("httpx.AsyncClient.head")
    async def test_dispatch_to_pdf_extractor_on_content_type(
        self,
        mock_head: AsyncMock,
        content_retriever: ContentRetrieverTool,
        mock_context: Dict[str, Any],
        mock_genie_for_retriever: MagicMock,
    ):
        tool = await content_retriever
        url = "http://example.com/document.pdf"

        dummy_request = httpx.Request("HEAD", url)
        mock_head.return_value = httpx.Response(
            200, headers={"content-type": "application/pdf"}, request=dummy_request
        )
        mock_genie_for_retriever.execute_tool.return_value = {
            "text_content": "PDF content",
            "error": None,
        }

        result = await tool.execute(
            {"url": url}, key_provider=AsyncMock(), context=mock_context
        )

        assert result["source_type"] == "pdf"
        assert result["content"] == "PDF content"
        mock_genie_for_retriever.execute_tool.assert_awaited_once_with(
            tool._pdf_extractor_id, url=url
        )

    @patch("httpx.AsyncClient.head")
    async def test_dispatch_to_web_scraper_on_html_type(
        self,
        mock_head: AsyncMock,
        content_retriever: ContentRetrieverTool,
        mock_context: Dict[str, Any],
        mock_genie_for_retriever: MagicMock,
    ):
        tool = await content_retriever
        url = "http://example.com/page.html"

        dummy_request = httpx.Request("HEAD", url)
        mock_head.return_value = httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            request=dummy_request,
        )
        mock_genie_for_retriever.execute_tool.return_value = {
            "content": "Web page content",
            "error": None,
        }

        result = await tool.execute(
            {"url": url}, key_provider=AsyncMock(), context=mock_context
        )

        assert result["source_type"] == "html"
        assert result["content"] == "Web page content"
        mock_genie_for_retriever.execute_tool.assert_awaited_once_with(
            tool._web_scraper_id, url=url
        )

    @patch("httpx.AsyncClient.head")
    async def test_fallback_on_head_request_error(
        self,
        mock_head: AsyncMock,
        content_retriever: ContentRetrieverTool,
        mock_context: Dict[str, Any],
        mock_genie_for_retriever: MagicMock,
    ):
        tool = await content_retriever
        url_pdf = "http://example.com/fallback.pdf"
        mock_head.side_effect = httpx.RequestError("Network error")
        mock_genie_for_retriever.execute_tool.return_value = {
            "text_content": "Fallback PDF content"
        }

        result = await tool.execute(
            {"url": url_pdf}, key_provider=AsyncMock(), context=mock_context
        )
        assert result["source_type"] == "pdf"
        assert result["content"] == "Fallback PDF content"
        mock_genie_for_retriever.execute_tool.assert_awaited_with(
            tool._pdf_extractor_id, url=url_pdf
        )

    async def test_missing_url_parameter(self, content_retriever: ContentRetrieverTool):
        tool = await content_retriever
        result = await tool.execute({}, key_provider=AsyncMock(), context={})
        assert "URL parameter is missing" in result["error"]

    async def test_missing_genie_instance_in_context(
        self, content_retriever: ContentRetrieverTool
    ):
        tool = await content_retriever
        result = await tool.execute(
            {"url": "http://a.com"}, key_provider=AsyncMock(), context={}
        )
        assert "Genie framework instance not found" in result["error"]
