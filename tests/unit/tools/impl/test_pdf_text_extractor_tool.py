### tests/unit/tools/impl/test_pdf_text_extractor_tool.py
import io
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from genie_tooling.tools.impl.pdf_text_extractor_tool import PDFTextExtractorTool


@pytest.fixture
def mock_pdf_reader() -> MagicMock:
    """Mocks the pypdf.PdfReader class."""
    mock_reader_instance = MagicMock()
    page1 = MagicMock()
    page1.extract_text.return_value = "This is page one."
    page2 = MagicMock()
    page2.extract_text.return_value = "This is page two."
    mock_reader_instance.pages = [page1, page2]
    return MagicMock(return_value=mock_reader_instance)


@pytest.fixture
async def pdf_tool() -> PDFTextExtractorTool:
    """Provides an initialized PDFTextExtractorTool."""
    tool = PDFTextExtractorTool()
    await tool.setup()
    return tool


@pytest.mark.asyncio
class TestPDFTextExtractorTool:
    @patch("httpx.AsyncClient.get")
    @patch("genie_tooling.tools.impl.pdf_text_extractor_tool.PdfReader")
    async def test_execute_success(
        self,
        mock_pdf_reader_cls: MagicMock,
        mock_get: AsyncMock,
        pdf_tool: PDFTextExtractorTool,
    ):
        tool = await pdf_tool

        # --- FIX: Configure the mock inside the test ---
        mock_reader_instance = MagicMock()
        page1 = MagicMock()
        page1.extract_text.return_value = "This is page one."
        page2 = MagicMock()
        page2.extract_text.return_value = "This is page two."
        mock_reader_instance.pages = [page1, page2]
        mock_pdf_reader_cls.return_value = mock_reader_instance
        # --- End of FIX ---

        mock_pdf_content = b"%PDF-1.4..."
        dummy_request = httpx.Request("GET", "http://example.com/doc.pdf")
        mock_get.return_value = httpx.Response(
            200, content=mock_pdf_content, request=dummy_request
        )

        result = await tool.execute(
            {"url": "http://example.com/doc.pdf"}, key_provider=AsyncMock(), context={}
        )

        assert result["error"] is None
        assert result["num_pages"] == 2
        assert "This is page one." in result["text_content"]
        assert "This is page two." in result["text_content"]

    @patch("httpx.AsyncClient.get")
    async def test_execute_http_error(
        self, mock_get: AsyncMock, pdf_tool: PDFTextExtractorTool
    ):
        tool = await pdf_tool
        dummy_request = httpx.Request("GET", "http://example.com/notfound.pdf")
        mock_get.return_value = httpx.Response(
            404, text="Not Found", request=dummy_request
        )
        mock_get.side_effect = httpx.HTTPStatusError(
            "Not Found", request=dummy_request, response=mock_get.return_value
        )
        result = await tool.execute(
            {"url": "http://example.com/notfound.pdf"},
            key_provider=AsyncMock(),
            context={},
        )
        assert "HTTP error 404" in result["error"]

    @patch("httpx.AsyncClient.get")
    @patch("genie_tooling.tools.impl.pdf_text_extractor_tool.PdfReader")
    async def test_execute_pdf_parse_error(
        self,
        mock_pdf_reader_cls: MagicMock,
        mock_get: AsyncMock,
        pdf_tool: PDFTextExtractorTool,
    ):
        tool = await pdf_tool
        mock_pdf_reader_cls.side_effect = Exception("Invalid PDF header")
        dummy_request = httpx.Request("GET", "http://example.com/bad.pdf")
        mock_get.return_value = httpx.Response(
            200, content=b"not a real pdf", request=dummy_request
        )

        result = await tool.execute(
            {"url": "http://example.com/bad.pdf"}, key_provider=AsyncMock(), context={}
        )
        assert "Failed to download or parse PDF" in result["error"]

    @patch("genie_tooling.tools.impl.pdf_text_extractor_tool.PYPDF_AVAILABLE", False)
    async def test_execute_pypdf_not_available(self, pdf_tool: PDFTextExtractorTool):
        tool = await pdf_tool
        result = await tool.execute(
            {"url": "http://a.com/a.pdf"}, key_provider=AsyncMock(), context={}
        )
        assert "pypdf" in result["error"]