"""
PDFTextExtractorTool: A tool for extracting text content from any PDF URL.
"""
import io
import logging
from typing import Any, Dict, Optional, Union

import httpx

from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.abc import Tool

logger = logging.getLogger(__name__)

try:
    from pypdf import PdfReader

    PYPDF_AVAILABLE = True
except ImportError:
    PdfReader = None
    PYPDF_AVAILABLE = False
    logger.warning(
        "PDFTextExtractorTool: 'pypdf' library not installed. "
        "This tool will not be functional. Please install it: poetry add pypdf"
    )


class PDFTextExtractorTool(Tool):
    plugin_id: str = "pdf_text_extractor_tool_v1"
    identifier: str = "pdf_text_extractor_tool_v1"

    _http_client: Optional[httpx.AsyncClient] = None

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        timeout = float(cfg.get("timeout_seconds", 30.0))
        self._http_client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)

    async def get_metadata(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier,
            "name": "PDF Text Extractor",
            "description_human": "Downloads a PDF from a given URL and extracts its text content.",
            "description_llm": "PDFReader: Extracts all text from a PDF document given its public URL. Use this for a 'deep dive' into any document that is a PDF. Args: url (str, req).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The public URL of the PDF file to read.",
                    }
                },
                "required": ["url"],
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "text_content": {"type": ["string", "null"]},
                    "num_pages": {"type": ["integer", "null"]},
                    "error": {"type": ["string", "null"]},
                },
            },
            "key_requirements": [],
            "tags": ["pdf", "text_extraction", "deep_dive", "file_content"],
            "version": "1.0.0",
            "cacheable": True,
            "cache_ttl_seconds": 3600 * 24 * 7,  # Cache for a week
        }

    async def execute(
        self, params: Dict[str, Any], key_provider: KeyProvider, context: Dict[str, Any]
    ) -> Dict[str, Union[str, int, None]]:
        if not PYPDF_AVAILABLE or not self._http_client:
            return {
                "text_content": None,
                "num_pages": None,
                "error": "Tool dependency (pypdf or httpx) not available.",
            }

        pdf_url = params["url"]

        try:
            response = await self._http_client.get(pdf_url)
            response.raise_for_status()

            pdf_file = io.BytesIO(response.content)
            reader = PdfReader(pdf_file)
            num_pages = len(reader.pages)
            text_content = ""
            for page in reader.pages:
                text_content += page.extract_text() + "\n\n"

            return {
                "text_content": text_content.strip(),
                "num_pages": num_pages,
                "error": None,
            }

        except httpx.HTTPStatusError as e:
            err_msg = f"HTTP error {e.response.status_code} fetching PDF from {pdf_url}"
            logger.error(f"{self.identifier}: {err_msg}", exc_info=True)
            return {"text_content": None, "num_pages": None, "error": err_msg}
        except Exception as e:
            logger.error(
                f"{self.identifier}: Error parsing PDF from {pdf_url}: {e}",
                exc_info=True,
            )
            return {
                "text_content": None,
                "num_pages": None,
                "error": f"Failed to download or parse PDF: {e!s}",
            }

    async def teardown(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
        self._http_client = None
