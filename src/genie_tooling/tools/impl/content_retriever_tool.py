# src/genie_tooling/tools/impl/content_retriever_tool.py
"""
ContentRetrieverTool: A meta-tool that intelligently dispatches to the correct
content extractor (HTML or PDF) based on the URL's content type.
"""
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

import httpx

from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.abc import Tool

if TYPE_CHECKING:
    from genie_tooling.genie import Genie

logger = logging.getLogger(__name__)


class ContentRetrieverTool(Tool):
    plugin_id: str = "content_retriever_tool_v1"
    identifier: str = "content_retriever_tool_v1"

    # Define default sub-tool IDs at the class level
    _pdf_extractor_id: str = "pdf_text_extractor_tool_v1"
    _web_scraper_id: str = "web_page_scraper_tool_v1"

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the tool and configures its sub-tool dependencies.

        Args:
            config: Configuration dictionary that can override the default
                    sub-tool IDs. Expects 'pdf_extractor_id' and 'web_scraper_id'.
        """
        cfg = config or {}
        # Override defaults if provided in the configuration
        self._pdf_extractor_id = cfg.get("pdf_extractor_id", self._pdf_extractor_id)
        self._web_scraper_id = cfg.get("web_scraper_id", self._web_scraper_id)
        logger.info(
            f"{self.identifier}: Setup complete. "
            f"Using PDF extractor: '{self._pdf_extractor_id}', "
            f"Web scraper: '{self._web_scraper_id}'."
        )

    async def get_metadata(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier,
            "name": "Web and PDF Content Retriever",
            "description_human": "Retrieves the full text content from a given URL, automatically determining if it's a web page (HTML) or a PDF. This is the primary tool for performing a 'deep dive' on a search result.",
            "description_llm": "ContentRetriever: Fetches text from any URL (HTML web page or PDF). Use this tool to get the full content of a source found via search. Args: url (str, req).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the web page or PDF to retrieve content from.",
                    }
                },
                "required": ["url"],
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "content": {"type": ["string", "null"]},
                    "source_type": {
                        "type": "string",
                        "description": "The detected type of content, e.g., 'pdf' or 'html'.",
                    },
                    "url": {"type": "string"},
                    "error": {"type": ["string", "null"]},
                },
            },
            "key_requirements": [],
            "tags": ["web", "pdf", "retrieval", "content", "deep_dive", "meta_tool"],
            "version": "1.1.0",  # Version incremented for the change
            "cacheable": True,
            "cache_ttl_seconds": 3600,
        }

    async def execute(
        self, params: Dict[str, Any], key_provider: KeyProvider, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        url = params.get("url")
        if not url:
            return {"content": None, "source_type": "unknown", "url": url, "error": "URL parameter is missing."}

        genie_instance: Optional["Genie"] = context.get("genie_framework_instance")
        if not genie_instance:
            return {"content": None, "source_type": "unknown", "url": url, "error": "Genie framework instance not found in context."}

        source_type = "html"
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
                head_response = await client.head(url)
                head_response.raise_for_status()
                content_type = head_response.headers.get("content-type", "").lower()
                if "application/pdf" in content_type:
                    source_type = "pdf"
        except httpx.RequestError as e:
            logger.warning(f"HEAD request for {url} failed: {e}. Falling back to file extension check.")
            if url.lower().endswith(".pdf"):
                source_type = "pdf"
        except httpx.HTTPStatusError as e:
            logger.warning(f"HEAD request for {url} failed with status {e.response.status_code}. Proceeding with GET anyway.")
            if url.lower().endswith(".pdf"):
                source_type = "pdf"
        except Exception as e:
            logger.error(f"Unexpected error during HEAD request for {url}: {e}. Assuming HTML.", exc_info=True)

        try:
            if source_type == "pdf":
                logger.info(f"ContentRetriever detected PDF for {url}. Dispatching to '{self._pdf_extractor_id}'.")
                result = await genie_instance.execute_tool(self._pdf_extractor_id, url=url)
                if isinstance(result, dict):
                    return {
                        "content": result.get("text_content"),
                        "source_type": "pdf",
                        "url": url,
                        "error": result.get("error"),
                    }
                else:
                    return {"content": None, "source_type": "pdf", "url": url, "error": f"PDF extractor returned unexpected type: {type(result)}"}
            else:
                logger.info(f"ContentRetriever detected HTML/other for {url}. Dispatching to '{self._web_scraper_id}'.")
                result = await genie_instance.execute_tool(self._web_scraper_id, url=url)
                if isinstance(result, dict):
                    return {
                        "content": result.get("content"),
                        "source_type": "html",
                        "url": url,
                        "error": result.get("error"),
                    }
                else:
                    return {"content": None, "source_type": "html", "url": url, "error": f"Web page extractor returned unexpected type: {type(result)}"}
        except Exception as e_dispatch:
            logger.error(f"Error dispatching to sub-tool from ContentRetrieverTool: {e_dispatch}", exc_info=True)
            return {"content": None, "source_type": source_type, "url": url, "error": f"Failed to execute sub-tool: {e_dispatch!s}"}
