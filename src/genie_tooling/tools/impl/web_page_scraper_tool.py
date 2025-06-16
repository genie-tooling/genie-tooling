# src/genie_tooling/tools/impl/web_page_scraper_tool.py
"""
WebPageScraperTool: A robust tool for fetching and extracting text from a web page URL.
"""
import logging
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx

from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.abc import Tool

logger = logging.getLogger(__name__)

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    import trafilatura

    TRAFILATURA_AVAILABLE = True
except ImportError:
    trafilatura = None
    TRAFILATURA_AVAILABLE = False


class WebPageScraperTool(Tool):
    plugin_id: str = "web_page_scraper_tool_v1"
    identifier: str = "web_page_scraper_tool_v1"

    _http_client: Optional[httpx.AsyncClient] = None

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initializes the persistent httpx client."""
        cfg = config or {}
        timeout = float(cfg.get("timeout_seconds", 20.0))
        headers = cfg.get(
            "headers",
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            },
        )
        self._http_client = httpx.AsyncClient(
            timeout=timeout, follow_redirects=True, headers=headers
        )
        logger.info(f"{self.identifier}: HTTP client initialized.")

    async def get_metadata(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier,
            "name": "Web Page Scraper",
            "description_human": "Fetches the full text content from a given web URL. Includes retries for transient errors and uses advanced extraction if available.",
            "description_llm": "WebScraper: Fetches text from a web URL. Use this after a search tool provides a relevant URL. Args: url (str, req).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the web page to scrape.",
                    }
                },
                "required": ["url"],
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "title": {"type": ["string", "null"]},
                    "content": {"type": ["string", "null"]},
                    "error": {"type": ["string", "null"]},
                },
            },
            "key_requirements": [],
            "tags": ["web", "scrape", "content", "html"],
            "version": "1.1.0",
        }

    async def execute(
        self, params: Dict[str, Any], key_provider: KeyProvider, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        url = params.get("url")
        if not url or not isinstance(url, str):
            return {
                "url": url,
                "title": None,
                "content": None,
                "error": "URL must be a non-empty string.",
            }

        if not self._http_client:
            return {
                "url": url,
                "title": None,
                "content": None,
                "error": "HTTP client not initialized. Tool setup may have failed.",
            }

        normalized_url = url
        if not url.lower().startswith(("http://", "https://")):
            normalized_url = f"https://{url}"

        try:
            response = await self._http_client.get(normalized_url)
            response.raise_for_status()

            html_content = response.text
            page_title = urlparse(normalized_url).netloc
            text_content = ""

            # Try to get the title with BeautifulSoup if available, as it's often more reliable.
            if BeautifulSoup:
                try:
                    soup = BeautifulSoup(html_content, "html.parser")
                    if soup.title and soup.title.string:
                        page_title = soup.title.string.strip()
                except Exception as e_title:
                    logger.debug(f"Could not parse title for {url} with BeautifulSoup: {e_title}")

            # Try to get main content with Trafilatura if available.
            if TRAFILATURA_AVAILABLE and trafilatura:
                extracted = trafilatura.extract(
                    html_content, include_comments=False, include_tables=True
                )
                if extracted:
                    text_content = extracted

            # If Trafilatura failed or wasn't available, fall back to BeautifulSoup for content.
            if not text_content and BeautifulSoup:
                # Re-use soup object if it was already created for title extraction
                if 'soup' not in locals():
                    soup = BeautifulSoup(html_content, "html.parser")

                for el_type in ["script", "style", "nav", "footer", "aside", "header"]:
                    for el in soup(el_type):
                        el.decompose()
                text_content = soup.get_text(separator=" ", strip=True)

            # If both advanced methods fail, use the raw HTML as a last resort.
            if not text_content:
                text_content = html_content

            return {
                "url": normalized_url,
                "title": page_title,
                "content": text_content[:25000],  # Truncate to a reasonable max length
                "error": None,
            }
        except httpx.RequestError as e:
            return {
                "url": normalized_url,
                "title": None,
                "content": None,
                "error": f"Network request error: {e}",
            }
        except Exception as e:
            return {
                "url": normalized_url,
                "title": None,
                "content": None,
                "error": f"Unexpected error: {e}",
            }

    async def teardown(self) -> None:
        """Closes the persistent httpx client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
            logger.info(f"{self.identifier}: HTTP client closed.")