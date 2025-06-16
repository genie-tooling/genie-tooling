# src/genie_tooling/tools/impl/web_page_scraper_tool.py
"""
WebPageScraperTool: A robust tool for fetching and extracting text from a web page URL.
"""
import logging
from typing import Any, Dict
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
            "version": "1.0.0",
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

        normalized_url = url
        if not url.lower().startswith(("http://", "https://")):
            normalized_url = f"https://{url}"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        }

        async with httpx.AsyncClient(
            timeout=20.0, follow_redirects=True, headers=headers
        ) as client:
            try:
                response = await client.get(normalized_url)
                response.raise_for_status()

                html_content = response.text
                page_title = urlparse(normalized_url).netloc
                if TRAFILATURA_AVAILABLE and trafilatura:
                    text_content = (
                        trafilatura.extract(
                            html_content, include_comments=False, include_tables=True
                        )
                        or ""
                    )
                    if BeautifulSoup:  # Still use BS4 for title if available
                        soup_title = BeautifulSoup(html_content, "html.parser")
                        if soup_title.title and soup_title.title.string:
                            page_title = soup_title.title.string.strip()
                elif BeautifulSoup:  # <<< FIX: Added check here
                    soup = BeautifulSoup(html_content, "html.parser")
                    if soup.title and soup.title.string:
                        page_title = soup.title.string.strip()
                    for el_type in [
                        "script",
                        "style",
                        "nav",
                        "footer",
                        "aside",
                        "header",
                    ]:
                        for el in soup(el_type):
                            el.decompose()
                    text_content = soup.get_text(separator=" ", strip=True)
                else:
                    text_content = html_content

                return {
                    "url": normalized_url,
                    "title": page_title,
                    "content": text_content[:25000],
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