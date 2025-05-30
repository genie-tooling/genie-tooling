import logging
from typing import Any, AsyncIterable, Dict, Optional, cast
from urllib.parse import urlparse

import httpx

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

# Attempt to import trafilatura, make it optional
try:
    import trafilatura  # type: ignore
except ImportError:
    trafilatura = None # type: ignore
    # No warning here, as its use is conditional

from genie_tooling.core.types import Document
from genie_tooling.document_loaders.abc import DocumentLoaderPlugin

logger = logging.getLogger(__name__)

class _ConcreteDocument:
    def __init__(self, content: str, metadata: Dict[str, Any], id: Optional[str] = None):
        self.content: str = content
        self.metadata: Dict[str, Any] = metadata
        self.id: Optional[str] = id

class WebPageLoader(DocumentLoaderPlugin):
    plugin_id: str = "web_page_loader_v1"
    description: str = "Loads and extracts textual content from a web page URL. Can use basic BS4 or advanced Trafilatura."

    _http_client: Optional[httpx.AsyncClient] = None
    _use_trafilatura: bool = False

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        timeout = float(cfg.get("timeout_seconds", 15.0))
        headers = cfg.get("headers", {"User-Agent": "GenieTooling/0.1 WebPageLoader"})
        self._http_client = httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True)

        self._use_trafilatura = bool(cfg.get("use_trafilatura", False))

        if self._use_trafilatura:
            if not trafilatura:
                logger.warning(
                    f"{self.plugin_id}: 'use_trafilatura' is True, but 'trafilatura' library "
                    "not installed. Falling back to BeautifulSoup or raw HTML. "
                    "Install with: poetry install --extras web_tools"
                )
                self._use_trafilatura = False # Fallback if import fails
            else:
                logger.info(f"{self.plugin_id}: Trafilatura will be used for content extraction.")

        if not self._use_trafilatura and not BeautifulSoup:
            logger.warning(f"{self.plugin_id}: Neither Trafilatura nor BeautifulSoup4 available. Text extraction will be raw HTML.")
        elif not BeautifulSoup and not self._use_trafilatura:
             logger.warning("WebPageLoader: 'beautifulsoup4' library not installed. Text extraction will be very basic (raw HTML).")


    async def load(self, source_uri: str, config: Optional[Dict[str, Any]] = None) -> AsyncIterable[Document]:
        if not self._http_client:
            logger.error(f"{self.plugin_id}: HTTP client not initialized. Cannot load URL.")
            if False: yield
            return

        cfg = config or {}
        trafilatura_config_options = {
            "include_comments": cfg.get("trafilatura_include_comments", False),
            "include_tables": cfg.get("trafilatura_include_tables", True),
            "no_fallback": cfg.get("trafilatura_no_fallback", False),
        }
        bs4_parser = cfg.get("bs4_parser", "html.parser")
        bs4_text_separator = cfg.get("bs4_text_separator", " ")

        page_title: str = urlparse(source_uri).netloc
        text_content: str = ""

        try:
            response = await self._http_client.get(source_uri)
            response.raise_for_status()
            html_content = response.text
            content_type = response.headers.get("content-type", "").lower()

            soup_for_title: Optional[Any] = None
            if "html" in content_type and BeautifulSoup:
                try:
                    soup_for_title = BeautifulSoup(html_content, "html.parser")
                    if soup_for_title.title and soup_for_title.title.string:
                        page_title = soup_for_title.title.string.strip()
                except Exception as e_title_parse:
                    logger.debug(f"Could not parse title with BS4 for {source_uri}: {e_title_parse}")

            if "html" not in content_type:
                logger.warning(f"{self.plugin_id}: Content type for {source_uri} is '{content_type}', not HTML. Using raw content.")
                text_content = html_content
            elif self._use_trafilatura and trafilatura: # Check trafilatura again in case it failed setup but was still True
                try:
                    extracted = trafilatura.extract(html_content, **trafilatura_config_options)
                    text_content = extracted if extracted else ""
                    if not text_content and not trafilatura_config_options["no_fallback"]:
                        if BeautifulSoup:
                            soup = BeautifulSoup(html_content, bs4_parser)
                            for el_type in ["script", "style", "nav", "footer", "aside"]:
                                for el in soup(el_type): el.decompose()
                            text_content = soup.get_text(separator=bs4_text_separator, strip=True)
                        else: text_content = html_content
                except Exception as e_traf:
                    logger.error(f"{self.plugin_id}: Error using Trafilatura for {source_uri}: {e_traf}. Falling back.", exc_info=True)
                    if BeautifulSoup:
                        soup = BeautifulSoup(html_content, bs4_parser)
                        for el_type in ["script", "style", "nav", "footer", "aside"]:
                            for el in soup(el_type): el.decompose()
                        text_content = soup.get_text(separator=bs4_text_separator, strip=True)
                    else: text_content = html_content
            elif BeautifulSoup:
                try:
                    soup = soup_for_title if soup_for_title else BeautifulSoup(html_content, bs4_parser)
                    for el_type in ["script", "style", "nav", "footer", "aside"]:
                        for el in soup(el_type): el.decompose()
                    text_content = soup.get_text(separator=bs4_text_separator, strip=True)
                except Exception as e_bs4:
                    logger.error(f"{self.plugin_id}: Error parsing HTML with BeautifulSoup for {source_uri}: {e_bs4}. Falling back to raw HTML.", exc_info=True)
                    text_content = html_content
            else:
                text_content = html_content

            doc_id = source_uri
            metadata = {
                "source_type": "web_page", "source_uri": source_uri, "url": source_uri,
                "title": page_title, "content_type": content_type,
                "status_code": response.status_code,
            }
            yield cast(Document, _ConcreteDocument(content=text_content.strip(), metadata=metadata, id=doc_id))

        except httpx.HTTPStatusError as e:
            logger.error(f"{self.plugin_id}: HTTP status error loading {source_uri}. Status: {e.response.status_code}.", exc_info=False)
        except httpx.RequestError as e:
            logger.error(f"{self.plugin_id}: Request error loading {source_uri}. Error: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"{self.plugin_id}: Unexpected error for {source_uri}: {e}", exc_info=True)

        if False: yield

    async def teardown(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        logger.debug(f"{self.plugin_id}: Teardown complete.")
