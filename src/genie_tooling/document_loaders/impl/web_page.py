"""WebPageLoader: Loads and extracts text content from a web page URL."""
import logging
from typing import Any, AsyncIterable, Dict, Optional, cast
from urllib.parse import urlparse

import httpx  # Requires: poetry add httpx

# Attempt to import BeautifulSoup, make it optional for basic HTML extraction
try:
    from bs4 import BeautifulSoup  # type: ignore # Requires: poetry add beautifulsoup4
except ImportError:
    BeautifulSoup = None # type: ignore

from genie_tooling.core.types import Document

# Updated import path for DocumentLoaderPlugin
from genie_tooling.document_loaders.abc import DocumentLoaderPlugin

logger = logging.getLogger(__name__)

class _ConcreteDocument:
    def __init__(self, content: str, metadata: Dict[str, Any], id: Optional[str] = None):
        self.content: str = content
        self.metadata: Dict[str, Any] = metadata
        self.id: Optional[str] = id


class WebPageLoader(DocumentLoaderPlugin):
    plugin_id: str = "web_page_loader_v1"
    description: str = "Loads and extracts primary textual content from a given web page URL."

    _http_client: Optional[httpx.AsyncClient] = None

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initializes an async HTTP client for reuse."""
        cfg = config or {}
        timeout = float(cfg.get("timeout_seconds", 15.0))
        headers = cfg.get("headers", {"User-Agent": "MyAgenticMiddleware/0.1 WebPageLoader"})
        self._http_client = httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True)
        if not BeautifulSoup:
            logger.warning("WebPageLoader: 'beautifulsoup4' library not installed. Text extraction will be very basic (raw HTML). "
                           "Install with: poetry install --extras web_tools")


    async def load(self, source_uri: str, config: Optional[Dict[str, Any]] = None) -> AsyncIterable[Document]:
        """
        Loads content from the given URL.
        Config options:
            "parser": str (e.g., "html.parser", "lxml", default: "html.parser" if BS4 available)
            "text_separator": str (default: " ") separator for BeautifulSoup's get_text()
        """
        if not self._http_client:
            logger.error("WebPageLoader.load: HTTP client not initialized. Cannot load URL.")
            if False:
                yield # type: ignore # Make it an async generator
            return

        cfg = config or {}
        parser = cfg.get("parser", "html.parser")
        text_separator = cfg.get("text_separator", " ")

        soup_instance: Optional[Any] = None # Initialize soup_instance to None

        try:
            response = await self._http_client.get(source_uri)
            response.raise_for_status()

            html_content = response.text
            content_type = response.headers.get("content-type", "").lower()

            text_content: str

            if "html" not in content_type:
                logger.warning(f"WebPageLoader: Content type for {source_uri} is '{content_type}', not HTML. Using raw content.")
                text_content = html_content
                # soup_instance remains None
            elif BeautifulSoup:
                try:
                    soup_instance = BeautifulSoup(html_content, parser)
                    for script_or_style in soup_instance(["script", "style"]):
                        script_or_style.decompose()
                    text_content = soup_instance.get_text(separator=text_separator, strip=True)
                except Exception as e_bs4:
                    logger.error(f"WebPageLoader.load: Error parsing HTML with BeautifulSoup for {source_uri}: {e_bs4}. Falling back to raw HTML.", exc_info=True)
                    text_content = html_content
                    # soup_instance might be partially formed or None if constructor failed
            else:
                text_content = html_content
                # soup_instance remains None

            # Determine title safely
            page_title: str
            if soup_instance and hasattr(soup_instance, "title") and soup_instance.title and hasattr(soup_instance.title, "string") and soup_instance.title.string:
                page_title = soup_instance.title.string
            else:
                page_title = urlparse(source_uri).netloc


            doc_id = source_uri
            metadata = {
                "source_type": "web_page",
                "source_uri": source_uri,
                "url": source_uri,
                "title": page_title,
                "content_type": content_type,
                "status_code": response.status_code,
            }
            yield cast(Document, _ConcreteDocument(content=text_content, metadata=metadata, id=doc_id))

        except httpx.HTTPStatusError as e:
            logger.error(f"WebPageLoader.load: HTTP status error loading {source_uri}. Status: {e.response.status_code}. Response: {e.response.text[:200]}...", exc_info=False)
        except httpx.RequestError as e:
            logger.error(f"WebPageLoader.load: Request error loading {source_uri}. Error: {e}", exc_info=True)
        except Exception as e: # This will catch the UnboundLocalError if soup logic isn't fixed
            logger.error(f"WebPageLoader.load: Unexpected error loading or processing web page {source_uri}: {e}", exc_info=True)

        if False: # pylint: disable=false-condition
             yield # type: ignore


    async def teardown(self) -> None:
        """Closes the async HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
            logger.debug("WebPageLoader: HTTP client closed.")
