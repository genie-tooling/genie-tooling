import asyncio
import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlencode

import httpx

from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.abc import Tool

logger = logging.getLogger(__name__)

class ArxivSearchTool(Tool):
    identifier: str = "arxiv_search_tool"
    plugin_id: str = "arxiv_search_tool" # Matches identifier for consistency
    ARXIV_API_BASE_URL = "http://export.arxiv.org/api/query"

    _http_client: Optional[httpx.AsyncClient] = None

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initializes an async HTTP client for reuse."""
        if self._http_client: # Close existing client if setup is called again
            await self._http_client.aclose()
        self._http_client = httpx.AsyncClient(timeout=20.0) # Generous timeout for ArXiv API
        logger.info(f"{self.identifier}: HTTP client initialized for ArXiv API.")

    async def get_metadata(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier,
            "name": "ArXiv Search",
            "description_human": "Searches the ArXiv.org preprint server for academic papers based on a query. Retrieves title, authors, summary, dates, PDF link, and categories.",
            "description_llm": "ArXivSearch: Finds academic papers on ArXiv.org. Args: query (str, req, e.g., 'quantum machine learning'), max_results (int, opt, default 5, 1-50), sort_by (str, opt, enum['relevance', 'lastUpdatedDate', 'submittedDate'], default 'relevance'), sort_order (str, opt, enum['ascending', 'descending'], default 'descending').",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (e.g., keywords, author names like 'au:Hinton', title fragments like 'ti:transformer network'). Use ArXiv query syntax for advanced searches."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return.",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 50 # Practical limit to avoid overwhelming responses
                    },
                    "sort_by": {
                        "type": "string",
                        "description": "Field to sort results by. 'lastUpdatedDate' or 'submittedDate' for newest/oldest, 'relevance' for best match.",
                        "enum": ["relevance", "lastUpdatedDate", "submittedDate"],
                        "default": "relevance"
                    },
                    "sort_order": {
                        "type": "string",
                        "description": "Order of sorting results ('ascending' or 'descending').",
                        "enum": ["ascending", "descending"],
                        "default": "descending"
                    }
                },
                "required": ["query"]
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "arxiv_id": {"type": "string", "description": "Unique ArXiv ID (e.g., '2301.12345v1')."},
                                "title": {"type": "string", "description": "Title of the paper."},
                                "authors": {"type": "array", "items": {"type": "string"}, "description": "List of author names."},
                                "summary": {"type": "string", "description": "Abstract or summary of the paper."},
                                "published_date": {"type": ["string", "null"], "description": "Date the paper was first published on ArXiv (ISO 8601 format)."},
                                "updated_date": {"type": ["string", "null"], "description": "Date the paper was last updated on ArXiv (ISO 8601 format)."},
                                "pdf_url": {"type": ["string", "null"], "description": "Direct link to the PDF version of the paper."},
                                "categories": {"type": "array", "items": {"type": "string"}, "description": "List of ArXiv categories (e.g., 'cs.AI', 'quant-ph')."}
                            },
                            "required": ["arxiv_id", "title", "authors", "summary"]
                        }
                    },
                    "total_results_available": {"type": ["integer", "null"], "description": "Total number of results found by ArXiv for the query (may be more than returned due to max_results)."},
                    "error": {"type": ["string", "null"], "description": "Error message if the search failed, otherwise null."}
                },
                "required": ["results", "error"]
            },
            "key_requirements": [], # ArXiv API is public, no keys needed
            "tags": ["search", "academic", "research", "papers", "arxiv", "science", "literature"],
            "version": "1.0.1", # Increment version if changes are made
            "cacheable": True,
            "cache_ttl_seconds": 3600 * 6 # Cache ArXiv results for 6 hours
        }

    def _parse_arxiv_entry(self, entry: ET.Element, ns: Dict[str, str]) -> Dict[str, Any]:
        """Helper to parse a single <entry> element from ArXiv Atom feed."""
        paper: Dict[str, Any] = {}

        # ArXiv ID is typically the last part of the <id> URL
        id_url = entry.findtext("atom:id", default="", namespaces=ns)
        paper["arxiv_id"] = id_url.split("/")[-1] if id_url else "unknown_id"

        paper["title"] = entry.findtext("atom:title", default="N/A Title", namespaces=ns).strip().replace("\n", " ")
        paper["summary"] = entry.findtext("atom:summary", default="N/A Summary", namespaces=ns).strip().replace("\n", " ")
        paper["published_date"] = entry.findtext("atom:published", default=None, namespaces=ns)
        paper["updated_date"] = entry.findtext("atom:updated", default=None, namespaces=ns)

        authors = []
        for author_element in entry.findall("atom:author", namespaces=ns):
            name = author_element.findtext("atom:name", default="", namespaces=ns)
            if name:
                authors.append(name.strip())
        paper["authors"] = authors

        categories = []
        # Primary category
        primary_cat_el = entry.find("arxiv:primary_category", namespaces=ns)
        if primary_cat_el is not None:
            primary_term = primary_cat_el.get("term")
            if primary_term:
                categories.append(primary_term)
        # All categories (includes primary, so filter duplicates)
        for cat_element in entry.findall("atom:category", namespaces=ns):
            term = cat_element.get("term")
            if term and term not in categories:
                categories.append(term)
        paper["categories"] = categories

        pdf_link_element = entry.find("atom:link[@title='pdf']", namespaces=ns)
        paper["pdf_url"] = pdf_link_element.get("href") if pdf_link_element is not None else None

        return paper

    async def execute(
        self, params: Dict[str, Any], key_provider: KeyProvider, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Union[List[Dict[str, Any]], str, int, None]]:
        if not self._http_client:
            return {"results": [], "total_results_available": None, "error": "Tool not initialized: HTTP client missing."}

        query = params["query"]
        max_results = params.get("max_results", 5)
        sort_by = params.get("sort_by", "relevance")
        sort_order = params.get("sort_order", "descending")

        # Construct query parameters for ArXiv API
        # Note: ArXiv uses `search_query` for the actual query string.
        # It supports Lucene query syntax (e.g., ti:title, au:author, all:keywords).
        # If the user query doesn't use field prefixes, prefixing with 'all:' is a safe default.
        search_query_param = query
        if not any(prefix in query.lower() for prefix in ["ti:", "au:", "abs:", "cat:", "all:"]):
            search_query_param = f"all:{query}"

        query_params_dict = {
            "search_query": search_query_param,
            "max_results": str(max_results),
            "sortBy": sort_by,
            "sortOrder": sort_order,
            "start": "0" # For pagination, always start at 0 for this simple tool
        }

        encoded_query = urlencode(query_params_dict)
        request_url = f"{self.ARXIV_API_BASE_URL}?{encoded_query}"

        logger.debug(f"{self.identifier}: Requesting ArXiv URL: {request_url}")

        try:
            response = await self._http_client.get(request_url)
            response.raise_for_status() # Raise HTTPStatusError for 4xx/5xx responses

            xml_content = response.text
            if not xml_content:
                logger.warning(f"{self.identifier}: Received empty response from ArXiv for query '{query}'.")
                return {"results": [], "total_results_available": 0, "error": "ArXiv API returned an empty response."}

            # Define XML namespaces
            namespaces = {
                "atom": "http://www.w3.org/2005/Atom",
                "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
                "arxiv": "http://arxiv.org/schemas/atom"
            }

            # Parse XML content
            # Run synchronous XML parsing in a thread to avoid blocking the event loop
            loop = asyncio.get_running_loop()
            root = await loop.run_in_executor(None, ET.fromstring, xml_content)

            results_list: List[Dict[str, Any]] = []
            for entry_element in root.findall("atom:entry", namespaces=namespaces):
                results_list.append(self._parse_arxiv_entry(entry_element, namespaces))

            total_results_elem = root.find("opensearch:totalResults", namespaces=namespaces)
            total_results_available = int(total_results_elem.text) if total_results_elem is not None and total_results_elem.text and total_results_elem.text.isdigit() else None

            logger.info(f"{self.identifier}: Successfully fetched {len(results_list)} results from ArXiv for query '{query}'. Total available: {total_results_available or 'N/A'}.")
            return {"results": results_list, "total_results_available": total_results_available, "error": None}

        except httpx.HTTPStatusError as e:
            err_msg = f"HTTP error {e.response.status_code}: {e.response.text[:200]}" # Limit error text length
            logger.error(f"{self.identifier}: {err_msg} for URL {request_url}", exc_info=False) # exc_info=False for cleaner logs unless debugging
            return {"results": [], "total_results_available": None, "error": err_msg}
        except ET.ParseError as e_xml:
            err_msg = f"XML parsing error: {str(e_xml)}. Response text: {response.text[:200]}"
            logger.error(f"{self.identifier}: {err_msg} for URL {request_url}", exc_info=False)
            return {"results": [], "total_results_available": None, "error": err_msg}
        except httpx.RequestError as e_req: # Network errors
            err_msg = f"Request error connecting to ArXiv: {str(e_req)}"
            logger.error(f"{self.identifier}: {err_msg} for URL {request_url}", exc_info=True)
            return {"results": [], "total_results_available": None, "error": err_msg}
        except Exception as e_unexpected:
            err_msg = f"Unexpected error during ArXiv search: {str(e_unexpected)}"
            logger.error(f"{self.identifier}: {err_msg} for URL {request_url}", exc_info=True)
            return {"results": [], "total_results_available": None, "error": err_msg}

    async def teardown(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        logger.debug(f"{self.identifier}: Teardown complete.")
