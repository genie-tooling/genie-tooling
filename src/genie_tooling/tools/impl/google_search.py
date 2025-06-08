import logging
from typing import Any, Dict, List, Optional, Union

import httpx

from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.abc import Tool

logger = logging.getLogger(__name__)

class GoogleSearchTool(Tool):
    identifier: str = "google_search_tool_v1"
    plugin_id: str = "google_search_tool_v1"
    API_BASE_URL = "https://www.googleapis.com/customsearch/v1"
    API_KEY_NAME = "GOOGLE_API_KEY"
    CSE_ID_NAME = "GOOGLE_CSE_ID"

    _http_client: Optional[httpx.AsyncClient] = None

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._http_client = httpx.AsyncClient(timeout=10.0)
        logger.debug(f"{self.identifier}: HTTP client initialized.")

    async def get_metadata(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier,
            "name": "Google Search",
            "description_human": "Performs a web search using Google Custom Search API to find information online. Requires GOOGLE_API_KEY and GOOGLE_CSE_ID.",
            "description_llm": "GoogleSearch: Finds web pages. Args: query (str, req), num_results (int, opt, default 5, 1-10).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "num_results": {
                        "type": "integer", "description": "Number of results to return.",
                        "default": 5, "minimum": 1, "maximum": 10
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
                                "title": {"type": "string"}, "link": {"type": "string"}, "snippet": {"type": "string"}
                            }, "required": ["title", "link", "snippet"]
                        }
                    },
                    "error": {"type": ["string", "null"], "description": "Error message if search failed."}
                },
                "required": ["results", "error"]
            },
            "key_requirements": [
                {"name": self.API_KEY_NAME, "description": "Google API Key for Custom Search JSON API."},
                {"name": self.CSE_ID_NAME, "description": "Google Programmable Search Engine ID."}
            ],
            "tags": ["search", "web", "information_retrieval", "google", "internet"],
            "version": "1.0.0",
            "cacheable": True, "cache_ttl_seconds": 3600
        }

    async def execute(
        self, params: Dict[str, Any], key_provider: KeyProvider, context: Dict[str, Any]
    ) -> Dict[str, Union[List[Dict[str,str]], str, None]]:
        if not self._http_client:
            return {"results": [], "error": "Tool not initialized: HTTP client missing."}

        api_key = await key_provider.get_key(self.API_KEY_NAME)
        cse_id = await key_provider.get_key(self.CSE_ID_NAME)

        if not api_key:
            return {"results": [], "error": f"Missing API key: {self.API_KEY_NAME}"}
        if not cse_id:
            return {"results": [], "error": f"Missing CSE ID: {self.CSE_ID_NAME}"}

        query = params["query"]
        num_results = params.get("num_results", 5)

        query_params = {"key": api_key, "cx": cse_id, "q": query, "num": num_results}
        try:
            response = await self._http_client.get(self.API_BASE_URL, params=query_params)
            response.raise_for_status()
            data = response.json()

            if "items" not in data:
                return {"results": [], "error": "No search results found or API error."}

            results_list: List[Dict[str,str]] = []
            for item in data.get("items", []):
                results_list.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", "")
                })
            return {"results": results_list, "error": None}
        except httpx.HTTPStatusError as e:
            err_msg = f"HTTP error {e.response.status_code}: {e.response.text[:200]}"
            logger.error(f"{self.identifier}: {err_msg}", exc_info=True)
            return {"results": [], "error": err_msg}
        except Exception as e:
            logger.error(f"{self.identifier}: Unexpected error: {e}", exc_info=True)
            return {"results": [], "error": f"Unexpected error: {str(e)}"}

    async def teardown(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        logger.debug(f"{self.identifier}: Teardown complete.")
