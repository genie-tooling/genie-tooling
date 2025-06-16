"""ArxivSearchTool: A tool for searching academic papers on ArXiv."""
import asyncio
import logging
from typing import Any, Dict, List, Union

from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.abc import Tool

logger = logging.getLogger(__name__)

try:
    import arxiv

    ARXIV_AVAILABLE = True
except ImportError:
    arxiv = None
    ARXIV_AVAILABLE = False
    logger.warning(
        "ArxivSearchTool: 'arxiv' library not installed. "
        "This tool will not be functional. Please install it: poetry add arxiv"
    )


class ArxivSearchTool(Tool):
    plugin_id: str = "arxiv_search_tool"
    identifier: str = "arxiv_search_tool"

    async def get_metadata(self) -> Dict[str, Any]:
        return {
            "identifier": self.identifier,
            "name": "ArXiv Search",
            "description_human": "Searches the ArXiv preprint server for academic papers on scientific and technical topics.",
            "description_llm": "ArxivSearch: Finds academic papers on ArXiv. Args: query (str, req), max_results (int, opt, default 3). Output includes a `url` key for each paper.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for ArXiv papers.",
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 3,
                        "description": "Maximum number of results to return.",
                    },
                },
                "required": ["query"],
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "entry_id": {"type": "string"},
                                "title": {"type": "string"},
                                "summary": {"type": "string"},
                                "authors": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "published_date": {"type": "string"},
                                "pdf_url": {"type": "string"},
                                "url": {
                                    "type": "string",
                                    "description": "The primary URL for the paper, typically the PDF link.",
                                },
                            },
                        },
                    },
                    "error": {"type": ["string", "null"]},
                },
                "required": ["results"],
            },
            "key_requirements": [],
            "tags": ["search", "research", "academic", "papers", "arxiv"],
            "version": "1.5.0",
            "cacheable": True,
            "cache_ttl_seconds": 3600 * 6,
        }

    async def execute(
        self, params: Dict[str, Any], key_provider: KeyProvider, context: Dict[str, Any]
    ) -> Dict[str, Union[List[Dict[str, Any]], str, None]]:
        if not ARXIV_AVAILABLE or not arxiv:
            return {"results": [], "error": "arxiv library not installed."}

        query = params["query"]
        max_results = params.get("max_results", 3)

        try:
            loop = asyncio.get_running_loop()

            search = arxiv.Search(
                query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
            )

            def sync_search():
                """Function to run in executor that consumes the generator."""
                results_list = []
                for result in search.results():

                    pdf_url = result.pdf_url
                    if pdf_url and pdf_url.startswith("http://"):
                        pdf_url = pdf_url.replace("http://", "https://", 1)
                    results_list.append(
                        {
                            "entry_id": result.entry_id,
                            "title": result.title,
                            "summary": result.summary,
                            "authors": [author.name for author in result.authors],
                            "published_date": result.published.isoformat(),
                            "pdf_url": pdf_url,
                            "url": pdf_url,  # Add the 'url' key for consistency
                        }
                    )
                return results_list

            results = await loop.run_in_executor(None, sync_search)
            return {"results": results, "error": None}

        except Exception as e:
            logger.error(
                "ArxivSearchTool: Error during ArXiv search for '%s': %s",
                query,
                e,
                exc_info=True,
            )
            return {
                "results": [],
                "error": f"An unexpected error occurred during ArXiv search: {e}",
            }
