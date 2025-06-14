# src/genie_tooling/tools/impl/community_google_search_tool.py
import asyncio
import functools
import logging
from typing import Any, Dict, Literal, Optional

from genie_tooling import tool
from genie_tooling.security.key_provider import KeyProvider

logger = logging.getLogger(__name__)

try:
    from googlesearch import search as googlesearch_search
    GOOGLESEARCH_LIB_AVAILABLE = True
except ImportError:
    googlesearch_search = None
    GOOGLESEARCH_LIB_AVAILABLE = False
    logger.warning(
        "CommunityGoogleSearchTool: 'googlesearch-python' library not installed. "
        "This tool will not be functional. Please install it (e.g., pip install googlesearch-python)."
    )

@tool
async def community_google_search(
    query: str,
    num_results: int = 10,
    lang: str = "en",
    region: Optional[str] = None,
    safe: Optional[Literal["active", "off"]] = "active",
    advanced: bool = False,
    unique: bool = False,
    sleep_interval: int = 0,
    context: Optional[Dict[str, Any]] = None,
    key_provider: Optional[KeyProvider] = None
) -> Dict[str, Any]:
    """
    Performs a Google search using the community 'googlesearch-python' library.
    This tool does not require API keys.

    Args:
        query (str): The search query.
        num_results (int): Number of results to return. Defaults to 10.
        lang (str): Language code for the search (e.g., 'en', 'fr'). Defaults to 'en'.
        region (Optional[str]): Country code for region-specific results (e.g., 'us', 'gb'). Defaults to None.
        safe (Optional[Literal["active", "off"]]): Safe search setting. 'active' (default) or 'off'.
        advanced (bool): If True, attempts to return more structured result objects (title, url, description). Defaults to False (returns list of URLs).
        unique (bool): If True, attempts to return unique links. Defaults to False.
        sleep_interval (int): Time in seconds to sleep between paged requests (if num_results > 100 or so). Defaults to 0.
        context (Optional[Dict[str, Any]]): Invocation context (unused by this tool directly).
        key_provider (Optional[KeyProvider]): Key provider (unused by this tool directly).

    Returns:
        Dict[str, Any]: A dictionary containing a list of search results or an error message.
                        If advanced=True, 'results' will be a list of dicts with 'title', 'url', 'description'.
                        If advanced=False, 'results' will be a list of URL strings.
    """
    if not GOOGLESEARCH_LIB_AVAILABLE or not googlesearch_search:
        logger.error("CommunityGoogleSearchTool: googlesearch library is not available.")
        return {"results": [], "error": "googlesearch library not installed or available."}

    logger.info(
        f"CommunityGoogleSearchTool: Performing search for '{query}' with num_results={num_results}, "
        f"lang='{lang}', region='{region}', safe='{safe}', advanced={advanced}, unique={unique}, sleep={sleep_interval}"
    )

    safe_search_param_for_lib = None if safe == "off" else "active"

    try:
        loop = asyncio.get_running_loop()

        search_function_with_kwargs = functools.partial(
            googlesearch_search,
            num_results=num_results,
            lang=lang,
            region=region,
            safe=safe_search_param_for_lib,
            advanced=advanced,
            sleep_interval=sleep_interval
        )

        search_results_iterable = await loop.run_in_executor(
            None,
            search_function_with_kwargs,
            query
        )

        results_list = []
        seen_urls = set()

        if advanced:
            for item in search_results_iterable:
                try:
                    url = getattr(item, "url", None)
                    if unique and url and url in seen_urls:
                        continue
                    if url:
                        seen_urls.add(url)

                    results_list.append({
                        "title": getattr(item, "title", "N/A"),
                        "url": url or "N/A",
                        "description": getattr(item, "description", "N/A")
                    })
                except Exception as e_item:
                    logger.warning(f"Could not parse advanced search item: {item}. Error: {e_item}")
                    results_list.append({"title": "Error parsing item", "url": str(item), "description": str(e_item)})
        else:
            for url_string in search_results_iterable:
                if isinstance(url_string, str):
                    if unique and url_string in seen_urls:
                        continue
                    seen_urls.add(url_string)
                    results_list.append(url_string)
                else:
                    logger.warning(f"Expected URL string from basic search, got {type(url_string)}: {url_string}")

        logger.info(f"CommunityGoogleSearchTool: Found {len(results_list)} results for '{query}'.")
        return {"results": results_list, "error": None}

    except Exception as e:
        logger.error(f"CommunityGoogleSearchTool: Error during search for '{query}': {e}", exc_info=True)
        return {"results": [], "error": f"An unexpected error occurred during search: {e!s}"}
