import logging
import re
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple

from genie_tooling.lookup.types import RankedToolResult
from genie_tooling.tool_lookup_providers.abc import ToolLookupProvider

logger = logging.getLogger(__name__)

class KeywordMatchLookupProvider(ToolLookupProvider):
    plugin_id: str = "keyword_match_lookup_v1"
    description: str = "Finds tools by matching keywords from the query against tool names, descriptions, and tags."

    _indexed_tools_data: ClassVar[Dict[str, Dict[str, Any]]] = {}

    async def index_tools(self, tools_data: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None) -> None:
        """Stores the provided formatted tool data for keyword matching."""
        self._indexed_tools_data = {item.get("identifier", ""): item for item in tools_data if item.get("identifier")}
        logger.info(f"{self.plugin_id}: Indexed {len(self._indexed_tools_data)} tools data for keyword matching.")

    async def add_tool(self, tool_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> bool:
        tool_id = tool_data.get("identifier")
        if not tool_id:
            return False
        self._indexed_tools_data[tool_id] = tool_data
        return True

    async def update_tool(self, tool_id: str, tool_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> bool:
        return await self.add_tool(tool_data, config)

    async def remove_tool(self, tool_id: str, config: Optional[Dict[str, Any]] = None) -> bool:
        if tool_id in self._indexed_tools_data:
            del self._indexed_tools_data[tool_id]
        return True

    def _extract_keywords_from_text(self, text: str) -> Set[str]:
        """Simple keyword extraction: lowercase, split by non-alphanum, remove short words."""
        if not text or not isinstance(text, str):
            return set()
        words = re.split(r"[\W_]+", text.lower())
        return {word for word in words if word and len(word) > 2}

    async def find_tools(
        self,
        natural_language_query: str,
        top_k: int = 5,
        config: Optional[Dict[str, Any]] = None
    ) -> List[RankedToolResult]:
        if not self._indexed_tools_data:
            logger.debug(f"{self.plugin_id}: No tools indexed. Returning empty results.")
            return []
        if not natural_language_query or not natural_language_query.strip():
            logger.debug(f"{self.plugin_id}: Empty query. Returning empty results.")
            return []

        query_keywords = self._extract_keywords_from_text(natural_language_query)
        if not query_keywords:
            logger.debug(f"{self.plugin_id}: No valid keywords extracted from query '{natural_language_query}'.")
            return []

        logger.debug(f"{self.plugin_id}: Query keywords: {query_keywords}")
        scored_tools: List[Tuple[float, Dict[str, Any], Set[str]]] = []

        for tool_identifier, tool_data_item in self._indexed_tools_data.items():
            current_score = 0.0
            field_weights = {
                "name": 3.0, "identifier": 2.5, "description_llm": 2.0,
                "description_human": 1.5, "tags": 2.0, "lookup_text_representation": 1.0,
            }
            searchable_text_parts: List[Tuple[str, float]] = []
            metadata_source = tool_data_item.get("_raw_metadata_snapshot", tool_data_item)

            name = metadata_source.get("name", tool_identifier if isinstance(tool_identifier, str) else "")
            if name and isinstance(name, str) and name.strip():
                searchable_text_parts.append((name, field_weights["name"]))

            if isinstance(tool_identifier, str) and tool_identifier.strip():
                searchable_text_parts.append((tool_identifier, field_weights["identifier"]))

            for field_name in ["description_llm", "description_human"]:
                desc_text = metadata_source.get(field_name)
                if desc_text and isinstance(desc_text, str) and desc_text.strip():
                    searchable_text_parts.append((desc_text, field_weights[field_name]))

            tags = metadata_source.get("tags")
            if tags and isinstance(tags, list):
                tags_str = " ".join(str(tag) for tag in tags if isinstance(tag, str) and str(tag).strip())
                if tags_str:
                    searchable_text_parts.append((tags_str, field_weights["tags"]))

            lookup_text = tool_data_item.get("lookup_text_representation")
            if lookup_text and isinstance(lookup_text, str) and lookup_text.strip():
                 searchable_text_parts.append((lookup_text, field_weights["lookup_text_representation"]))

            if not searchable_text_parts:
                logger.debug(f"{self.plugin_id}: Tool '{tool_identifier}' has no text in designated fields to search. Skipping.")
                continue

            matched_keywords_overall: Set[str] = set()
            tool_has_any_extractable_keywords = False
            for text_to_search, weight in searchable_text_parts:
                tool_field_keywords = self._extract_keywords_from_text(text_to_search)
                if tool_field_keywords:
                    tool_has_any_extractable_keywords = True
                common_keywords_in_field = query_keywords.intersection(tool_field_keywords)
                current_score += len(common_keywords_in_field) * weight
                matched_keywords_overall.update(common_keywords_in_field)

            if not tool_has_any_extractable_keywords:
                logger.debug(f"{self.plugin_id}: Tool '{tool_identifier}' yielded no usable keywords from its text fields. Skipping.")
                continue

            if current_score > 0:
                logger.debug(f"{self.plugin_id}: Tool '{tool_identifier}' score: {current_score}, matched keywords: {matched_keywords_overall}")
                scored_tools.append((current_score, tool_data_item, matched_keywords_overall.copy()))

        scored_tools.sort(key=lambda x: x[0], reverse=True)
        results: List[RankedToolResult] = []
        for score, tool_data, matched_keys in scored_tools[:top_k]:
            snippet = f"Matched keywords: {', '.join(sorted(matched_keys)[:3])}" if matched_keys else "Keyword match."
            if len(matched_keys) > 3:
                snippet += "..."
            results.append(
                RankedToolResult(
                    tool_identifier=tool_data.get("identifier", "unknown_tool_id_in_result"),
                    score=score,
                    matched_tool_data=tool_data,
                    description_snippet=snippet,
                    matched_keywords=sorted(matched_keys)
                )
            )
        logger.info(f"{self.plugin_id}: Found {len(results)} tools via keyword match for query. Top score: {results[0].score if results else 'N/A'}.")
        return results

    async def teardown(self) -> None:
        self._indexed_tools_data = {}
        logger.debug(f"{self.plugin_id}: Torn down (indexed data cleared).")
