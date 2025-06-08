import logging
from typing import Any, Dict, List, Optional, cast

from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.definition_formatters.abc import DefinitionFormatter
from genie_tooling.tool_lookup_providers.abc import (
    ToolLookupProvider as ToolLookupProviderPlugin,
)
from genie_tooling.tools.manager import ToolManager

from .types import RankedToolResult

logger = logging.getLogger(__name__)

DEFAULT_LOOKUP_PROVIDER_ID = "embedding_similarity_lookup_v1"
DEFAULT_INDEXING_FORMATTER_ID = "compact_text_formatter_plugin_v1"

class ToolLookupService:
    def __init__(
        self,
        tool_manager: ToolManager,
        plugin_manager: PluginManager,
        default_provider_id: str = DEFAULT_LOOKUP_PROVIDER_ID,
        default_indexing_formatter_id: str = DEFAULT_INDEXING_FORMATTER_ID
    ):
        self._tool_manager = tool_manager
        self._plugin_manager = plugin_manager
        self._default_provider_id = default_provider_id
        self._default_indexing_formatter_id = default_indexing_formatter_id
        self._is_indexed_map: Dict[str, bool] = {}
        logger.info(f"ToolLookupService initialized. Default provider: '{default_provider_id}', default formatter plugin for indexing: '{default_indexing_formatter_id}'.")

    async def _get_formatted_tool_data(self, tool_id: str, formatter_id: str) -> Optional[Dict[str, Any]]:
        """Formats a single tool's metadata for indexing."""
        formatter_config = {"plugin_manager": self._plugin_manager}
        indexing_formatter_any = await self._plugin_manager.get_plugin_instance(formatter_id, config=formatter_config)
        if not indexing_formatter_any or not isinstance(indexing_formatter_any, DefinitionFormatter):
            logger.error(f"ToolLookupService: Indexing formatter plugin '{formatter_id}' not found or invalid.")
            return None

        indexing_formatter = cast(DefinitionFormatter, indexing_formatter_any)
        tool_instance = await self._tool_manager.get_tool(tool_id)
        if not tool_instance:
            logger.warning(f"Tool '{tool_id}' not found in ToolManager for formatting.")
            return None

        try:
            raw_metadata = await tool_instance.get_metadata()
            formatted_data = indexing_formatter.format(tool_metadata=raw_metadata)
            if isinstance(formatted_data, dict):
                formatted_data.setdefault("identifier", tool_instance.identifier)
                if "lookup_text_representation" not in formatted_data:
                    desc_llm = raw_metadata.get("description_llm", "")
                    desc_human = raw_metadata.get("description_human", "")
                    name = raw_metadata.get("name", tool_instance.identifier)
                    formatted_data["lookup_text_representation"] = f"Tool: {name}. Description: {desc_llm or desc_human}"
                formatted_data["_raw_metadata_snapshot"] = {k: v for k, v in raw_metadata.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
                return formatted_data
            elif isinstance(formatted_data, str):
                return {
                    "identifier": tool_instance.identifier,
                    "lookup_text_representation": formatted_data,
                    "_raw_metadata_snapshot": {k: v for k, v in raw_metadata.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
                }
            else:
                logger.warning(f"Formatter '{formatter_id}' for tool '{tool_id}' produced unexpected type: {type(formatted_data)}. Skipping.")
                return None
        except Exception as e_format:
            logger.error(f"Error formatting tool '{tool_id}' for indexing with '{formatter_id}': {e_format}", exc_info=True)
            return None

    async def _ensure_provider_is_indexed(self, provider_id: str, formatter_id: str, provider_config: Optional[Dict[str, Any]]):
        """Ensures the specified provider has been indexed at least once."""
        if not self._is_indexed_map.get(provider_id, False):
            logger.info(f"Index for provider '{provider_id}' not built or invalidated. Triggering full re-index.")
            await self.reindex_all_tools(provider_id, formatter_id, provider_config)

    async def reindex_all_tools(self, provider_id: str, formatter_id: Optional[str] = None, provider_config: Optional[Dict[str, Any]] = None) -> bool:
        """Performs a full re-indexing of all available tools for a given provider."""
        provider_setup_config = {"plugin_manager": self._plugin_manager, **(provider_config or {})}
        provider_instance = await self._plugin_manager.get_plugin_instance(provider_id, config=provider_setup_config)
        if not isinstance(provider_instance, ToolLookupProviderPlugin):
            logger.error(f"ToolLookupService: Provider '{provider_id}' not found or invalid for reindexing.")
            self._is_indexed_map[provider_id] = False
            return False

        formatter_id_to_use = formatter_id or self._default_indexing_formatter_id
        all_tools = await self._tool_manager.list_tools(enabled_only=True)
        tools_data_for_indexing = [
            data for tool in all_tools
            if (data := await self._get_formatted_tool_data(tool.identifier, formatter_id_to_use)) is not None
        ]

        if not tools_data_for_indexing:
            logger.warning(f"No tool data could be formatted for provider '{provider_id}'. Index will be empty.")

        try:
            await provider_instance.index_tools(tools_data=tools_data_for_indexing, config=provider_setup_config)
            self._is_indexed_map[provider_id] = True
            logger.info(f"Successfully re-indexed all {len(tools_data_for_indexing)} tools for provider '{provider_id}'.")
            return True
        except Exception as e_index:
            logger.error(f"Error during full re-indexing with provider '{provider_id}': {e_index}", exc_info=True)
            self._is_indexed_map[provider_id] = False
            return False

    async def add_or_update_tools(self, tool_ids: List[str], provider_id_override: Optional[str] = None, formatter_id_override: Optional[str] = None, provider_config_override: Optional[Dict[str, Any]] = None):
        """Incrementally adds or updates specific tools in the lookup index."""
        target_provider_id = provider_id_override or self._default_provider_id
        formatter_id_to_use = formatter_id_override or self._default_indexing_formatter_id
        await self._ensure_provider_is_indexed(target_provider_id, formatter_id_to_use, provider_config_override)

        provider_instance = await self._plugin_manager.get_plugin_instance(target_provider_id, config=provider_config_override)
        if not isinstance(provider_instance, ToolLookupProviderPlugin):
            logger.error(f"Cannot add/update tools: Provider '{target_provider_id}' not available.")
            return

        for tool_id in tool_ids:
            tool_data = await self._get_formatted_tool_data(tool_id, formatter_id_to_use)
            if tool_data:
                # FIX: Use keyword arguments for the call to make it less brittle and fix the test's KeyError.
                await provider_instance.update_tool(
                    tool_id=tool_id,
                    tool_data=tool_data,
                    config=provider_config_override
                )

    async def remove_tools(self, tool_ids: List[str], provider_id_override: Optional[str] = None, provider_config_override: Optional[Dict[str, Any]] = None):
        """Incrementally removes specific tools from the lookup index."""
        target_provider_id = provider_id_override or self._default_provider_id
        provider_instance = await self._plugin_manager.get_plugin_instance(target_provider_id, config=provider_config_override)
        if not isinstance(provider_instance, ToolLookupProviderPlugin):
            logger.error(f"Cannot remove tools: Provider '{target_provider_id}' not available.")
            return

        for tool_id in tool_ids:
            await provider_instance.remove_tool(tool_id, config=provider_config_override)

    async def find_tools(self, natural_language_query: str, top_k: int = 5, provider_id_override: Optional[str] = None, indexing_formatter_id_override: Optional[str] = None, provider_config_override: Optional[Dict[str, Any]] = None) -> List[RankedToolResult]:
        if not natural_language_query or not natural_language_query.strip():
            logger.debug("ToolLookupService: Empty query provided. Returning no results.")
            return []

        target_provider_id = provider_id_override or self._default_provider_id
        formatter_id_to_use = indexing_formatter_id_override or self._default_indexing_formatter_id

        await self._ensure_provider_is_indexed(target_provider_id, formatter_id_to_use, provider_config_override)

        provider_instance = await self._plugin_manager.get_plugin_instance(target_provider_id, config=provider_config_override)
        if not isinstance(provider_instance, ToolLookupProviderPlugin):
            logger.error(f"Tool lookup provider '{target_provider_id}' not found or invalid.")
            return []

        try:
            return await provider_instance.find_tools(natural_language_query=natural_language_query, top_k=top_k, config=provider_config_override)
        except Exception as e_find:
            logger.error(f"Error finding tools with provider '{target_provider_id}': {e_find}", exc_info=True)
            return []

    def invalidate_all_indices(self) -> None:
        """Marks all provider indices as stale, forcing a full re-index on next use."""
        logger.info("ToolLookupService: All provider indices invalidated.")
        self._is_indexed_map.clear()
