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
DEFAULT_INDEXING_FORMATTER_ID = "compact_text_formatter_plugin_v1" # This should be a plugin_id

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
        self._default_indexing_formatter_id = default_indexing_formatter_id # This is a plugin_id

        self._indexed_tools_data_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._index_validity_map: Dict[str, bool] = {}
        logger.info(f"ToolLookupService initialized. Default provider: '{default_provider_id}', default formatter plugin for indexing: '{default_indexing_formatter_id}'.")

    async def _get_formatted_tool_data_for_provider(
        self, provider: ToolLookupProviderPlugin, formatter_plugin_id_override: Optional[str] = None
    ) -> Optional[List[Dict[str, Any]]]:
        formatter_id_to_use = formatter_plugin_id_override or self._default_indexing_formatter_id

        # Get formatter instance using its plugin_id
        indexing_formatter_any = await self._plugin_manager.get_plugin_instance(formatter_id_to_use)
        if not indexing_formatter_any or not isinstance(indexing_formatter_any, DefinitionFormatter):
            logger.error(f"ToolLookupService: Indexing formatter plugin '{formatter_id_to_use}' not found or invalid. Cannot prepare tool data.")
            return None

        indexing_formatter = cast(DefinitionFormatter, indexing_formatter_any)
        tools = await self._tool_manager.list_tools(enabled_only=True)
        if not tools:
            logger.debug("ToolLookupService: No tools available to format for indexing.")
            return []

        formatted_tool_data_list: List[Dict[str, Any]] = []
        for tool_instance in tools:
            try:
                raw_metadata = await tool_instance.get_metadata()
                formatted_data_item = indexing_formatter.format(tool_metadata=raw_metadata)
                # Ensure the formatted data is a dict and contains necessary fields
                if isinstance(formatted_data_item, dict):
                    if "identifier" not in formatted_data_item: formatted_data_item["identifier"] = tool_instance.identifier
                    if "lookup_text_representation" not in formatted_data_item:
                         desc_llm = raw_metadata.get("description_llm", "")
                         desc_human = raw_metadata.get("description_human", "")
                         name = raw_metadata.get("name", tool_instance.identifier)
                         formatted_data_item["lookup_text_representation"] = f"Tool: {name}. Description: {desc_llm or desc_human}"
                    # Store the raw metadata snapshot for potential use by the lookup provider or later stages
                    formatted_data_item["_raw_metadata_snapshot"] = {k:v for k,v in raw_metadata.items() if isinstance(v, (str,int,float,bool,list,dict,type(None)))}
                    formatted_tool_data_list.append(formatted_data_item)
                elif isinstance(formatted_data_item, str): # If formatter returns just a string
                    formatted_tool_data_list.append({
                        "identifier": tool_instance.identifier,
                        "lookup_text_representation": formatted_data_item,
                        "_raw_metadata_snapshot": {k:v for k,v in raw_metadata.items() if isinstance(v, (str,int,float,bool,list,dict,type(None)))}
                    })
                else:
                    logger.warning(f"Formatter '{formatter_id_to_use}' for tool '{tool_instance.identifier}' produced unexpected type: {type(formatted_data_item)}. Skipping.")
            except Exception as e_format:
                logger.error(f"Error formatting tool '{tool_instance.identifier}' for indexing with '{formatter_id_to_use}': {e_format}", exc_info=True)
        return formatted_tool_data_list

    async def reindex_tools_for_provider(self, provider_id: str, formatter_plugin_id_override: Optional[str] = None, provider_config: Optional[Dict[str,Any]] = None) -> bool:
        # This provider_config is for the ToolLookupProvider's setup method
        provider_setup_config = {"plugin_manager": self._plugin_manager, **(provider_config or {})}

        provider_instance_any = await self._plugin_manager.get_plugin_instance(provider_id, config=provider_setup_config)
        if not provider_instance_any or not isinstance(provider_instance_any, ToolLookupProviderPlugin):
            logger.error(f"ToolLookupService: Provider '{provider_id}' not found or invalid for reindexing.")
            self._index_validity_map[provider_id] = False
            return False
        provider_instance = cast(ToolLookupProviderPlugin, provider_instance_any)

        tools_data_for_indexing = await self._get_formatted_tool_data_for_provider(provider_instance, formatter_plugin_id_override)
        if tools_data_for_indexing is None:
            logger.error(f"Failed to prepare tool data for provider '{provider_id}' due to formatter error. Indexing aborted.")
            self._index_validity_map[provider_id] = False
            return False

        try:
            # Pass provider_setup_config to index_tools as well, as it might contain runtime settings for indexing
            await provider_instance.index_tools(tools_data=tools_data_for_indexing, config=provider_setup_config)
            self._indexed_tools_data_cache[provider_id] = tools_data_for_indexing
            self._index_validity_map[provider_id] = True
            logger.info(f"Successfully re-indexed tools for provider '{provider_id}'. Indexed {len(tools_data_for_indexing)} items.")
            return True
        except Exception as e_index:
            logger.error(f"Error during reindexing with provider '{provider_id}': {e_index}", exc_info=True)
            self._index_validity_map[provider_id] = False
            return False

    async def find_tools(
        self,
        natural_language_query: str,
        top_k: int = 5,
        provider_id_override: Optional[str] = None,
        indexing_formatter_id_override: Optional[str] = None, # This is a plugin_id
        provider_config_override: Optional[Dict[str, Any]] = None
    ) -> List[RankedToolResult]:
        if not natural_language_query or not natural_language_query.strip():
            logger.debug("ToolLookupService: Empty query provided. Returning no results.")
            return []

        target_provider_id = provider_id_override or self._default_provider_id
        # This config is for the provider's setup when get_plugin_instance is called
        current_provider_config = {"plugin_manager": self._plugin_manager, **(provider_config_override or {})}

        # Ensure the global config from MiddlewareConfig for this provider_id is also included
        # This part might be tricky if MiddlewareConfig isn't directly available here.
        # Assume for now that provider_config_override is comprehensive or that
        # ToolLookupService is configured with provider-specific configs if needed at this level.
        # The EmbeddingSimilarityLookupProvider.setup() will receive this current_provider_config.

        provider_instance_any = await self._plugin_manager.get_plugin_instance(target_provider_id, config=current_provider_config)
        if not provider_instance_any or not isinstance(provider_instance_any, ToolLookupProviderPlugin):
            logger.error(f"Tool lookup provider '{target_provider_id}' not found or invalid.")
            return []
        provider_instance = cast(ToolLookupProviderPlugin, provider_instance_any)

        if not self._index_validity_map.get(target_provider_id, False):
            logger.info(f"Index for provider '{target_provider_id}' not valid or first run. Triggering re-index.")
            # Pass the current_provider_config to reindex, as it contains settings for the provider's setup/operation
            reindex_successful = await self.reindex_tools_for_provider(
                target_provider_id,
                indexing_formatter_id_override, # This is the formatter_plugin_id
                provider_config=current_provider_config
            )
            if not reindex_successful:
                logger.error(f"Re-indexing failed for '{target_provider_id}'. Cannot perform lookup.")
                return []
        try:
            # Pass current_provider_config to find_tools as well, for runtime options
            return await provider_instance.find_tools(
                natural_language_query=natural_language_query,
                top_k=top_k,
                config=current_provider_config
            )
        except Exception as e_find:
            logger.error(f"Error finding tools with provider '{target_provider_id}': {e_find}", exc_info=True)
            return []

    def invalidate_index(self, provider_id: Optional[str] = None) -> None:
        # ... (remains the same)
        if provider_id:
            if provider_id in self._index_validity_map:
                self._index_validity_map[provider_id] = False
                logger.info(f"ToolLookupService: Index for provider '{provider_id}' invalidated.")
        else:
            for pid in list(self._index_validity_map.keys()):
                self._index_validity_map[pid] = False
            logger.info("ToolLookupService: All provider indices invalidated.")
