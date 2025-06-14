### src/genie_tooling/lookup/service.py
import logging
import uuid
from typing import Any, Dict, List, Optional, cast

from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.definition_formatters.abc import DefinitionFormatter
from genie_tooling.observability.manager import (
    InteractionTracingManager,
)
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
        default_provider_id: Optional[str] = DEFAULT_LOOKUP_PROVIDER_ID,
        default_indexing_formatter_id: Optional[str] = DEFAULT_INDEXING_FORMATTER_ID,
        tracing_manager: Optional[InteractionTracingManager] = None,
    ):
        self._tool_manager = tool_manager
        self._plugin_manager = plugin_manager
        self._default_provider_id = default_provider_id
        self._default_indexing_formatter_id = default_indexing_formatter_id
        self._tracing_manager = tracing_manager
        self._is_indexed_map: Dict[str, bool] = {}
        logger.info(f"ToolLookupService initialized. Default provider: '{default_provider_id}', default formatter plugin for indexing: '{default_indexing_formatter_id}'. Tracing enabled: {self._tracing_manager is not None}")

    async def _trace(self, event_name: str, data: Dict, level: str = "info", correlation_id: Optional[str] = None):
        """A helper to send trace events, simplifying calls from other methods."""
        if self._tracing_manager:
            event_data_with_msg = data.copy()
            if "message" not in event_data_with_msg:
                if "error" in data:
                    event_data_with_msg["message"] = str(data["error"])
                elif "status" in data:
                    event_data_with_msg["message"] = f"Status: {data['status']}"
                else:
                    event_data_with_msg["message"] = event_name.split(".")[-1].replace("_", " ").capitalize()

            final_event_name = event_name
            if not event_name.startswith("log.") and level in ["debug", "info", "warning", "error", "critical"]:
                 final_event_name = f"log.{level}"

            await self._tracing_manager.trace_event(
                final_event_name,
                event_data_with_msg,
                "ToolLookupService",
                correlation_id
            )
        else:
            log_level_to_use = level
            if event_name.startswith("log."):
                level_from_event = event_name.split(".")[1]
                if level_from_event in ["debug", "info", "warning", "error", "critical"]:
                    log_level_to_use = level_from_event
            log_func = getattr(logger, log_level_to_use, logger.info)
            log_message = data.get("message", str(data))
            log_func(f"ToolLookupService Event: {event_name} | {log_message}")


    async def _get_formatted_tool_data(self, tool_id: str, formatter_id: str, correlation_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        formatter_config = {"plugin_manager": self._plugin_manager}
        indexing_formatter_any = await self._plugin_manager.get_plugin_instance(formatter_id, config=formatter_config)
        if not indexing_formatter_any or not isinstance(indexing_formatter_any, DefinitionFormatter):
            await self._trace("log.error", {"message": f"Indexing formatter plugin '{formatter_id}' not found or invalid."}, level="error", correlation_id=correlation_id)
            return None

        indexing_formatter = cast(DefinitionFormatter, indexing_formatter_any)
        tool_instance = await self._tool_manager.get_tool(tool_id)
        if not tool_instance:
            await self._trace("log.warning", {"message": f"Tool '{tool_id}' not found in ToolManager for formatting."}, level="warning", correlation_id=correlation_id)
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
                await self._trace("log.warning", {"message": f"Formatter '{formatter_id}' for tool '{tool_id}' produced unexpected type: {type(formatted_data)}. Skipping."}, level="warning", correlation_id=correlation_id)
                return None
        except Exception as e_format:
            await self._trace("log.error", {"message": f"Error formatting tool '{tool_id}' for indexing with '{formatter_id}': {e_format}", "exc_info": True}, level="error", correlation_id=correlation_id)
            return None

    async def _ensure_provider_is_indexed(self, provider_id: str, formatter_id: str, provider_config: Optional[Dict[str, Any]], correlation_id: Optional[str] = None):
        if not self._is_indexed_map.get(provider_id, False):
            await self._trace("tool_lookup.index.needed", {"message": f"Index for provider '{provider_id}' not built or invalidated. Triggering full re-index."}, level="debug", correlation_id=correlation_id)
            await self.reindex_all_tools(provider_id, formatter_id, provider_config, correlation_id=correlation_id)
        else:
            await self._trace("tool_lookup.index.exists", {"message": f"Index for provider '{provider_id}' already exists and is considered valid."}, level="debug", correlation_id=correlation_id)

    async def add_or_update_tools(
        self,
        tool_ids: List[str],
        provider_id_override: Optional[str] = None,
        indexing_formatter_id_override: Optional[str] = None,
        provider_config_override: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> None:
        """
        Adds or updates a list of tools in the specified lookup provider's index.
        This is an incremental update.
        """
        corr_id = correlation_id or str(uuid.uuid4())
        await self._trace("tool_lookup.add_or_update.start", {"tool_ids": tool_ids}, "debug", corr_id)

        target_provider_id = provider_id_override or self._default_provider_id
        if not target_provider_id:
            await self._trace("log.warning", {"message": "No tool lookup provider specified for add/update. Skipping."}, "warning", corr_id)
            return

        formatter_id_to_use = indexing_formatter_id_override or self._default_indexing_formatter_id
        if not formatter_id_to_use:
            await self._trace("log.error", {"message": f"No indexing formatter ID available for provider '{target_provider_id}'. Cannot update index."}, "error", corr_id)
            return

        await self._ensure_provider_is_indexed(target_provider_id, formatter_id_to_use, provider_config_override, correlation_id=corr_id)

        provider_instance = await self._plugin_manager.get_plugin_instance(target_provider_id, config=provider_config_override)
        if not isinstance(provider_instance, ToolLookupProviderPlugin):
            await self._trace("log.error", {"message": f"Tool lookup provider '{target_provider_id}' not found or invalid for update."}, "error", corr_id)
            return

        for tool_id in tool_ids:
            tool_data = await self._get_formatted_tool_data(tool_id, formatter_id_to_use, correlation_id=corr_id)
            if tool_data:
                try:
                    await provider_instance.update_tool(tool_id, tool_data, config=provider_config_override)
                    await self._trace("tool_lookup.add_or_update.success", {"tool_id": tool_id}, "debug", corr_id)
                except Exception as e_update:
                    await self._trace("log.error", {"message": f"Error updating tool '{tool_id}' in provider '{target_provider_id}': {e_update}", "exc_info": True}, "error", corr_id)

    async def remove_tools(
        self,
        tool_ids: List[str],
        provider_id_override: Optional[str] = None,
        provider_config_override: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> None:
        """Removes a list of tools from the specified lookup provider's index."""
        corr_id = correlation_id or str(uuid.uuid4())
        await self._trace("tool_lookup.remove.start", {"tool_ids": tool_ids}, "debug", corr_id)

        target_provider_id = provider_id_override or self._default_provider_id
        if not target_provider_id:
            await self._trace("log.warning", {"message": "No tool lookup provider specified for remove. Skipping."}, "warning", corr_id)
            return

        if not self._is_indexed_map.get(target_provider_id, False):
            await self._trace("log.debug", {"message": f"Index for provider '{target_provider_id}' not built. Skipping tool removal as it's not indexed anyway."}, "debug", corr_id)
            return

        provider_instance = await self._plugin_manager.get_plugin_instance(target_provider_id, config=provider_config_override)
        if not isinstance(provider_instance, ToolLookupProviderPlugin):
            await self._trace("log.error", {"message": f"Tool lookup provider '{target_provider_id}' not found or invalid for removal."}, "error", corr_id)
            return

        for tool_id in tool_ids:
            try:
                await provider_instance.remove_tool(tool_id, config=provider_config_override)
                await self._trace("tool_lookup.remove.success", {"tool_id": tool_id}, "debug", corr_id)
            except Exception as e_remove:
                await self._trace("log.error", {"message": f"Error removing tool '{tool_id}' from provider '{target_provider_id}': {e_remove}", "exc_info": True}, "error", corr_id)

    async def reindex_all_tools(self, provider_id: str, formatter_id: Optional[str] = None, provider_config: Optional[Dict[str, Any]] = None, correlation_id: Optional[str] = None) -> bool:
        await self._trace("tool_lookup.reindex_all.start", {"provider_id": provider_id, "formatter_id": formatter_id or "default"}, correlation_id=correlation_id)
        provider_setup_config = {"plugin_manager": self._plugin_manager, **(provider_config or {})}
        provider_instance = await self._plugin_manager.get_plugin_instance(provider_id, config=provider_setup_config)
        if not isinstance(provider_instance, ToolLookupProviderPlugin):
            await self._trace("log.error", {"message": f"Provider '{provider_id}' not found or invalid for reindexing."}, level="error", correlation_id=correlation_id)
            self._is_indexed_map[provider_id] = False
            return False

        formatter_id_to_use = formatter_id or self._default_indexing_formatter_id
        if not formatter_id_to_use:
            await self._trace("log.error", {"message": f"No indexing formatter ID available for provider '{provider_id}'. Cannot index."}, level="error", correlation_id=correlation_id)
            return False

        all_tools = await self._tool_manager.list_tools(enabled_only=True)
        tools_data_for_indexing = [
            data for tool in all_tools
            if (data := await self._get_formatted_tool_data(tool.identifier, formatter_id_to_use, correlation_id=correlation_id)) is not None
        ]

        if not tools_data_for_indexing:
            await self._trace("log.warning", {"message": f"No tool data could be formatted for provider '{provider_id}'. Index will be empty."}, level="warning", correlation_id=correlation_id)

        try:
            await provider_instance.index_tools(tools_data=tools_data_for_indexing, config=provider_setup_config)
            self._is_indexed_map[provider_id] = True
            await self._trace("tool_lookup.reindex_all.success", {"message": f"Successfully re-indexed all {len(tools_data_for_indexing)} tools for provider '{provider_id}'."}, correlation_id=correlation_id)
            return True
        except Exception as e_index:
            await self._trace("log.error", {"message": f"Error during full re-indexing with provider '{provider_id}': {e_index}", "exc_info": True}, level="error", correlation_id=correlation_id)
            self._is_indexed_map[provider_id] = False
            return False

    async def find_tools(self, natural_language_query: str, top_k: int = 5, provider_id_override: Optional[str] = None, indexing_formatter_id_override: Optional[str] = None, provider_config_override: Optional[Dict[str, Any]] = None, correlation_id: Optional[str] = None) -> List[RankedToolResult]:
        corr_id = correlation_id or str(uuid.uuid4())
        await self._trace("tool_lookup.find_tools.start", {"query": natural_language_query, "top_k": top_k, "provider_override": provider_id_override}, correlation_id=corr_id)

        if not natural_language_query or not natural_language_query.strip():
            await self._trace("log.debug", {"message": "Empty query provided. Returning no results."}, level="debug", correlation_id=corr_id)
            return []

        target_provider_id = provider_id_override or self._default_provider_id
        if not target_provider_id:
            await self._trace("log.warning", {"message": "No tool lookup provider specified or configured. Cannot find tools."}, level="warning", correlation_id=corr_id)
            return []

        formatter_id_to_use = indexing_formatter_id_override or self._default_indexing_formatter_id
        if not formatter_id_to_use:
            await self._trace("log.error", {"message": f"No indexing formatter ID available for provider '{target_provider_id}'. Cannot ensure index."}, level="error", correlation_id=corr_id)
            return []

        await self._ensure_provider_is_indexed(target_provider_id, formatter_id_to_use, provider_config_override, correlation_id=corr_id)

        provider_instance = await self._plugin_manager.get_plugin_instance(target_provider_id, config=provider_config_override)
        if not isinstance(provider_instance, ToolLookupProviderPlugin):
            await self._trace("log.error", {"message": f"Tool lookup provider '{target_provider_id}' not found or invalid."}, level="error", correlation_id=corr_id)
            return []

        try:
            results = await provider_instance.find_tools(natural_language_query=natural_language_query, top_k=top_k, config=provider_config_override)
            await self._trace("tool_lookup.find_tools.success", {"num_results": len(results)}, correlation_id=corr_id)
            return results
        except Exception as e_find:
            await self._trace("log.error", {"message": f"Error finding tools with provider '{target_provider_id}': {e_find}", "exc_info": True}, level="error", correlation_id=corr_id)
            return []

    async def invalidate_all_indices(self, correlation_id: Optional[str] = None) -> None:
        await self._trace("tool_lookup.index.invalidate_all", {"message": "All provider indices invalidated."}, correlation_id=correlation_id)
        self._is_indexed_map.clear()

    async def teardown(self) -> None:
        await self._trace("tool_lookup.teardown", {"message": "ToolLookupService tearing down."}, correlation_id=None)
        self._is_indexed_map.clear()
        self._tracing_manager = None
