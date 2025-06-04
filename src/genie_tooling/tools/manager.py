### src/genie_tooling/tools/manager.py
"""ToolManager for managing and accessing ToolPlugins."""
import inspect
import logging
from typing import Any, Dict, List, Optional, cast

from genie_tooling.core.plugin_manager import PluginManager

# Updated import path for DefinitionFormatter
from genie_tooling.definition_formatters.abc import DefinitionFormatter

from .abc import Tool

logger = logging.getLogger(__name__)

class ToolManager:
    """Manages the lifecycle and access to ToolPlugins."""
    def __init__(self, plugin_manager: PluginManager):
        self._plugin_manager = plugin_manager
        self._tools: Dict[str, Tool] = {} # identifier -> Tool instance
        self._tool_initial_configs: Dict[str, Dict[str, Any]] = {}
        logger.debug("ToolManager initialized.")

    async def initialize_tools(self, tool_configurations: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self._tool_initial_configs = tool_configurations or {}
        self._tools.clear()

        if not self._plugin_manager._discovered_plugin_classes: # type: ignore
            await self._plugin_manager.discover_plugins()

        logger.debug(f"Initializing tools based on tool_configurations. Number of configured tools: {len(self._tool_initial_configs)}")

        # Iterate only through tools explicitly mentioned in tool_configurations
        for plugin_id_or_alias, tool_specific_config in self._tool_initial_configs.items():
            # Resolve alias to canonical ID if necessary (PluginManager's get_plugin_instance might handle this,
            # or we can do it here if we have access to PLUGIN_ID_ALIASES)
            # For now, assume plugin_manager handles resolution or plugin_id_or_alias is canonical.
            # A more robust way would be to resolve alias here before passing to get_plugin_instance.
            # However, PluginManager.get_plugin_instance should ideally handle aliases if it's a common pattern.
            # Let's assume plugin_id_or_alias is the key used to fetch the class.
            # The class itself will have the canonical plugin_id and identifier.

            plugin_class = self._plugin_manager.list_discovered_plugin_classes().get(plugin_id_or_alias)

            if not plugin_class:
                logger.warning(f"Tool plugin class for ID/alias '{plugin_id_or_alias}' not found in discovered plugins. Skipping.")
                continue

            init_kwargs = {}
            try:
                if inspect.isclass(plugin_class):
                    constructor_params = inspect.signature(plugin_class.__init__).parameters
                    if "plugin_manager" in constructor_params:
                        init_kwargs["plugin_manager"] = self._plugin_manager
            except (ValueError, TypeError):
                 logger.debug(f"Could not inspect __init__ for {plugin_class.__name__ if inspect.isclass(plugin_class) else plugin_class}. Assuming default constructor.")

            try:
                # get_plugin_instance will instantiate and call setup
                instance_any = await self._plugin_manager.get_plugin_instance(
                    plugin_id_or_alias, # Use the key from tool_configurations
                    config=tool_specific_config or {}, # Pass the specific config for this tool
                    **init_kwargs
                )

                if not instance_any or not isinstance(instance_any, Tool):
                    if instance_any:
                         logger.debug(f"Plugin '{plugin_id_or_alias}' (class {plugin_class.__name__ if inspect.isclass(plugin_class) else plugin_class}) instantiated but is not a Tool.")
                    else:
                        logger.warning(f"Plugin '{plugin_id_or_alias}' (class {plugin_class.__name__ if inspect.isclass(plugin_class) else plugin_class}) did not yield a valid instance or failed setup.")
                    continue

                instance = cast(Tool, instance_any)

                if instance.identifier in self._tools:
                    logger.warning(f"Duplicate tool identifier '{instance.identifier}' encountered from plugin ID/alias '{plugin_id_or_alias}'. "
                                   f"Source: '{self._plugin_manager.get_plugin_source(plugin_id_or_alias)}'. Overwriting previous tool with same identifier.")
                self._tools[instance.identifier] = instance
                logger.debug(f"Initialized tool: '{instance.identifier}' from plugin ID/alias '{plugin_id_or_alias}' (Source: {self._plugin_manager.get_plugin_source(plugin_id_or_alias)})")
            except Exception as e:
                logger.error(f"Error initializing tool plugin from ID/alias '{plugin_id_or_alias}' (class {plugin_class.__name__ if inspect.isclass(plugin_class) else plugin_class}): {e}", exc_info=True)

        logger.info(f"ToolManager initialized. Loaded {len(self._tools)} explicitly configured tools.")

    async def list_available_formatters(self) -> List[Dict[str, str]]:
        formatters_info: List[Dict[str, str]] = []
        if not self._plugin_manager._discovered_plugin_classes: # type: ignore
            await self._plugin_manager.discover_plugins()

        for plugin_id, _plugin_class_obj in self._plugin_manager.list_discovered_plugin_classes().items():
            try:
                instance = await self._plugin_manager.get_plugin_instance(plugin_id, config={})
                if isinstance(instance, DefinitionFormatter):
                    formatter_instance = cast(DefinitionFormatter, instance)
                    formatters_info.append({
                        "id": formatter_instance.formatter_id,
                        "description": formatter_instance.description,
                        "plugin_id": plugin_id
                    })
            except Exception as e:
                logger.debug(f"Could not instantiate or check plugin '{plugin_id}' as DefinitionFormatter: {e}")
        return formatters_info

    async def get_tool(self, identifier: str) -> Optional[Tool]:
        tool = self._tools.get(identifier)
        if not tool:
            logger.debug(f"Tool with identifier '{identifier}' not found in ToolManager (not explicitly configured or loaded).")
        return tool

    async def list_tools(self, enabled_only: bool = True) -> List[Tool]:
        # 'enabled_only' flag is less relevant now as only explicitly configured tools are loaded.
        # However, keeping it for API consistency if future "disabled" states are introduced.
        return list(self._tools.values())

    async def list_tool_summaries(self, pagination_params: Optional[Dict[str, Any]] = None) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        all_tools = await self.list_tools() # This now returns only explicitly configured tools
        for tool_instance in all_tools:
            try:
                metadata = await tool_instance.get_metadata()
                summaries.append({
                    "identifier": tool_instance.identifier,
                    "name": metadata.get("name", tool_instance.identifier),
                    "short_description": metadata.get("description_llm", metadata.get("description_human", ""))[:120] + "...",
                    "tags": metadata.get("tags", [])
                })
            except Exception as e:
                logger.error(f"Error getting metadata for tool '{tool_instance.identifier}': {e}", exc_info=True)

        default_page_size = 20
        page = 1
        page_size = default_page_size

        if pagination_params:
            page = max(1, int(pagination_params.get("page", 1)))
            page_size_input = pagination_params.get("page_size", default_page_size)
            try:
                page_size_val = int(page_size_input)
                if page_size_val >= 1:
                    page_size = page_size_val
                else:
                    page_size = default_page_size
                    logger.debug(f"Invalid page_size '{page_size_input}', using default {default_page_size}.")
            except (ValueError, TypeError):
                page_size = default_page_size
                logger.debug(f"Non-integer page_size '{page_size_input}', using default {default_page_size}.")

        start_index = (page - 1) * page_size
        end_index = start_index + page_size
        paginated_summaries = summaries[start_index:end_index]
        total_items = len(summaries)
        total_pages = (total_items + page_size - 1) // page_size if total_items > 0 else 1

        pagination_meta = {
            "total_items": total_items, "current_page": page, "page_size": page_size,
            "total_pages": total_pages, "has_next": page < total_pages, "has_prev": page > 1
        }
        return paginated_summaries, pagination_meta

    async def get_formatted_tool_definition(self, tool_identifier: str, formatter_id: str) -> Optional[Any]:
        tool = await self.get_tool(tool_identifier)
        if not tool:
            return None # ToolManager now only holds explicitly configured tools.

        formatter_instance_any = await self._plugin_manager.get_plugin_instance(formatter_id)
        if not formatter_instance_any or not isinstance(formatter_instance_any, DefinitionFormatter):
            logger.warning(f"DefinitionFormatter plugin '{formatter_id}' not found or invalid.")
            return None
        formatter_instance = cast(DefinitionFormatter, formatter_instance_any)

        try:
            raw_metadata = await tool.get_metadata()
            return formatter_instance.format(tool_metadata=raw_metadata)
        except Exception as e:
            logger.error(f"Error formatting tool '{tool_identifier}' with formatter '{formatter_id}': {e}", exc_info=True)
            return None
