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

        logger.debug(f"Initializing tools from {len(self._plugin_manager.list_discovered_plugin_classes())} discovered plugin classes.")
        for plugin_id, plugin_class in self._plugin_manager.list_discovered_plugin_classes().items():
#            if not issubclass(plugin_class, Tool):
#                logger.debug(f"Plugin '{plugin_id}' (class {plugin_class.__name__}) is not a Tool. Skipping tool initialization for it.")
#                continue
            init_kwargs = {}
            try: # Safely try to inspect __init__
                if inspect.isclass(plugin_class):
                    constructor_params = inspect.signature(plugin_class.__init__).parameters
                    if "plugin_manager" in constructor_params:
                        init_kwargs["plugin_manager"] = self._plugin_manager
            except (ValueError, TypeError):
                 logger.debug(f"Could not inspect __init__ for {plugin_class.__name__ if inspect.isclass(plugin_class) else plugin_class}. Assuming default constructor.")

            tool_setup_config = self._tool_initial_configs.get(plugin_id, {})

            try:
                instance_any = await self._plugin_manager.get_plugin_instance(plugin_id, config=tool_setup_config, **init_kwargs)

                if not instance_any or not isinstance(instance_any, Tool):
                    if instance_any:
                         logger.debug(f"Plugin '{plugin_id}' (class {plugin_class.__name__ if inspect.isclass(plugin_class) else plugin_class}) instantiated but is not a Tool.")
                    else:
                        logger.warning(f"Plugin '{plugin_id}' (class {plugin_class.__name__ if inspect.isclass(plugin_class) else plugin_class}) did not yield a valid instance or failed setup.")
                    continue

                instance = cast(Tool, instance_any)

                if instance.identifier in self._tools:
                    logger.warning(f"Duplicate tool identifier '{instance.identifier}' encountered from class {plugin_class.__name__ if inspect.isclass(plugin_class) else plugin_class}. "
                                   f"Source: '{self._plugin_manager.get_plugin_source(plugin_id)}'. Overwriting previous tool with same identifier.")
                self._tools[instance.identifier] = instance
                logger.debug(f"Initialized tool: '{instance.identifier}' from {plugin_class.__name__ if inspect.isclass(plugin_class) else plugin_class} (Source: {self._plugin_manager.get_plugin_source(plugin_id)})")
            except Exception as e:
                logger.error(f"Error initializing tool plugin '{plugin_id}' (class {plugin_class.__name__ if inspect.isclass(plugin_class) else plugin_class}): {e}", exc_info=True)

        logger.info(f"ToolManager initialized. Loaded {len(self._tools)} tools.")

    async def list_available_formatters(self) -> List[Dict[str, str]]:
        formatters_info: List[Dict[str, str]] = []
        if not self._plugin_manager._discovered_plugin_classes: # type: ignore
            await self._plugin_manager.discover_plugins()

        for plugin_id, _plugin_class_obj in self._plugin_manager.list_discovered_plugin_classes().items():
            # Try to get an instance to check its type, as issubclass can be tricky with protocols
            # This has a small overhead of instantiating plugins that might not be formatters
            try:
                # Pass a default empty config for setup, as we don't have specific formatter configs here
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
            logger.debug(f"Tool with identifier '{identifier}' not found in ToolManager.")
        return tool

    async def list_tools(self, enabled_only: bool = True) -> List[Tool]:
        return list(self._tools.values())

    async def list_tool_summaries(self, pagination_params: Optional[Dict[str, Any]] = None) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        all_tools = await self.list_tools()
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

            # Handle page_size: if 0 or invalid, use default, otherwise use provided value if >= 1
            page_size_input = pagination_params.get("page_size", default_page_size)
            try:
                page_size_val = int(page_size_input)
                if page_size_val >= 1:
                    page_size = page_size_val
                else: # Covers 0 and negative numbers by using default
                    page_size = default_page_size
                    logger.debug(f"Invalid page_size '{page_size_input}', using default {default_page_size}.")
            except (ValueError, TypeError): # Handle non-integer input
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
        if not tool: return None

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
