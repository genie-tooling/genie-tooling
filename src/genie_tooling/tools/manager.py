import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, cast

from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.definition_formatters.abc import DefinitionFormatter

from .abc import Tool

logger = logging.getLogger(__name__)

class ToolManager:
    def __init__(self, plugin_manager: PluginManager):
        self._plugin_manager = plugin_manager
        self._tools: Dict[str, Tool] = {}
        self._tool_initial_configs: Dict[str, Dict[str, Any]] = {}
        logger.debug("ToolManager initialized.")

    async def initialize_tools(self, tool_configurations: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self._tool_initial_configs = tool_configurations or {}
        self._tools.clear()

        if not self._plugin_manager._discovered_plugin_classes:
            await self._plugin_manager.discover_plugins()

        logger.debug(f"Initializing tools based on tool_configurations. Number of configured tools: {len(self._tool_initial_configs)}")

        for plugin_id_or_alias, tool_specific_config in self._tool_initial_configs.items():
            plugin_class = self._plugin_manager.list_discovered_plugin_classes().get(plugin_id_or_alias)

            if not plugin_class:
                logger.debug(f"Tool ID/alias '{plugin_id_or_alias}' not found as a discovered plugin class. It may be a function-based tool to be registered later.")
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
                instance_any = await self._plugin_manager.get_plugin_instance(
                    plugin_id_or_alias, config=tool_specific_config or {}, **init_kwargs
                )

                if not instance_any or not isinstance(instance_any, Tool):
                    if instance_any:
                         logger.debug(f"Plugin '{plugin_id_or_alias}' (class {plugin_class.__name__ if inspect.isclass(plugin_class) else plugin_class}) instantiated but is not a Tool.")
                    else:
                        logger.warning(f"Plugin '{plugin_id_or_alias}' (class {plugin_class.__name__ if inspect.isclass(plugin_class) else plugin_class}) did not yield a valid instance or failed setup.")
                    continue

                instance = cast(Tool, instance_any)

                if instance.identifier in self._tools:
                    logger.warning(f"Duplicate tool identifier '{instance.identifier}' encountered from plugin ID/alias '{plugin_id_or_alias}'. Source: '{self._plugin_manager.get_plugin_source(plugin_id_or_alias)}'. Overwriting previous tool with same identifier.")
                self._tools[instance.identifier] = instance
                logger.debug(f"Initialized tool: '{instance.identifier}' from plugin ID/alias '{plugin_id_or_alias}' (Source: {self._plugin_manager.get_plugin_source(plugin_id_or_alias)})")
            except Exception as e:
                logger.error(f"Error initializing tool plugin from ID/alias '{plugin_id_or_alias}' (class {plugin_class.__name__ if inspect.isclass(plugin_class) else plugin_class}): {e}", exc_info=True)

        logger.info(f"ToolManager initialized. Loaded {len(self._tools)} explicitly configured class-based tools.")

    def register_decorated_tools(self, functions: List[Callable], auto_enable: bool):
        """
        Processes a list of @tool decorated functions, enabling them based on the auto_enable flag.
        """
        from genie_tooling.genie import (
            FunctionToolWrapper,  # Local import to avoid circular dependency
        )

        registered_count = 0
        for func_item in functions:
            metadata = getattr(func_item, "_tool_metadata_", None)
            original_func_to_call = getattr(func_item, "_original_function_", func_item)

            if not (metadata and isinstance(metadata, dict) and callable(original_func_to_call)):
                logger.warning(f"Function '{getattr(func_item, '__name__', str(func_item))}' not @tool decorated. Skipping.")
                continue

            tool_wrapper = FunctionToolWrapper(original_func_to_call, metadata)
            tool_id = tool_wrapper.identifier

            if tool_id in self._tools:
                logger.debug(f"Tool '{tool_id}' from decorated function was already loaded from explicit configuration. Explicit config takes precedence.")
                continue

            if auto_enable:
                self._tools[tool_id] = tool_wrapper
                registered_count += 1
                logger.info(f"Auto-enabled tool '{tool_id}' from decorated function '{func_item.__name__}'.")
            else:
                logger.warning(
                    f"Tool '{tool_id}' from decorated function '{func_item.__name__}' was registered but is NOT active. "
                    "To enable it, set `auto_enable_registered_tools=True` in MiddlewareConfig or add "
                    f"'{tool_id}' to the `tool_configurations` dictionary."
                )
        if registered_count > 0:
            logger.info(f"ToolManager: Auto-enabled {registered_count} decorated tools.")


    async def list_available_formatters(self) -> List[Dict[str, str]]:
        formatters_info: List[Dict[str, str]] = []
        if not self._plugin_manager._discovered_plugin_classes:
            await self._plugin_manager.discover_plugins()

        for plugin_id, _plugin_class_obj in self._plugin_manager.list_discovered_plugin_classes().items():
            try:
                instance = await self._plugin_manager.get_plugin_instance(plugin_id, config={})
                if isinstance(instance, DefinitionFormatter):
                    formatter_instance = cast(DefinitionFormatter, instance)
                    formatters_info.append({"id": formatter_instance.formatter_id, "description": formatter_instance.description, "plugin_id": plugin_id})
            except Exception as e:
                logger.debug(f"Could not instantiate or check plugin '{plugin_id}' as DefinitionFormatter: {e}")
        return formatters_info

    async def get_tool(self, identifier: str) -> Optional[Tool]:
        tool = self._tools.get(identifier)
        if not tool:
            logger.debug(f"Tool with identifier '{identifier}' not found in ToolManager (not explicitly configured or loaded).")
        return tool

    async def list_tools(self, enabled_only: bool = True) -> List[Tool]:
        return list(self._tools.values())

    async def list_tool_summaries(self, pagination_params: Optional[Dict[str, Any]] = None) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        all_tools = await self.list_tools()
        for tool_instance in all_tools:
            try:
                metadata = await tool_instance.get_metadata()
                summaries.append({"identifier": tool_instance.identifier, "name": metadata.get("name", tool_instance.identifier), "short_description": metadata.get("description_llm", metadata.get("description_human", ""))[:120] + "...", "tags": metadata.get("tags", [])})
            except Exception as e:
                logger.error(f"Error getting metadata for tool '{tool_instance.identifier}': {e}", exc_info=True)

        default_page_size = 20; page = 1; page_size = default_page_size
        if pagination_params:
            page = max(1, int(pagination_params.get("page", 1)))
            page_size_input = pagination_params.get("page_size", default_page_size)
            try:
                page_size_val = int(page_size_input)
                if page_size_val >= 1: page_size = page_size_val
                else: page_size = default_page_size; logger.debug(f"Invalid page_size '{page_size_input}', using default {default_page_size}.")
            except (ValueError, TypeError): page_size = default_page_size; logger.debug(f"Non-integer page_size '{page_size_input}', using default {default_page_size}.")
        start_index = (page - 1) * page_size; end_index = start_index + page_size
        paginated_summaries = summaries[start_index:end_index]; total_items = len(summaries)
        total_pages = (total_items + page_size - 1) // page_size if total_items > 0 else 1
        pagination_meta = {"total_items": total_items, "current_page": page, "page_size": page_size, "total_pages": total_pages, "has_next": page < total_pages, "has_prev": page > 1}
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
