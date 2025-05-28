"""ToolInvoker: Core component for managing the lifecycle of a tool call."""
import asyncio
import logging
from typing import Any, Dict, Optional, cast

from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import StructuredError
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.manager import ToolManager

from .errors import DEFAULT_INVOKER_ERROR_FORMATTER_ID, ErrorFormatter
from .strategies.abc import InvocationStrategy

logger = logging.getLogger(__name__)

DEFAULT_STRATEGY_ID = "default_async_invocation_strategy_v1"

class ToolInvoker:
    """
    Orchestrates the invocation of a tool, including strategy selection,
    parameter validation (delegated to strategy), execution, result transformation,
    caching (delegated to strategy), and error handling.
    """
    def __init__(
        self,
        tool_manager: ToolManager,
        plugin_manager: PluginManager,
        default_strategy_id: str = DEFAULT_STRATEGY_ID
    ):
        self._tool_manager = tool_manager
        self._plugin_manager = plugin_manager
        self._default_strategy_id = default_strategy_id
        logger.info(f"ToolInvoker initialized with default strategy: {self._default_strategy_id}")

    async def _get_default_error_formatter(self) -> Optional[ErrorFormatter]:
        # self._plugin_manager should be a resolved instance if fixture was defined correctly
        plugin_manager_instance = self._plugin_manager
        if asyncio.iscoroutine(plugin_manager_instance): # Should not happen if fixture is correct
            plugin_manager_instance = await plugin_manager_instance

        try:
            formatter = await plugin_manager_instance.get_plugin_instance(DEFAULT_INVOKER_ERROR_FORMATTER_ID) # type: ignore
            if formatter and isinstance(formatter, ErrorFormatter):
                return formatter
        except Exception as e:
            logger.error(f"Failed to load default error formatter '{DEFAULT_INVOKER_ERROR_FORMATTER_ID}': {e}", exc_info=True)
        return None

    async def invoke(
        self,
        tool_identifier: str,
        params: Dict[str, Any],
        key_provider: KeyProvider,
        context: Optional[Dict[str, Any]] = None,
        strategy_id: Optional[str] = None,
        validator_id: Optional[str] = None,
        transformer_id: Optional[str] = None,
        error_handler_id: Optional[str] = None,
        error_formatter_id: Optional[str] = None,
        cache_provider_id: Optional[str] = None,
        cache_config: Optional[Dict[str, Any]] = None
    ) -> Any:
        logger.debug(f"Attempting to invoke tool '{tool_identifier}' with params: {params} (strategy: {strategy_id or self._default_strategy_id})")

        # self._tool_manager should be a resolved instance
        tool_manager_instance = self._tool_manager
        if asyncio.iscoroutine(tool_manager_instance): # Should not happen
             tool_manager_instance = await tool_manager_instance
        tool = await tool_manager_instance.get_tool(tool_identifier) # type: ignore

        if not tool:
            error_msg = f"Tool '{tool_identifier}' not found."
            logger.warning(error_msg)
            structured_error: StructuredError = {"type": "ToolNotFound", "message": error_msg, "details": {"tool_id": tool_identifier}}
            formatter = await self._get_default_error_formatter()
            return formatter.format(structured_error) if formatter else structured_error

        chosen_strategy_id = strategy_id or self._default_strategy_id

        plugin_manager_instance = self._plugin_manager
        if asyncio.iscoroutine(plugin_manager_instance): # Should not happen
            plugin_manager_instance = await plugin_manager_instance
        strategy_instance_any = await plugin_manager_instance.get_plugin_instance(chosen_strategy_id) # type: ignore

        if not strategy_instance_any or not isinstance(strategy_instance_any, InvocationStrategy):
            error_msg = f"InvocationStrategy '{chosen_strategy_id}' not found or invalid."
            logger.error(error_msg)
            structured_error: StructuredError = {"type": "ConfigurationError", "message": error_msg, "details": {"strategy_id": chosen_strategy_id}}
            formatter = await self._get_default_error_formatter()
            return formatter.format(structured_error) if formatter else structured_error

        strategy_instance = cast(InvocationStrategy, strategy_instance_any)

        invoker_strategy_config = {
            "plugin_manager": plugin_manager_instance,
            "validator_id": validator_id,
            "transformer_id": transformer_id,
            "error_handler_id": error_handler_id,
            "error_formatter_id": error_formatter_id,
            "cache_provider_id": cache_provider_id,
            "cache_config": cache_config or {},
        }

        try:
            logger.debug(f"Executing tool '{tool.identifier}' via strategy '{strategy_instance.plugin_id}'.")
            result = await strategy_instance.invoke(
                tool=tool,
                params=params,
                key_provider=key_provider,
                context=context,
                invoker_config=invoker_strategy_config
            )
            logger.debug(f"Invocation of tool '{tool.identifier}' completed. Result type: {type(result)}")
            return result
        except Exception as e:
            error_msg = f"Critical unhandled error during strategy execution for tool '{tool_identifier}': {str(e)}"
            logger.critical(error_msg, exc_info=True)
            structured_error: StructuredError = {"type": "InternalExecutionError", "message": error_msg, "details": {"tool_id": tool_identifier, "strategy_id": chosen_strategy_id}}
            formatter = await self._get_default_error_formatter()
            return formatter.format(structured_error) if formatter else structured_error
