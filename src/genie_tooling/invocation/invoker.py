"""ToolInvoker: Core component for managing the lifecycle of a tool call."""
import asyncio
import logging
from typing import Any, Dict, Optional, cast

from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import StructuredError
from genie_tooling.error_formatters.abc import ErrorFormatter
from genie_tooling.invocation_strategies.abc import (
    InvocationStrategy,  # Corrected import path
)
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.abc import Tool
from genie_tooling.tools.manager import ToolManager

logger = logging.getLogger(__name__)

# Default strategy ID if none is provided to invoke()
DEFAULT_STRATEGY_ID = "default_async_invocation_strategy_v1"
# Default error formatter ID used by the invoker itself for invoker-level errors
DEFAULT_INVOKER_ERROR_FORMATTER_ID = "llm_error_formatter_v1"

class ToolInvoker:
    """
    Orchestrates the invocation of a tool, primarily by selecting and delegating
    to an InvocationStrategy. It handles high-level errors like tool/strategy
    not found.
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
        """Loads the default error formatter for invoker-level errors."""
        # Ensure plugin_manager is awaited if it's a coroutine (though typically not)
        plugin_manager_instance = self._plugin_manager
        if asyncio.iscoroutine(plugin_manager_instance): # Should not be needed if PM is sync
            plugin_manager_instance = await plugin_manager_instance # type: ignore

        try:
            formatter_any = await plugin_manager_instance.get_plugin_instance(DEFAULT_INVOKER_ERROR_FORMATTER_ID) # type: ignore
            if formatter_any and isinstance(formatter_any, ErrorFormatter):
                return cast(ErrorFormatter, formatter_any)
            logger.error(f"Default error formatter '{DEFAULT_INVOKER_ERROR_FORMATTER_ID}' not found or invalid type.")
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
        # These component IDs are passed down to the strategy
        validator_id: Optional[str] = None,
        transformer_id: Optional[str] = None,
        error_handler_id: Optional[str] = None,
        error_formatter_id: Optional[str] = None, # Formatter for strategy-level errors
        cache_provider_id: Optional[str] = None,
        cache_config: Optional[Dict[str, Any]] = None
    ) -> Any:
        logger.debug(f"Attempting to invoke tool '{tool_identifier}' with params: {params} (strategy: {strategy_id or self._default_strategy_id})")

        # Ensure tool_manager is awaited if it's a coroutine (typically not)
        tool_manager_instance = self._tool_manager
        if asyncio.iscoroutine(tool_manager_instance): # Should not be needed if TM is sync
            tool_manager_instance = await tool_manager_instance # type: ignore
        tool: Optional[Tool] = await tool_manager_instance.get_tool(tool_identifier) # type: ignore

        if not tool:
            error_msg = f"Tool '{tool_identifier}' not found."
            logger.warning(error_msg)
            structured_error: StructuredError = {"type": "ToolNotFound", "message": error_msg, "details": {"tool_id": tool_identifier}}
            formatter = await self._get_default_error_formatter()
            return formatter.format(structured_error) if formatter else structured_error

        chosen_strategy_id = strategy_id or self._default_strategy_id

        plugin_manager_instance = self._plugin_manager
        if asyncio.iscoroutine(plugin_manager_instance): # Should not be needed
            plugin_manager_instance = await plugin_manager_instance # type: ignore

        strategy_instance_any = await plugin_manager_instance.get_plugin_instance(chosen_strategy_id) # type: ignore

        if not strategy_instance_any or not isinstance(strategy_instance_any, InvocationStrategy):
            error_msg = f"InvocationStrategy '{chosen_strategy_id}' not found or invalid."
            logger.error(error_msg)
            structured_error: StructuredError = {"type": "ConfigurationError", "message": error_msg, "details": {"strategy_id": chosen_strategy_id}}
            formatter = await self._get_default_error_formatter()
            return formatter.format(structured_error) if formatter else structured_error

        strategy_instance = cast(InvocationStrategy, strategy_instance_any)

        # Prepare config for the strategy's invoke method
        invoker_strategy_config = {
            "plugin_manager": plugin_manager_instance, # Pass the (awaited if necessary) PM
            "validator_id": validator_id,
            "transformer_id": transformer_id,
            "error_handler_id": error_handler_id,
            "error_formatter_id": error_formatter_id, # Formatter for strategy to use
            "cache_provider_id": cache_provider_id,
            "cache_config": cache_config or {}, # Pass specific cache config if any
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
            # This catch block is for truly unexpected errors during strategy.invoke() itself,
            # not errors handled *by* the strategy.
            error_msg = f"Critical unhandled error during strategy execution for tool '{tool_identifier}': {str(e)}"
            logger.critical(error_msg, exc_info=True)
            structured_error: StructuredError = {
                "type": "InternalExecutionError",
                "message": error_msg,
                "details": {"tool_id": tool_identifier, "strategy_id": chosen_strategy_id}
            }
            formatter = await self._get_default_error_formatter()
            return formatter.format(structured_error) if formatter else structured_error

    async def teardown(self) -> None:
        # ToolInvoker itself doesn't own plugins, PluginManager does.
        # Teardown of strategies would be handled by PluginManager.
        logger.info("ToolInvoker teardown (no specific resources to release here).")
        pass
