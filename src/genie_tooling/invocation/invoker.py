import logging
from typing import Any, Dict, Optional, cast

from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import StructuredError
from genie_tooling.error_formatters.abc import ErrorFormatter
from genie_tooling.invocation_strategies.abc import InvocationStrategy
from genie_tooling.observability.manager import InteractionTracingManager
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.abc import Tool
from genie_tooling.tools.manager import ToolManager

logger = logging.getLogger(__name__)
DEFAULT_STRATEGY_ID = "default_async_invocation_strategy_v1"
DEFAULT_INVOKER_ERROR_FORMATTER_ID = "llm_error_formatter_v1"

class ToolInvoker:
    def __init__(self, tool_manager: ToolManager, plugin_manager: PluginManager, default_strategy_id: str = DEFAULT_STRATEGY_ID):
        self._tool_manager = tool_manager
        self._plugin_manager = plugin_manager
        self._default_strategy_id = default_strategy_id
        logger.info(f"ToolInvoker initialized with default strategy: {self._default_strategy_id}")

    async def _get_default_error_formatter(self) -> Optional[ErrorFormatter]:
        try:
            formatter_any = await self._plugin_manager.get_plugin_instance(DEFAULT_INVOKER_ERROR_FORMATTER_ID) # type: ignore
            if formatter_any and isinstance(formatter_any, ErrorFormatter): return cast(ErrorFormatter, formatter_any)
            logger.error(f"Default error formatter '{DEFAULT_INVOKER_ERROR_FORMATTER_ID}' not found or invalid type.")
        except Exception as e: logger.error(f"Failed to load default error formatter '{DEFAULT_INVOKER_ERROR_FORMATTER_ID}': {e}", exc_info=True)
        return None

    async def invoke(
        self,
        tool_identifier: str,
        params: Dict[str, Any],
        key_provider: KeyProvider,
        context: Optional[Dict[str, Any]] = None,
        strategy_id: Optional[str] = None, # Note: strategy_id override is part of invoker_config for DefaultAsyncInvocationStrategy
        invoker_config: Optional[Dict[str, Any]] = None
    ) -> Any:
        current_invoker_config = invoker_config or {}
        # Extract components that DefaultAsyncInvocationStrategy expects in its invoker_config
        # These would be passed by Genie.execute_tool when setting up invoker_strategy_config
        tracing_manager: Optional[InteractionTracingManager] = current_invoker_config.get("tracing_manager")
        correlation_id: Optional[str] = current_invoker_config.get("correlation_id")

        async def _trace(event_name: str, data: Dict, level: str = "info"):
            if tracing_manager:
                # Ensure message is part of data for consistent logging format
                event_data_with_msg = data.copy()
                if "message" not in event_data_with_msg and "error" in event_data_with_msg:
                    event_data_with_msg["message"] = event_data_with_msg["error"]
                elif "message" not in event_data_with_msg:
                     event_data_with_msg["message"] = event_name.split(".")[-1]

                final_event_name = event_name if event_name.startswith("log.") else f"log.{level}"
                await tracing_manager.trace_event(final_event_name, event_data_with_msg, f"ToolInvoker:{tool_identifier}", correlation_id)

        await _trace("tool_invoker.invoke.start", {"tool_id": tool_identifier, "params": params, "strategy_id": strategy_id or self._default_strategy_id})

        tool: Optional[Tool] = await self._tool_manager.get_tool(tool_identifier) # type: ignore
        if not tool:
            error_msg = f"Tool '{tool_identifier}' not found."
            await _trace("log.warning", {"message": error_msg})
            await _trace("tool_invoker.invoke.tool_not_found", {"error": error_msg})
            structured_error: StructuredError = {"type": "ToolNotFound", "message": error_msg, "details": {"tool_id": tool_identifier}}
            formatter = await self._get_default_error_formatter()
            return formatter.format(structured_error) if formatter else structured_error

        chosen_strategy_id = strategy_id or self._default_strategy_id

        # Prepare the configuration for the strategy itself
        strategy_setup_config = {"plugin_manager": self._plugin_manager}
        # Add strategy-specific config from invoker_config if present
        strategy_specific_config = current_invoker_config.get("strategy_configurations", {}).get(chosen_strategy_id, {})
        strategy_setup_config.update(strategy_specific_config)

        strategy_instance_any = await self._plugin_manager.get_plugin_instance(chosen_strategy_id, config=strategy_setup_config) # type: ignore

        if not strategy_instance_any or not isinstance(strategy_instance_any, InvocationStrategy):
            error_msg = f"InvocationStrategy '{chosen_strategy_id}' not found or invalid."
            await _trace("log.error", {"message": error_msg})
            await _trace("tool_invoker.invoke.strategy_not_found", {"error": error_msg, "strategy_id": chosen_strategy_id})
            structured_error: StructuredError = {"type": "ConfigurationError", "message": error_msg, "details": {"strategy_id": chosen_strategy_id}}
            formatter = await self._get_default_error_formatter()
            return formatter.format(structured_error) if formatter else structured_error

        strategy_instance = cast(InvocationStrategy, strategy_instance_any)

        # Prepare the invoker_config for the strategy's invoke method
        # This now correctly uses current_invoker_config which contains PM, tracer, etc.
        # and can also contain overrides for validator_id, transformer_id etc.
        strategy_invoke_config = current_invoker_config.copy()

        try:
            await _trace("log.debug", {"message": f"Executing tool '{tool.identifier}' via strategy '{strategy_instance.plugin_id}'."})
            result = await strategy_instance.invoke(
                tool=tool,
                params=params,
                key_provider=key_provider,
                context=context, # Pass the enriched context from Genie
                invoker_config=strategy_invoke_config
            )
            await _trace("tool_invoker.invoke.strategy_success", {"result_type": type(result).__name__})
            await _trace("log.debug", {"message": f"Invocation of tool '{tool.identifier}' completed. Result type: {type(result)}"})
            return result
        except Exception as e:
            error_msg = f"Critical unhandled error during strategy execution for tool '{tool_identifier}': {e!s}"
            await _trace("log.critical", {"message": error_msg, "exc_info": True})
            await _trace("tool_invoker.invoke.strategy_error", {"error": error_msg, "exception_type": type(e).__name__})
            structured_error: StructuredError = {"type": "InternalExecutionError", "message": error_msg, "details": {"tool_id": tool_identifier, "strategy_id": chosen_strategy_id}}
            formatter = await self._get_default_error_formatter()
            return formatter.format(structured_error) if formatter else structured_error

    async def teardown(self) -> None:
        logger.info("ToolInvoker teardown (no specific resources to release here).")
        pass
