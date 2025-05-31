### src/genie_tooling/invocation/invoker.py
"""ToolInvoker: Core component for managing the lifecycle of a tool call."""
import asyncio
import logging
from typing import Any, Dict, Optional, cast

from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import StructuredError
from genie_tooling.error_formatters.abc import ErrorFormatter
from genie_tooling.invocation_strategies.abc import (
    InvocationStrategy,
)
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.abc import Tool
from genie_tooling.tools.manager import ToolManager

# P1.5 Imports
from genie_tooling.guardrails.manager import GuardrailManager
from genie_tooling.observability.manager import InteractionTracingManager

logger = logging.getLogger(__name__)

DEFAULT_STRATEGY_ID = "default_async_invocation_strategy_v1"
DEFAULT_INVOKER_ERROR_FORMATTER_ID = "llm_error_formatter_v1"

class ToolInvoker:
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
        plugin_manager_instance = self._plugin_manager
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
        validator_id: Optional[str] = None,
        transformer_id: Optional[str] = None,
        error_handler_id: Optional[str] = None,
        error_formatter_id: Optional[str] = None,
        cache_provider_id: Optional[str] = None,
        cache_config: Optional[Dict[str, Any]] = None,
        # P1.5: Pass managers via invoker_config within the strategy's invoke call
        invoker_config: Optional[Dict[str, Any]] = None # This will carry guardrail/tracing managers
    ) -> Any:
        # P1.5: Extract managers from invoker_config if provided
        current_invoker_config = invoker_config or {}
        tracing_manager: Optional[InteractionTracingManager] = current_invoker_config.get("tracing_manager")
        guardrail_manager: Optional[GuardrailManager] = current_invoker_config.get("guardrail_manager")
        correlation_id: Optional[str] = current_invoker_config.get("correlation_id")

        async def _trace(event_name: str, data: Dict):
            if tracing_manager:
                await tracing_manager.trace_event(event_name, data, f"ToolInvoker:{tool_identifier}", correlation_id)
        
        await _trace("tool_invoker.invoke.start", {"tool_id": tool_identifier, "params": params, "strategy_id": strategy_id or self._default_strategy_id})

        tool_manager_instance = self._tool_manager
        tool: Optional[Tool] = await tool_manager_instance.get_tool(tool_identifier) # type: ignore

        if not tool:
            error_msg = f"Tool '{tool_identifier}' not found."
            logger.warning(error_msg)
            await _trace("tool_invoker.invoke.tool_not_found", {"error": error_msg})
            structured_error: StructuredError = {"type": "ToolNotFound", "message": error_msg, "details": {"tool_id": tool_identifier}}
            formatter = await self._get_default_error_formatter()
            return formatter.format(structured_error) if formatter else structured_error

        chosen_strategy_id = strategy_id or self._default_strategy_id
        plugin_manager_instance = self._plugin_manager
        strategy_instance_any = await plugin_manager_instance.get_plugin_instance(chosen_strategy_id) # type: ignore

        if not strategy_instance_any or not isinstance(strategy_instance_any, InvocationStrategy):
            error_msg = f"InvocationStrategy '{chosen_strategy_id}' not found or invalid."
            logger.error(error_msg)
            await _trace("tool_invoker.invoke.strategy_not_found", {"error": error_msg, "strategy_id": chosen_strategy_id})
            structured_error: StructuredError = {"type": "ConfigurationError", "message": error_msg, "details": {"strategy_id": chosen_strategy_id}}
            formatter = await self._get_default_error_formatter()
            return formatter.format(structured_error) if formatter else structured_error

        strategy_instance = cast(InvocationStrategy, strategy_instance_any)

        # Prepare config for the strategy's invoke method, including P1.5 managers
        # The invoker_config passed to this method might already contain these, or we add them.
        # The strategy itself will then use these managers.
        strategy_invoke_config = {
            "plugin_manager": plugin_manager_instance,
            "validator_id": validator_id,
            "transformer_id": transformer_id,
            "error_handler_id": error_handler_id,
            "error_formatter_id": error_formatter_id,
            "cache_provider_id": cache_provider_id,
            "cache_config": cache_config or {},
            "tracing_manager": tracing_manager, # Pass down
            "guardrail_manager": guardrail_manager, # Pass down
            "correlation_id": correlation_id, # Pass down
        }
        # Merge with any other invoker_config items that might have been passed directly
        strategy_invoke_config.update(current_invoker_config)


        try:
            logger.debug(f"Executing tool '{tool.identifier}' via strategy '{strategy_instance.plugin_id}'.")
            # P1.5: Tool Usage Guardrail (moved to DefaultAsyncInvocationStrategy for better lifecycle management)
            # If guardrail_manager:
            #     usage_violation = await guardrail_manager.check_tool_usage_guardrails(tool, params, context) # params might not be validated yet
            #     if usage_violation["action"] == "block":
            #         await _trace("tool_invoker.invoke.tool_usage_blocked", {"violation": usage_violation})
            #         # TODO: How to format this error? Use default error formatter?
            #         return {"error": f"Tool usage blocked by guardrail: {usage_violation.get('reason')}", "guardrail_violation": usage_violation}
            
            result = await strategy_instance.invoke(
                tool=tool,
                params=params,
                key_provider=key_provider,
                context=context,
                invoker_config=strategy_invoke_config # Pass the combined config
            )
            await _trace("tool_invoker.invoke.strategy_success", {"result_type": type(result).__name__})
            
            # P1.5: Output Guardrail (moved to DefaultAsyncInvocationStrategy)
            # if guardrail_manager and result and not (isinstance(result, dict) and result.get("error")): # Don't check errors
            #     tool_metadata = await tool.get_metadata()
            #     output_schema = tool_metadata.get("output_schema", {})
            #     output_violation = await guardrail_manager.check_output_guardrails(result, {"type": "tool_output", "tool_id": tool.identifier, "schema": output_schema})
            #     if output_violation["action"] == "block":
            #         await _trace("tool_invoker.invoke.output_blocked", {"violation": output_violation})
            #         return {"error": f"Tool output blocked by guardrail: {output_violation.get('reason')}", "original_result": result, "guardrail_violation": output_violation}
            
            logger.debug(f"Invocation of tool '{tool.identifier}' completed. Result type: {type(result)}")
            return result
        except Exception as e:
            error_msg = f"Critical unhandled error during strategy execution for tool '{tool_identifier}': {str(e)}"
            logger.critical(error_msg, exc_info=True)
            await _trace("tool_invoker.invoke.strategy_error", {"error": error_msg, "exception_type": type(e).__name__})
            structured_error: StructuredError = {
                "type": "InternalExecutionError",
                "message": error_msg,
                "details": {"tool_id": tool_identifier, "strategy_id": chosen_strategy_id}
            }
            formatter = await self._get_default_error_formatter()
            return formatter.format(structured_error) if formatter else structured_error

    async def teardown(self) -> None:
        logger.info("ToolInvoker teardown (no specific resources to release here).")
        pass