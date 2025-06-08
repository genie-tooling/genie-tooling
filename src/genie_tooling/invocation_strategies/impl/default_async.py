"""DefaultAsyncInvocationStrategy: A standard async invocation lifecycle."""
import hashlib
import json  # For stable cache key generation
import logging
from typing import Any, Dict, Optional

from genie_tooling.cache_providers.abc import CacheProvider
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import StructuredError
from genie_tooling.error_formatters import ErrorFormatter, LLMErrorFormatter
from genie_tooling.error_handlers import DefaultErrorHandler, ErrorHandler
from genie_tooling.input_validators import (
    InputValidationException,
    InputValidator,
    JSONSchemaInputValidator,
)
from genie_tooling.invocation_strategies.abc import InvocationStrategy
from genie_tooling.observability.manager import InteractionTracingManager
from genie_tooling.output_transformers import (
    OutputTransformationException,
    OutputTransformer,
    PassThroughOutputTransformer,
)
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.abc import Tool

logger = logging.getLogger(__name__)

DEFAULT_VALIDATOR_ID = JSONSchemaInputValidator.plugin_id # type: ignore
DEFAULT_TRANSFORMER_ID = PassThroughOutputTransformer.plugin_id # type: ignore
DEFAULT_ERROR_HANDLER_ID = DefaultErrorHandler.plugin_id # type: ignore
DEFAULT_ERROR_FORMATTER_ID = LLMErrorFormatter.plugin_id # type: ignore

class DefaultAsyncInvocationStrategy(InvocationStrategy):
    plugin_id: str = "default_async_invocation_strategy_v1"
    description: str = "Default strategy: validates input, checks cache, executes tool, transforms output, caches result, handles errors."

    async def _get_component(self, plugin_manager: PluginManager, component_type_name: str, requested_id: Optional[str], default_id: str, expected_protocol: type) -> Optional[Any]:
        chosen_id = requested_id or default_id
        if not chosen_id:
            logger.warning(f"{component_type_name} ID not specified and no default available. Component will not be loaded.")
            return None
        component = await plugin_manager.get_plugin_instance(chosen_id)
        if component and isinstance(component, expected_protocol):
            logger.debug(f"Loaded {component_type_name}: '{chosen_id}'")
            return component
        elif component:
            logger.error(f"{component_type_name} '{chosen_id}' loaded but is not a valid {expected_protocol.__name__}. Requested: {requested_id}, Default: {default_id}")
        else:
            logger.warning(f"{component_type_name} '{chosen_id}' could not be loaded. Requested: {requested_id}, Default: {default_id}")
        return None


    async def invoke(
        self,
        tool: Tool,
        params: Dict[str, Any],
        key_provider: KeyProvider,
        context: Optional[Dict[str, Any]],
        invoker_config: Dict[str, Any]
    ) -> Any:
        plugin_manager: PluginManager = invoker_config["plugin_manager"]
        tracing_manager: Optional[InteractionTracingManager] = invoker_config.get("tracing_manager")
        correlation_id: Optional[str] = invoker_config.get("correlation_id")

        async def _trace(event_name: str, data: Dict):
            if tracing_manager:
                await tracing_manager.trace_event(event_name, data, f"DefaultAsyncInvocationStrategy:{tool.identifier}", correlation_id)

        error_handler = await self._get_component(plugin_manager, "ErrorHandler", invoker_config.get("error_handler_id"), DEFAULT_ERROR_HANDLER_ID, ErrorHandler)
        error_formatter = await self._get_component(plugin_manager, "ErrorFormatter", invoker_config.get("error_formatter_id"), DEFAULT_ERROR_FORMATTER_ID, ErrorFormatter)

        if not error_handler or not error_formatter:
            critical_error_msg = "Critical failure: ErrorHandler or ErrorFormatter could not be loaded by strategy."
            logger.critical(critical_error_msg)
            await _trace("invocation.strategy.critical_error", {"error": critical_error_msg})
            return {"type": "StrategyConfigurationError", "message": critical_error_msg}

        try:
            validator = await self._get_component(plugin_manager, "InputValidator", invoker_config.get("validator_id"), DEFAULT_VALIDATOR_ID, InputValidator)
            transformer = await self._get_component(plugin_manager, "OutputTransformer", invoker_config.get("transformer_id"), DEFAULT_TRANSFORMER_ID, OutputTransformer)
            cache_provider_id = invoker_config.get("cache_provider_id")
            cache_provider: Optional[CacheProvider] = None
            if cache_provider_id:
                cache_provider = await self._get_component(plugin_manager, "CacheProvider", cache_provider_id, cache_provider_id, CacheProvider)
                if not cache_provider:
                    logger.warning(f"CacheProvider '{cache_provider_id}' requested but could not be loaded. Caching will be disabled for this call.")

            if not validator:
                 logger.warning(f"InputValidator could not be loaded. Input parameters for tool '{tool.identifier}' will not be validated by strategy.")
            if not transformer:
                 logger.warning(f"OutputTransformer could not be loaded. Tool output for tool '{tool.identifier}' will not be transformed by strategy.")

            tool_metadata = await tool.get_metadata()
            input_schema = tool_metadata.get("input_schema", {})
            output_schema = tool_metadata.get("output_schema", {})

            await _trace("invocation.validation.start", {"params": params, "schema_keys": list(input_schema.get("properties", {}).keys())})
            validated_params = params
            if validator and input_schema:
                try:
                    validated_params = validator.validate(params=params, schema=input_schema)
                    await _trace("invocation.validation.success", {"validated_params": validated_params})
                except InputValidationException as e_val:
                    logger.warning(f"Input validation failed for tool '{tool.identifier}': {e_val}", exc_info=True)
                    await _trace("invocation.validation.error", {"error": str(e_val), "details": e_val.errors})
                    structured_err = error_handler.handle(exception=e_val, tool=tool, context=context)
                    return error_formatter.format(structured_error=structured_err)

            cache_key: Optional[str] = None
            if cache_provider and tool_metadata.get("cacheable", False):
                try:
                    stable_key_material_str = json.dumps({"tool_id": tool.identifier, "params": validated_params}, sort_keys=True, separators=(",", ":"))
                    cache_key = f"tool_cache:{tool.identifier}:{hashlib.md5(stable_key_material_str.encode('utf-8')).hexdigest()}"
                    await _trace("invocation.cache.check", {"cache_key": cache_key})
                    cached_result = await cache_provider.get(cache_key)
                    if cached_result is not None:
                        logger.info(f"Cache hit for tool '{tool.identifier}', key '{cache_key}'.")
                        await _trace("invocation.cache.hit", {"cache_key": cache_key})
                        return cached_result
                    else:
                        await _trace("invocation.cache.miss", {"cache_key": cache_key})
                except Exception as e_cache_key:
                    logger.error(f"Error generating cache key or reading from cache for tool '{tool.identifier}': {e_cache_key}", exc_info=True)
                    await _trace("invocation.cache.error", {"error": str(e_cache_key)})
                    cache_key = None

            await _trace("tool.execute.start", {"params": validated_params})
            raw_result: Any
            try:
                raw_result = await tool.execute(params=validated_params, key_provider=key_provider, context=context)
                await _trace("tool.execute.end", {"result_type": type(raw_result).__name__})
            except Exception as e_exec:
                logger.error(f"Execution error for tool '{tool.identifier}': {e_exec}", exc_info=True)
                await _trace("tool.execute.error", {"error": str(e_exec), "type": type(e_exec).__name__})
                structured_err = error_handler.handle(exception=e_exec, tool=tool, context=context)
                return error_formatter.format(structured_error=structured_err)

            transformed_result = raw_result
            if transformer:
                await _trace("output_transformer.start", {"raw_result_type": type(raw_result).__name__})
                try:
                    transformed_result = transformer.transform(output=raw_result, schema=output_schema)
                    await _trace("output_transformer.success", {"transformed_result_type": type(transformed_result).__name__})
                except Exception as e_trans:
                    logger.error(f"Output transformation failed for tool '{tool.identifier}': {e_trans}", exc_info=True)
                    await _trace("output_transformer.error", {"error": str(e_trans), "type": type(e_trans).__name__})
                    structured_err = error_handler.handle(exception=e_trans, tool=tool, context=context)
                    return error_formatter.format(structured_error=structured_err)

            if cache_key and cache_provider and tool_metadata.get("cacheable", False):
                try:
                    ttl = tool_metadata.get("cache_ttl_seconds")
                    cache_write_config = invoker_config.get("cache_config", {})
                    final_ttl = cache_write_config.get("ttl_seconds", ttl)
                    await cache_provider.set(cache_key, transformed_result, ttl_seconds=final_ttl)
                    await _trace("invocation.cache.set", {"cache_key": cache_key, "ttl": final_ttl})
                except Exception as e_cache_write:
                    logger.error(f"Error writing to cache for tool '{tool.identifier}': {e_cache_write}", exc_info=True)
                    await _trace("invocation.cache.error", {"error": str(e_cache_write)})

            return transformed_result

        except Exception as e_strat:
            critical_error_msg = f"Unhandled error within DefaultAsyncInvocationStrategy for tool '{tool.identifier}': {str(e_strat)}"
            logger.critical(critical_error_msg, exc_info=True)
            await _trace("invocation.strategy.critical_error", {"error": critical_error_msg, "exception_type": type(e_strat).__name__})
            try:
                s_err: StructuredError = {"type": "StrategyExecutionError", "message": critical_error_msg, "details": {"tool_id": tool.identifier, "strategy_id": self.plugin_id}}
                return error_formatter.format(s_err)
            except Exception as e_format_fail:
                final_error_msg = f"Critical strategy error AND error formatter failed: {e_format_fail}. Original error: {critical_error_msg}"
                logger.critical(final_error_msg, exc_info=True)
                return {"type": "CriticalStrategyAndFormatterError", "message": f"{critical_error_msg} Additionally, the error formatter failed: {e_format_fail}"}
