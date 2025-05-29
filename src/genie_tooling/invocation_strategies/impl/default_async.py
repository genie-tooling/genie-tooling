"""DefaultAsyncInvocationStrategy: A standard async invocation lifecycle."""
import hashlib
import json  # For stable cache key generation
import logging
from typing import Any, Dict, Optional

# Updated import path for CacheProvider
from genie_tooling.cache_providers.abc import CacheProvider
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import StructuredError
from genie_tooling.error_formatters import ErrorFormatter, LLMErrorFormatter

# Updated import paths for ErrorHandler, ErrorFormatter, and their implementations
from genie_tooling.error_handlers import DefaultErrorHandler, ErrorHandler

# Updated import paths for InputValidator, InputValidationException, and JSONSchemaInputValidator
from genie_tooling.input_validators import (
    InputValidationException,
    InputValidator,
    JSONSchemaInputValidator,
)

# Updated import path for InvocationStrategy
from genie_tooling.invocation_strategies.abc import InvocationStrategy

# Updated import paths for OutputTransformer and PassThroughOutputTransformer
from genie_tooling.output_transformers import (
    OutputTransformer,
    PassThroughOutputTransformer,
)
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.abc import Tool

logger = logging.getLogger(__name__)

# Default plugin IDs used by this strategy if not overridden by ToolInvoker
DEFAULT_VALIDATOR_ID = JSONSchemaInputValidator.plugin_id # type: ignore # Assuming plugin_id is class var
DEFAULT_TRANSFORMER_ID = PassThroughOutputTransformer.plugin_id # type: ignore
DEFAULT_ERROR_HANDLER_ID = DefaultErrorHandler.plugin_id # type: ignore
DEFAULT_ERROR_FORMATTER_ID = LLMErrorFormatter.plugin_id # type: ignore
# DEFAULT_CACHE_PROVIDER_ID = "in_memory_cache_provider_v1" # Example, assuming InMemoryCacheProvider.plugin_id

class DefaultAsyncInvocationStrategy(InvocationStrategy):
    plugin_id: str = "default_async_invocation_strategy_v1"
    description: str = "Default strategy: validates input, checks cache, executes tool, transforms output, caches result, handles errors."

    async def _get_component(self, plugin_manager: PluginManager, component_type_name: str, requested_id: Optional[str], default_id: str, expected_protocol: type) -> Optional[Any]:
        """Helper to load a component plugin."""
        chosen_id = requested_id or default_id
        component = await plugin_manager.get_plugin_instance(chosen_id)
        if component and isinstance(component, expected_protocol):
            logger.debug(f"Loaded {component_type_name}: '{chosen_id}'")
            return component
        elif component: # Loaded but wrong type
            logger.error(f"{component_type_name} '{chosen_id}' loaded but is not a valid {expected_protocol.__name__}. Requested: {requested_id}, Default: {default_id}")
        else: # Not loaded
            logger.error(f"{component_type_name} '{chosen_id}' could not be loaded. Requested: {requested_id}, Default: {default_id}")
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

        # Load essential components (error handler and formatter first for robust error reporting)
        error_handler = await self._get_component(plugin_manager, "ErrorHandler", invoker_config.get("error_handler_id"), DEFAULT_ERROR_HANDLER_ID, ErrorHandler)
        error_formatter = await self._get_component(plugin_manager, "ErrorFormatter", invoker_config.get("error_formatter_id"), DEFAULT_ERROR_FORMATTER_ID, ErrorFormatter)

        if not error_handler or not error_formatter:
            critical_error_msg = "Critical failure: ErrorHandler or ErrorFormatter could not be loaded by strategy."
            logger.critical(critical_error_msg)
            # Fallback to raw dict if formatter itself failed
            return {"type": "StrategyConfigurationError", "message": critical_error_msg}

        try:
            validator = await self._get_component(plugin_manager, "InputValidator", invoker_config.get("validator_id"), DEFAULT_VALIDATOR_ID, InputValidator)
            transformer = await self._get_component(plugin_manager, "OutputTransformer", invoker_config.get("transformer_id"), DEFAULT_TRANSFORMER_ID, OutputTransformer)

            # Cache provider is optional; only load if an ID is provided.
            cache_provider_id = invoker_config.get("cache_provider_id") # Might be None
            cache_provider: Optional[CacheProvider] = None
            if cache_provider_id:
                cache_provider = await self._get_component(plugin_manager, "CacheProvider", cache_provider_id, cache_provider_id, CacheProvider) # default is same as requested if specified
                if not cache_provider:
                    logger.warning(f"CacheProvider '{cache_provider_id}' requested but could not be loaded. Caching will be disabled for this call.")

            if not validator: # Validator is highly recommended
                 logger.warning(f"InputValidator could not be loaded. Input parameters for tool '{tool.identifier}' will not be validated by strategy.")
            if not transformer: # Transformer is also highly recommended
                 logger.warning(f"OutputTransformer could not be loaded. Tool output for '{tool.identifier}' will not be transformed by strategy.")


            tool_metadata = await tool.get_metadata()
            input_schema = tool_metadata.get("input_schema", {})
            output_schema = tool_metadata.get("output_schema", {})

            # 1. Input Validation
            validated_params = params
            if validator and input_schema:
                try:
                    # Validator might modify params (e.g., type coercion, defaults)
                    validated_params = validator.validate(params=params, schema=input_schema)
                    logger.debug(f"Input validated successfully for tool '{tool.identifier}'.")
                except InputValidationException as e_val:
                    logger.warning(f"Input validation failed for tool '{tool.identifier}': {e_val}", exc_info=True)
                    structured_err = error_handler.handle(exception=e_val, tool=tool, context=context)
                    return error_formatter.format(structured_error=structured_err)

            # 2. Caching (Read)
            cache_key: Optional[str] = None
            if cache_provider and tool_metadata.get("cacheable", False):
                # Generate a deterministic cache key
                # Including context in cache key if it influences tool output significantly.
                # For simplicity, basic key from tool_id and validated_params.
                key_material = {"tool_id": tool.identifier, "params": validated_params}
                # Sort dict for consistent hashing before serializing
                try:
                    stable_key_material_str = json.dumps(key_material, sort_keys=True, separators=(",", ":"))
                    cache_key = f"tool_cache:{tool.identifier}:{hashlib.md5(stable_key_material_str.encode('utf-8')).hexdigest()}"

                    cached_result = await cache_provider.get(cache_key)
                    if cached_result is not None:
                        logger.info(f"Cache hit for tool '{tool.identifier}', key '{cache_key}'.")
                        # Assuming cached result is already in the final, transformed format.
                        return cached_result
                    else:
                        logger.debug(f"Cache miss for tool '{tool.identifier}', key '{cache_key}'.")
                except Exception as e_cache_key:
                    logger.error(f"Error generating cache key or reading from cache for tool '{tool.identifier}': {e_cache_key}", exc_info=True)
                    cache_key = None # Disable caching for this call if key generation fails

            # 3. Execution
            logger.info(f"Executing tool '{tool.identifier}' with validated params: {validated_params}")
            raw_result: Any
            try:
                raw_result = await tool.execute(
                    params=validated_params,
                    key_provider=key_provider,
                    context=context
                )
                logger.debug(f"Tool '{tool.identifier}' executed successfully. Raw result type: {type(raw_result)}")
            except Exception as e_exec: # Catch errors from tool.execute()
                logger.error(f"Execution error for tool '{tool.identifier}': {e_exec}", exc_info=True)
                structured_err = error_handler.handle(exception=e_exec, tool=tool, context=context)
                return error_formatter.format(structured_error=structured_err)

            # 4. Output Transformation
            transformed_result = raw_result
            if transformer: # If a transformer is available
                try:
                    transformed_result = transformer.transform(output=raw_result, schema=output_schema)
                    logger.debug(f"Output transformed successfully for tool '{tool.identifier}'. Transformed result type: {type(transformed_result)}")
                except Exception as e_trans:
                    logger.error(f"Output transformation failed for tool '{tool.identifier}': {e_trans}", exc_info=True)
                    structured_err = error_handler.handle(exception=e_trans, tool=tool, context=context) # Handle transformation error
                    return error_formatter.format(structured_error=structured_err)

            # 5. Caching (Write)
            if cache_key and cache_provider and tool_metadata.get("cacheable", False): # Check again in case it was disabled
                try:
                    ttl = tool_metadata.get("cache_ttl_seconds") # Tool can specify TTL
                    cache_write_config = invoker_config.get("cache_config", {}) # Get specific cache write config
                    final_ttl = cache_write_config.get("ttl_seconds", ttl) # Override TTL via call if needed

                    await cache_provider.set(cache_key, transformed_result, ttl_seconds=final_ttl)
                    logger.info(f"Result cached for tool '{tool.identifier}', key '{cache_key}', TTL: {final_ttl}s.")
                except Exception as e_cache_write:
                    logger.error(f"Error writing to cache for tool '{tool.identifier}': {e_cache_write}", exc_info=True)

            return transformed_result

        except Exception as e_strat: # Catch-all for unexpected errors within the strategy itself
            critical_error_msg = f"Unhandled error within DefaultAsyncInvocationStrategy for tool '{tool.identifier}': {str(e_strat)}"
            logger.critical(critical_error_msg, exc_info=True)
            # Use the loaded error handler and formatter if available
            try:
                # This error is about the strategy/components, not the tool execution directly
                # so creating a generic StructuredError.
                s_err: StructuredError = {"type": "StrategyExecutionError", "message": critical_error_msg, "details": {"tool_id": tool.identifier, "strategy_id": self.plugin_id}}
                # Error handler might not be appropriate here as it's for tool errors, but formatter is useful.
                return error_formatter.format(s_err) if error_formatter else s_err
            except Exception: # If formatting also fails
                return {"type": "CriticalStrategyError", "message": critical_error_msg} # Raw dict fallback

    # async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass (defined in Plugin protocol)
    # async def teardown(self) -> None: pass
