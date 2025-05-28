"""Unit tests for DefaultAsyncInvocationStrategy."""
import logging
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
# Updated import path for CacheProvider
from genie_tooling.cache_providers.abc import CacheProvider
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import Plugin
# Updated import paths for ErrorHandler, ErrorFormatter, and their implementations
from genie_tooling.error_handlers import DefaultErrorHandler, ErrorHandler
from genie_tooling.error_formatters import ErrorFormatter, LLMErrorFormatter
from genie_tooling.invocation_strategies.impl.default_async import ( # Updated path for DefaultAsyncInvocationStrategy
    DEFAULT_ERROR_FORMATTER_ID,
    DEFAULT_ERROR_HANDLER_ID,
    DEFAULT_TRANSFORMER_ID,
    DEFAULT_VALIDATOR_ID,
    DefaultAsyncInvocationStrategy,
)
# Updated import paths for OutputTransformer, OutputTransformationException, and PassThroughOutputTransformer
from genie_tooling.output_transformers import (
    OutputTransformationException,
    OutputTransformer,
    PassThroughOutputTransformer,
)
# Updated import paths for InputValidator and JSONSchemaInputValidator
from genie_tooling.input_validators import (
    InputValidator,
    JSONSchemaInputValidator,
)
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.abc import Tool


class MockToolForStrategy(Tool, Plugin):
    identifier: str = "mock_strat_tool"
    plugin_id: str = "mock_strat_tool"
    _metadata: dict
    _execute_count: int = 0
    execute_should_raise: Optional[Exception] = None

    def __init__(self, cacheable=False, cache_ttl=None, execute_result="tool_executed_successfully", input_schema=None, output_schema=None):
        self._metadata = {
            "identifier": self.identifier,
            "name": "Mock Tool for Strategy",
            "description_llm": "A mock tool.",
            "input_schema": input_schema or {"type": "object", "properties": {"param": {"type": "string"}}}, # Default schema expects a string
            "output_schema": output_schema or {"type": "object", "properties": {"output": {"type": "string"}}},
            "cacheable": cacheable,
            "cache_ttl_seconds": cache_ttl
        }
        self._execute_result = execute_result
        self._execute_count = 0
        self.execute_should_raise = None

    async def get_metadata(self):
        return self._metadata

    async def execute(self, params, key_provider, context=None):
        self._execute_count += 1
        if self.execute_should_raise:
            raise self.execute_should_raise
        if isinstance(self._execute_result, Exception):
            raise self._execute_result
        return self._execute_result

    def reset_execute_count(self):
        self._execute_count = 0

    async def setup(self, config=None): pass
    async def teardown(self): pass

@pytest.fixture
def mock_plugin_manager_for_strategy(mocker) -> PluginManager:
    pm = mocker.MagicMock(spec=PluginManager)
    async def default_get_plugin_side_effect(plugin_id_req: str, config=None, **kwargs):
        if plugin_id_req == DEFAULT_VALIDATOR_ID: return JSONSchemaInputValidator()
        if plugin_id_req == DEFAULT_TRANSFORMER_ID: return PassThroughOutputTransformer()
        if plugin_id_req == DEFAULT_ERROR_HANDLER_ID: return DefaultErrorHandler()
        if plugin_id_req == DEFAULT_ERROR_FORMATTER_ID: return LLMErrorFormatter()
        return AsyncMock(spec=Plugin)
    pm.get_plugin_instance = AsyncMock(side_effect=default_get_plugin_side_effect)
    return pm

@pytest.fixture
def mock_key_provider_for_strategy(mocker) -> KeyProvider:
    return mocker.AsyncMock(spec=KeyProvider)

@pytest.fixture
def default_strategy() -> DefaultAsyncInvocationStrategy:
    return DefaultAsyncInvocationStrategy()

async def configure_pm_for_strategy(
    pm_mock: PluginManager,
    validator: Optional[InputValidator] = JSONSchemaInputValidator(),
    transformer: Optional[OutputTransformer] = PassThroughOutputTransformer(),
    error_handler: Optional[ErrorHandler] = DefaultErrorHandler(),
    error_formatter: Optional[ErrorFormatter] = LLMErrorFormatter(),
    cache_provider: Optional[CacheProvider] = None,
    cache_provider_id_to_match: Optional[str] = "test_cache_provider_id"
):
    async def side_effect(plugin_id_req: str, config=None, **kwargs):
        if validator and plugin_id_req == validator.plugin_id: return validator
        if transformer and plugin_id_req == transformer.plugin_id: return transformer
        if error_handler and plugin_id_req == error_handler.plugin_id: return error_handler
        if error_formatter and plugin_id_req == error_formatter.plugin_id: return error_formatter
        if cache_provider and plugin_id_req == cache_provider_id_to_match: return cache_provider
        if plugin_id_req == DEFAULT_VALIDATOR_ID and not validator : return JSONSchemaInputValidator()
        if plugin_id_req == DEFAULT_TRANSFORMER_ID and not transformer : return PassThroughOutputTransformer()
        if plugin_id_req == DEFAULT_ERROR_HANDLER_ID and not error_handler : return DefaultErrorHandler()
        if plugin_id_req == DEFAULT_ERROR_FORMATTER_ID and not error_formatter : return LLMErrorFormatter()
        return AsyncMock(spec=Plugin)
    pm_mock.get_plugin_instance.side_effect = side_effect


@pytest.mark.asyncio
async def test_strategy_invoke_successful_flow(
    default_strategy: DefaultAsyncInvocationStrategy,
    mock_plugin_manager_for_strategy: PluginManager,
    mock_key_provider_for_strategy: KeyProvider
):
    tool_instance = MockToolForStrategy(execute_result={"output": "success"})
    params = {"param": "test_value"} # Valid for default schema
    await configure_pm_for_strategy(mock_plugin_manager_for_strategy)
    invoker_cfg = {"plugin_manager": mock_plugin_manager_for_strategy}

    result = await default_strategy.invoke(
        tool=tool_instance, params=params, key_provider=mock_key_provider_for_strategy,
        context=None, invoker_config=invoker_cfg
    )
    assert result == {"output": "success"}
    assert tool_instance._execute_count == 1

@pytest.mark.asyncio
async def test_strategy_caching_hit_and_miss(
    default_strategy: DefaultAsyncInvocationStrategy,
    mock_plugin_manager_for_strategy: PluginManager,
    mock_key_provider_for_strategy: KeyProvider
):
    tool_instance = MockToolForStrategy(cacheable=True, cache_ttl=60, execute_result={"output": "cached_value"})
    params = {"param": "test_value"} # Valid for default schema

    mock_cache_provider = AsyncMock(spec=CacheProvider)
    mock_cache_provider.get = AsyncMock(return_value=None)
    mock_cache_provider.set = AsyncMock()
    mock_cache_provider.plugin_id = "my_cache_v1"


    await configure_pm_for_strategy(mock_plugin_manager_for_strategy, cache_provider=mock_cache_provider, cache_provider_id_to_match="my_cache_v1")

    invoker_cfg_with_cache = {
        "plugin_manager": mock_plugin_manager_for_strategy,
        "cache_provider_id": "my_cache_v1",
    }

    result1 = await default_strategy.invoke(
        tool=tool_instance, params=params, key_provider=mock_key_provider_for_strategy,
        context=None, invoker_config=invoker_cfg_with_cache
    )
    assert result1 == {"output": "cached_value"}
    assert tool_instance._execute_count == 1
    mock_cache_provider.get.assert_awaited_once()
    mock_cache_provider.set.assert_awaited_once()
    args, kwargs = mock_cache_provider.set.call_args
    assert kwargs.get("ttl_seconds") == 60

    tool_instance.reset_execute_count()
    mock_cache_provider.get.reset_mock()
    mock_cache_provider.set.reset_mock()
    mock_cache_provider.get.return_value = {"output": "cached_value"}

    result2 = await default_strategy.invoke(
        tool=tool_instance, params=params, key_provider=mock_key_provider_for_strategy,
        context=None, invoker_config=invoker_cfg_with_cache
    )
    assert result2 == {"output": "cached_value"}
    assert tool_instance._execute_count == 0
    mock_cache_provider.get.assert_awaited_once()
    mock_cache_provider.set.assert_not_awaited()

@pytest.mark.asyncio
async def test_strategy_tool_not_cacheable(
    default_strategy: DefaultAsyncInvocationStrategy,
    mock_plugin_manager_for_strategy: PluginManager,
    mock_key_provider_for_strategy: KeyProvider
):
    tool_instance = MockToolForStrategy(cacheable=False)
    params = {"param": "another_value"} # Valid for default schema

    mock_cache_provider = AsyncMock(spec=CacheProvider)
    mock_cache_provider.plugin_id = "cache_id"
    await configure_pm_for_strategy(mock_plugin_manager_for_strategy, cache_provider=mock_cache_provider, cache_provider_id_to_match="cache_id")
    invoker_cfg_with_cache = {"plugin_manager": mock_plugin_manager_for_strategy, "cache_provider_id": "cache_id"}

    await default_strategy.invoke(
        tool=tool_instance, params=params, key_provider=mock_key_provider_for_strategy,
        context=None, invoker_config=invoker_cfg_with_cache
    )
    assert tool_instance._execute_count == 1
    mock_cache_provider.get.assert_not_awaited()
    mock_cache_provider.set.assert_not_awaited()

@pytest.mark.asyncio
async def test_strategy_cache_provider_fails_to_load(
    default_strategy: DefaultAsyncInvocationStrategy,
    mock_plugin_manager_for_strategy: PluginManager,
    mock_key_provider_for_strategy: KeyProvider,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.WARNING)
    tool_instance = MockToolForStrategy(cacheable=True)
    params = {"param": "test"} # Valid for default schema

    async def get_instance_cache_fail(plugin_id_req: str, config=None, **kwargs):
        if plugin_id_req == "failing_cache_provider": return None
        if plugin_id_req == DEFAULT_VALIDATOR_ID: return JSONSchemaInputValidator()
        if plugin_id_req == DEFAULT_TRANSFORMER_ID: return PassThroughOutputTransformer()
        if plugin_id_req == DEFAULT_ERROR_HANDLER_ID: return DefaultErrorHandler()
        if plugin_id_req == DEFAULT_ERROR_FORMATTER_ID: return LLMErrorFormatter()
        return AsyncMock(spec=Plugin)
    mock_plugin_manager_for_strategy.get_plugin_instance.side_effect = get_instance_cache_fail

    invoker_cfg = {
        "plugin_manager": mock_plugin_manager_for_strategy,
        "cache_provider_id": "failing_cache_provider"
    }
    await default_strategy.invoke(
        tool=tool_instance, params=params, key_provider=mock_key_provider_for_strategy,
        context=None, invoker_config=invoker_cfg
    )
    assert tool_instance._execute_count == 1
    assert "CacheProvider 'failing_cache_provider' requested but could not be loaded. Caching will be disabled" in caplog.text

@pytest.mark.asyncio
@patch("json.dumps") # Patch json.dumps to simulate failure during cache key generation
async def test_strategy_cache_key_generation_fails(
    mock_json_dumps: MagicMock, # Injected mock
    default_strategy: DefaultAsyncInvocationStrategy,
    mock_plugin_manager_for_strategy: PluginManager,
    mock_key_provider_for_strategy: KeyProvider,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR)
    tool_instance = MockToolForStrategy(cacheable=True)
    # Use valid params for the schema to ensure validation passes
    params = {"param": "valid_string_for_schema"}

    # Make json.dumps raise an error when called by the strategy for cache key
    mock_json_dumps.side_effect = TypeError("Simulated json.dumps failure for cache key")

    mock_cache_provider = AsyncMock(spec=CacheProvider)
    mock_cache_provider.plugin_id = "cache_id_for_key_fail_test" # Unique ID
    await configure_pm_for_strategy(
        mock_plugin_manager_for_strategy,
        cache_provider=mock_cache_provider,
        cache_provider_id_to_match="cache_id_for_key_fail_test"
    )
    invoker_cfg = {
        "plugin_manager": mock_plugin_manager_for_strategy,
        "cache_provider_id": "cache_id_for_key_fail_test"
    }

    await default_strategy.invoke(
        tool=tool_instance, params=params, key_provider=mock_key_provider_for_strategy,
        context=None, invoker_config=invoker_cfg
    )

    # If cache key generation fails, the tool should still be executed.
    assert tool_instance._execute_count == 1
    assert "Error generating cache key or reading from cache" in caplog.text
    assert "Simulated json.dumps failure for cache key" in caplog.text
    mock_cache_provider.get.assert_not_awaited()
    mock_cache_provider.set.assert_not_awaited()


@pytest.mark.asyncio
async def test_strategy_cache_set_fails(
    default_strategy: DefaultAsyncInvocationStrategy,
    mock_plugin_manager_for_strategy: PluginManager,
    mock_key_provider_for_strategy: KeyProvider,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR)
    tool_instance = MockToolForStrategy(cacheable=True, execute_result="res")
    params = {"param": "test"} # Valid for default schema

    mock_cache_provider = AsyncMock(spec=CacheProvider)
    mock_cache_provider.plugin_id = "cache_id"
    mock_cache_provider.get.return_value = None
    mock_cache_provider.set.side_effect = RuntimeError("Cache set failed")
    await configure_pm_for_strategy(mock_plugin_manager_for_strategy, cache_provider=mock_cache_provider, cache_provider_id_to_match="cache_id")
    invoker_cfg = {"plugin_manager": mock_plugin_manager_for_strategy, "cache_provider_id": "cache_id"}

    result = await default_strategy.invoke(
        tool=tool_instance, params=params, key_provider=mock_key_provider_for_strategy,
        context=None, invoker_config=invoker_cfg
    )
    assert result == "res"
    assert tool_instance._execute_count == 1
    assert "Error writing to cache" in caplog.text
    mock_cache_provider.set.assert_awaited_once()


@pytest.mark.asyncio
async def test_strategy_handles_input_validation_failure(
    default_strategy: DefaultAsyncInvocationStrategy,
    mock_plugin_manager_for_strategy: PluginManager,
    mock_key_provider_for_strategy: KeyProvider
):
    tool_instance = MockToolForStrategy(
        input_schema={"type": "object", "properties": {"num": {"type": "integer"}}, "required": ["num"]}
    )
    params = {"num": "not_an_integer"} # Invalid

    await configure_pm_for_strategy(mock_plugin_manager_for_strategy)
    invoker_cfg = {"plugin_manager": mock_plugin_manager_for_strategy}

    result = await default_strategy.invoke(
        tool=tool_instance, params=params, key_provider=mock_key_provider_for_strategy,
        context=None, invoker_config=invoker_cfg
    )
    assert tool_instance._execute_count == 0
    assert isinstance(result, str)
    assert "Error executing tool 'mock_strat_tool'" in result
    assert "(InputValidationException)" in result
    assert "Input validation failed." in result


@pytest.mark.asyncio
async def test_strategy_handles_tool_execution_error(
    default_strategy: DefaultAsyncInvocationStrategy,
    mock_plugin_manager_for_strategy: PluginManager,
    mock_key_provider_for_strategy: KeyProvider
):
    tool_instance = MockToolForStrategy()
    tool_instance.execute_should_raise = ValueError("Tool failed internally!")
    params = {"param": "valid"} # Valid for default schema

    await configure_pm_for_strategy(mock_plugin_manager_for_strategy)
    invoker_cfg = {"plugin_manager": mock_plugin_manager_for_strategy}

    result = await default_strategy.invoke(
        tool=tool_instance, params=params, key_provider=mock_key_provider_for_strategy,
        context=None, invoker_config=invoker_cfg
    )
    assert tool_instance._execute_count == 1
    assert isinstance(result, str)
    assert "Error executing tool 'mock_strat_tool' (ValueError): Tool failed internally!" in result


@pytest.mark.asyncio
async def test_strategy_handles_output_transformation_failure(
    default_strategy: DefaultAsyncInvocationStrategy,
    mock_plugin_manager_for_strategy: PluginManager,
    mock_key_provider_for_strategy: KeyProvider
):
    tool_instance = MockToolForStrategy(execute_result={"raw": "data"})
    params = {"param": "valid"} # Valid for default schema

    mock_transformer = AsyncMock(spec=OutputTransformer)
    mock_transformer.transform = MagicMock(side_effect=OutputTransformationException("Bad transform"))
    mock_transformer.plugin_id = "mock_failing_transformer_v1"


    await configure_pm_for_strategy(
        mock_plugin_manager_for_strategy,
        transformer=mock_transformer # type: ignore
    )
    invoker_cfg = {
        "plugin_manager": mock_plugin_manager_for_strategy,
        "transformer_id": "mock_failing_transformer_v1"
    }

    result = await default_strategy.invoke(
        tool=tool_instance, params=params, key_provider=mock_key_provider_for_strategy,
        context=None, invoker_config=invoker_cfg
    )
    assert tool_instance._execute_count == 1
    assert isinstance(result, str)
    assert "Error executing tool 'mock_strat_tool' (OutputTransformationException): Bad transform" in result


@pytest.mark.asyncio
async def test_strategy_critical_component_load_failure_error_handler(
    default_strategy: DefaultAsyncInvocationStrategy,
    mock_plugin_manager_for_strategy: PluginManager,
    mock_key_provider_for_strategy: KeyProvider,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.CRITICAL)
    tool_instance = MockToolForStrategy()
    params = {"param": "any"} # Valid for default schema

    async def get_instance_fail_handler(plugin_id_req: str, config=None, **kwargs):
        if plugin_id_req == DEFAULT_ERROR_HANDLER_ID: return None
        if plugin_id_req == DEFAULT_ERROR_FORMATTER_ID: return LLMErrorFormatter()
        if plugin_id_req == DEFAULT_VALIDATOR_ID: return JSONSchemaInputValidator()
        if plugin_id_req == DEFAULT_TRANSFORMER_ID: return PassThroughOutputTransformer()
        return AsyncMock(spec=Plugin)
    mock_plugin_manager_for_strategy.get_plugin_instance.side_effect = get_instance_fail_handler

    invoker_cfg = {"plugin_manager": mock_plugin_manager_for_strategy}
    result = await default_strategy.invoke(
        tool=tool_instance, params=params, key_provider=mock_key_provider_for_strategy,
        context=None, invoker_config=invoker_cfg
    )

    assert "Critical failure: ErrorHandler or ErrorFormatter could not be loaded" in caplog.text
    assert isinstance(result, dict)
    assert result["type"] == "StrategyConfigurationError"


@pytest.mark.asyncio
async def test_strategy_validator_or_transformer_fails_to_load_warning(
    default_strategy: DefaultAsyncInvocationStrategy,
    mock_plugin_manager_for_strategy: PluginManager,
    mock_key_provider_for_strategy: KeyProvider,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.WARNING)
    tool_instance = MockToolForStrategy(execute_result="ok")
    params = {"param": "any"} # Valid for default schema

    async def get_instance_fail_optional(plugin_id_req: str, config=None, **kwargs):
        if plugin_id_req == DEFAULT_VALIDATOR_ID: return None
        if plugin_id_req == DEFAULT_TRANSFORMER_ID: return None
        if plugin_id_req == DEFAULT_ERROR_HANDLER_ID: return DefaultErrorHandler()
        if plugin_id_req == DEFAULT_ERROR_FORMATTER_ID: return LLMErrorFormatter()
        return AsyncMock(spec=Plugin)
    mock_plugin_manager_for_strategy.get_plugin_instance.side_effect = get_instance_fail_optional

    invoker_cfg = {"plugin_manager": mock_plugin_manager_for_strategy}
    result = await default_strategy.invoke(
        tool=tool_instance, params=params, key_provider=mock_key_provider_for_strategy,
        context=None, invoker_config=invoker_cfg
    )
    assert result == "ok"
    assert "InputValidator could not be loaded. Input parameters for tool 'mock_strat_tool' will not be validated" in caplog.text
    assert "OutputTransformer could not be loaded. Tool output for 'mock_strat_tool' will not be transformed" in caplog.text


@pytest.mark.asyncio
async def test_strategy_unhandled_exception_within_strategy(
    default_strategy: DefaultAsyncInvocationStrategy,
    mock_plugin_manager_for_strategy: PluginManager,
    mock_key_provider_for_strategy: KeyProvider,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.CRITICAL)
    tool_instance = MockToolForStrategy()
    params = {"param": "test"} # Valid for default schema

    tool_instance.get_metadata = AsyncMock(side_effect=RuntimeError("Internal strategy-level problem!"))

    await configure_pm_for_strategy(mock_plugin_manager_for_strategy)
    invoker_cfg = {"plugin_manager": mock_plugin_manager_for_strategy}

    result = await default_strategy.invoke(
        tool=tool_instance, params=params, key_provider=mock_key_provider_for_strategy,
        context=None, invoker_config=invoker_cfg
    )

    assert "Unhandled error within DefaultAsyncInvocationStrategy" in caplog.text
    assert "Internal strategy-level problem!" in caplog.text
    assert isinstance(result, str)
    assert "Error executing tool 'mock_strat_tool' (StrategyExecutionError): Unhandled error within DefaultAsyncInvocationStrategy for tool 'mock_strat_tool': Internal strategy-level problem!" in result
