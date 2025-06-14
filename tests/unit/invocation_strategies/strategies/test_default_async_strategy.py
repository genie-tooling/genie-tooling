### tests/unit/invocation/strategies/test_default_async_strategy.py
"""Unit tests for DefaultAsyncInvocationStrategy."""
import logging
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genie_tooling.cache_providers.abc import CacheProvider
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import Plugin
from genie_tooling.error_formatters import ErrorFormatter, LLMErrorFormatter
from genie_tooling.error_handlers import DefaultErrorHandler, ErrorHandler
from genie_tooling.input_validators import (
    InputValidationException,
    InputValidator,
    JSONSchemaInputValidator,
)
from genie_tooling.invocation_strategies.impl.default_async import (
    DEFAULT_ERROR_FORMATTER_ID,
    DEFAULT_ERROR_HANDLER_ID,
    DEFAULT_TRANSFORMER_ID,
    DEFAULT_VALIDATOR_ID,
    DefaultAsyncInvocationStrategy,
)
from genie_tooling.output_transformers import (
    OutputTransformationException,
    OutputTransformer,
    PassThroughOutputTransformer,
)
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.abc import Tool

STRATEGY_LOGGER_NAME = "genie_tooling.invocation_strategies.impl.default_async"
strategy_module_logger = logging.getLogger(STRATEGY_LOGGER_NAME)


class MockToolForStrategy(Tool, Plugin):
    identifier: str = "mock_strat_tool"
    plugin_id: str = "mock_strat_tool"
    _metadata: dict
    _execute_count: int = 0
    execute_should_raise: Optional[Exception] = None

    def __init__(self, cacheable=False, cache_ttl=None, execute_result="tool_executed_successfully", input_schema=None, output_schema=None, identifier_override=None):
        self.identifier: str = identifier_override or "mock_strat_tool"
        self.plugin_id: str = identifier_override or "mock_strat_tool"
        self._metadata = {
            "identifier": self.identifier,
            "name": "Mock Tool for Strategy",
            "description_llm": "A mock tool.",
            "input_schema": input_schema or {"type": "object", "properties": {"param": {"type": "string"}}},
            "output_schema": output_schema or {"type": "object", "properties": {"output": {"type": "string"}}},
        }
        if cacheable:
            self._metadata["cacheable"] = True
        if cache_ttl is not None:
            self._metadata["cache_ttl_seconds"] = cache_ttl
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

@pytest.fixture()
def mock_plugin_manager_for_strategy(mocker) -> PluginManager:
    pm = mocker.MagicMock(spec=PluginManager)
    pm.get_plugin_instance = AsyncMock() # Default to a simple AsyncMock
    return pm

@pytest.fixture()
def mock_key_provider_for_strategy(mocker) -> KeyProvider:
    return mocker.AsyncMock(spec=KeyProvider)

@pytest.fixture()
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
    if validator and isinstance(validator, Plugin) and not hasattr(validator, "plugin_id"): type(validator).plugin_id = DEFAULT_VALIDATOR_ID
    if transformer and isinstance(transformer, Plugin) and not hasattr(transformer, "plugin_id"): type(transformer).plugin_id = DEFAULT_TRANSFORMER_ID
    if error_handler and isinstance(error_handler, Plugin) and not hasattr(error_handler, "plugin_id"): type(error_handler).plugin_id = DEFAULT_ERROR_HANDLER_ID
    if error_formatter and isinstance(error_formatter, Plugin) and not hasattr(error_formatter, "plugin_id"): type(error_formatter).plugin_id = DEFAULT_ERROR_FORMATTER_ID
    if cache_provider and isinstance(cache_provider, Plugin) and not hasattr(cache_provider, "plugin_id") and cache_provider_id_to_match: type(cache_provider).plugin_id = cache_provider_id_to_match


    async def side_effect(plugin_id_req: str, config_param=None, **kwargs_param):
        if validator is not None and plugin_id_req == getattr(validator, "plugin_id", DEFAULT_VALIDATOR_ID): return validator
        if transformer is not None and plugin_id_req == getattr(transformer, "plugin_id", DEFAULT_TRANSFORMER_ID): return transformer
        if error_handler is not None and plugin_id_req == getattr(error_handler, "plugin_id", DEFAULT_ERROR_HANDLER_ID): return error_handler
        if error_formatter is not None and plugin_id_req == getattr(error_formatter, "plugin_id", DEFAULT_ERROR_FORMATTER_ID): return error_formatter
        if cache_provider is not None and plugin_id_req == cache_provider_id_to_match: return cache_provider
        if validator is None and plugin_id_req == DEFAULT_VALIDATOR_ID: return None
        if transformer is None and plugin_id_req == DEFAULT_TRANSFORMER_ID: return None
        if error_handler is None and plugin_id_req == DEFAULT_ERROR_HANDLER_ID: return None
        if error_formatter is None and plugin_id_req == DEFAULT_ERROR_FORMATTER_ID: return None
        if cache_provider is None and plugin_id_req == cache_provider_id_to_match: return None
        if plugin_id_req == DEFAULT_VALIDATOR_ID: return JSONSchemaInputValidator()
        if plugin_id_req == DEFAULT_TRANSFORMER_ID: return PassThroughOutputTransformer()
        if plugin_id_req == DEFAULT_ERROR_HANDLER_ID: return DefaultErrorHandler()
        if plugin_id_req == DEFAULT_ERROR_FORMATTER_ID: return LLMErrorFormatter()

        strategy_module_logger.warning(f"Mock PM: Unhandled get_plugin_instance for '{plugin_id_req}'. Returning new AsyncMock.")
        generic_mock = AsyncMock(spec=Plugin)
        generic_mock.plugin_id = plugin_id_req
        return generic_mock

    pm_mock.get_plugin_instance.side_effect = side_effect


@pytest.mark.asyncio()
async def test_strategy_invoke_successful_flow(
    default_strategy: DefaultAsyncInvocationStrategy,
    mock_plugin_manager_for_strategy: PluginManager,
    mock_key_provider_for_strategy: KeyProvider
):
    tool_instance = MockToolForStrategy(execute_result={"output": "success"})
    params = {"param": "test_value"}
    await configure_pm_for_strategy(mock_plugin_manager_for_strategy) # Use defaults
    invoker_cfg = {"plugin_manager": mock_plugin_manager_for_strategy}
    result = await default_strategy.invoke(
        tool=tool_instance, params=params, key_provider=mock_key_provider_for_strategy,
        context=None, invoker_config=invoker_cfg
    )
    assert result == {"output": "success"}
    assert tool_instance._execute_count == 1

@pytest.mark.asyncio()
async def test_strategy_caching_hit_and_miss(
    default_strategy: DefaultAsyncInvocationStrategy,
    mock_plugin_manager_for_strategy: PluginManager,
    mock_key_provider_for_strategy: KeyProvider
):
    tool_instance = MockToolForStrategy(cacheable=True, cache_ttl=60, execute_result={"output": "cached_value"})
    params = {"param": "test_value"}
    mock_cache_provider = AsyncMock(spec=CacheProvider)
    mock_cache_provider.get = AsyncMock(return_value=None)
    mock_cache_provider.set = AsyncMock()
    await configure_pm_for_strategy(mock_plugin_manager_for_strategy, cache_provider=mock_cache_provider, cache_provider_id_to_match="my_cache_v1")
    invoker_cfg_with_cache = {"plugin_manager": mock_plugin_manager_for_strategy, "cache_provider_id": "my_cache_v1"}
    result1 = await default_strategy.invoke(tool=tool_instance, params=params, key_provider=mock_key_provider_for_strategy, context=None, invoker_config=invoker_cfg_with_cache)
    assert result1 == {"output": "cached_value"}; assert tool_instance._execute_count == 1
    mock_cache_provider.get.assert_awaited_once()
    mock_cache_provider.set.assert_awaited_once()
    args, kwargs = mock_cache_provider.set.call_args
    assert kwargs.get("ttl_seconds") == 60

    tool_instance.reset_execute_count(); mock_cache_provider.get.reset_mock(); mock_cache_provider.set.reset_mock()
    mock_cache_provider.get.return_value = {"output": "cached_value"}
    result2 = await default_strategy.invoke(tool=tool_instance, params=params, key_provider=mock_key_provider_for_strategy, context=None, invoker_config=invoker_cfg_with_cache)
    assert result2 == {"output": "cached_value"}; assert tool_instance._execute_count == 0
    mock_cache_provider.get.assert_awaited_once(); mock_cache_provider.set.assert_not_awaited()

@pytest.mark.asyncio()
async def test_strategy_tool_not_cacheable(
    default_strategy: DefaultAsyncInvocationStrategy,
    mock_plugin_manager_for_strategy: PluginManager,
    mock_key_provider_for_strategy: KeyProvider
):
    tool_instance = MockToolForStrategy(cacheable=False)
    params = {"param": "another_value"}
    mock_cache_provider = AsyncMock(spec=CacheProvider)
    await configure_pm_for_strategy(mock_plugin_manager_for_strategy, cache_provider=mock_cache_provider, cache_provider_id_to_match="cache_id")
    invoker_cfg_with_cache = {"plugin_manager": mock_plugin_manager_for_strategy, "cache_provider_id": "cache_id"}
    await default_strategy.invoke(tool=tool_instance, params=params, key_provider=mock_key_provider_for_strategy, context=None, invoker_config=invoker_cfg_with_cache)
    assert tool_instance._execute_count == 1
    mock_cache_provider.get.assert_not_awaited(); mock_cache_provider.set.assert_not_awaited()

@pytest.mark.asyncio()
async def test_strategy_tool_cacheable_no_ttl_in_metadata(
    default_strategy: DefaultAsyncInvocationStrategy,
    mock_plugin_manager_for_strategy: PluginManager,
    mock_key_provider_for_strategy: KeyProvider
):
    tool_instance = MockToolForStrategy(cacheable=True, cache_ttl=None, execute_result={"output": "data"})
    params = {"param": "test_value_no_meta_ttl"}
    mock_cache_provider = AsyncMock(spec=CacheProvider)
    mock_cache_provider.get = AsyncMock(return_value=None)
    mock_cache_provider.set = AsyncMock()
    await configure_pm_for_strategy(
        mock_plugin_manager_for_strategy,
        cache_provider=mock_cache_provider,
        cache_provider_id_to_match="my_cache_no_meta_ttl"
    )
    invoker_cfg = {
        "plugin_manager": mock_plugin_manager_for_strategy,
        "cache_provider_id": "my_cache_no_meta_ttl"
    }
    await default_strategy.invoke(
        tool=tool_instance, params=params, key_provider=mock_key_provider_for_strategy,
        context=None, invoker_config=invoker_cfg
    )
    assert tool_instance._execute_count == 1
    mock_cache_provider.set.assert_awaited_once()
    _args, kwargs_set = mock_cache_provider.set.call_args
    assert kwargs_set.get("ttl_seconds") is None


@pytest.mark.asyncio()
async def test_strategy_cache_provider_fails_to_load(
    default_strategy: DefaultAsyncInvocationStrategy,
    mock_plugin_manager_for_strategy: PluginManager,
    mock_key_provider_for_strategy: KeyProvider,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.WARNING, logger=STRATEGY_LOGGER_NAME)
    tool_instance = MockToolForStrategy(cacheable=True, execute_result="tool_output_cpfl")
    params = {"param": "test_cpfl"}
    await configure_pm_for_strategy(
        mock_plugin_manager_for_strategy,
        cache_provider=None,
        cache_provider_id_to_match="failing_cache_provider"
    )
    invoker_cfg = {"plugin_manager": mock_plugin_manager_for_strategy, "cache_provider_id": "failing_cache_provider"}
    result = await default_strategy.invoke(tool=tool_instance, params=params, key_provider=mock_key_provider_for_strategy, context=None, invoker_config=invoker_cfg)
    assert result == "tool_output_cpfl"
    assert tool_instance._execute_count == 1
    assert "CacheProvider 'failing_cache_provider' requested but could not be loaded. Caching will be disabled" in caplog.text

@pytest.mark.asyncio()
@patch("json.dumps")
async def test_strategy_cache_key_generation_fails(
    mock_json_dumps: MagicMock,
    default_strategy: DefaultAsyncInvocationStrategy,
    mock_plugin_manager_for_strategy: PluginManager,
    mock_key_provider_for_strategy: KeyProvider,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR, logger=STRATEGY_LOGGER_NAME)
    tool_instance = MockToolForStrategy(cacheable=True)
    params = {"param": "valid_string_for_schema"}
    mock_json_dumps.side_effect = TypeError("Simulated json.dumps failure for cache key")
    mock_cache_provider = AsyncMock(spec=CacheProvider)
    await configure_pm_for_strategy(mock_plugin_manager_for_strategy, cache_provider=mock_cache_provider, cache_provider_id_to_match="cache_id_for_key_fail_test")
    invoker_cfg = {"plugin_manager": mock_plugin_manager_for_strategy, "cache_provider_id": "cache_id_for_key_fail_test"}
    await default_strategy.invoke(tool=tool_instance, params=params, key_provider=mock_key_provider_for_strategy, context=None, invoker_config=invoker_cfg)
    assert tool_instance._execute_count == 1
    assert "Error generating cache key or reading from cache" in caplog.text
    assert "Simulated json.dumps failure for cache key" in caplog.text
    mock_cache_provider.get.assert_not_awaited(); mock_cache_provider.set.assert_not_awaited()

@pytest.mark.asyncio()
async def test_strategy_cache_get_raises_exception(
    default_strategy: DefaultAsyncInvocationStrategy,
    mock_plugin_manager_for_strategy: PluginManager,
    mock_key_provider_for_strategy: KeyProvider,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR, logger=STRATEGY_LOGGER_NAME)
    tool_instance = MockToolForStrategy(cacheable=True, execute_result="tool_res")
    params = {"param": "cache_get_fail"}
    mock_cache_provider = AsyncMock(spec=CacheProvider)
    mock_cache_provider.get.side_effect = RuntimeError("Cache GET failed")
    mock_cache_provider.set = AsyncMock()
    await configure_pm_for_strategy(
        mock_plugin_manager_for_strategy,
        cache_provider=mock_cache_provider,
        cache_provider_id_to_match="cache_get_fail_id"
    )
    invoker_cfg = {
        "plugin_manager": mock_plugin_manager_for_strategy,
        "cache_provider_id": "cache_get_fail_id"
    }
    result = await default_strategy.invoke(
        tool=tool_instance, params=params, key_provider=mock_key_provider_for_strategy,
        context=None, invoker_config=invoker_cfg
    )
    assert result == "tool_res"
    assert tool_instance._execute_count == 1
    assert "Error generating cache key or reading from cache" in caplog.text
    assert "Cache GET failed" in caplog.text
    mock_cache_provider.get.assert_awaited_once()
    mock_cache_provider.set.assert_not_awaited()


@pytest.mark.asyncio()
async def test_strategy_cache_set_fails(
    default_strategy: DefaultAsyncInvocationStrategy,
    mock_plugin_manager_for_strategy: PluginManager,
    mock_key_provider_for_strategy: KeyProvider,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR, logger=STRATEGY_LOGGER_NAME)
    tool_instance = MockToolForStrategy(cacheable=True, execute_result="res")
    params = {"param": "test"}
    mock_cache_provider = AsyncMock(spec=CacheProvider)
    mock_cache_provider.get.return_value = None
    mock_cache_provider.set.side_effect = RuntimeError("Cache set failed")
    await configure_pm_for_strategy(mock_plugin_manager_for_strategy, cache_provider=mock_cache_provider, cache_provider_id_to_match="cache_id")
    invoker_cfg = {"plugin_manager": mock_plugin_manager_for_strategy, "cache_provider_id": "cache_id"}
    result = await default_strategy.invoke(tool=tool_instance, params=params, key_provider=mock_key_provider_for_strategy, context=None, invoker_config=invoker_cfg)
    assert result == "res"; assert tool_instance._execute_count == 1
    assert "Error writing to cache" in caplog.text; mock_cache_provider.set.assert_awaited_once()

@pytest.mark.asyncio()
async def test_strategy_cache_ttl_override_from_invoker_config(
    default_strategy: DefaultAsyncInvocationStrategy,
    mock_plugin_manager_for_strategy: PluginManager,
    mock_key_provider_for_strategy: KeyProvider
):
    tool_instance = MockToolForStrategy(cacheable=True, cache_ttl=3600, execute_result={"data": "ttl_test"})
    params = {"param": "test_ttl_override"}
    mock_cache_provider = AsyncMock(spec=CacheProvider)
    mock_cache_provider.get = AsyncMock(return_value=None)
    mock_cache_provider.set = AsyncMock()
    await configure_pm_for_strategy(
        mock_plugin_manager_for_strategy,
        cache_provider=mock_cache_provider,
        cache_provider_id_to_match="cache_ttl_override_id"
    )

    invoker_cache_config_override = {"ttl_seconds": 120}
    invoker_cfg = {
        "plugin_manager": mock_plugin_manager_for_strategy,
        "cache_provider_id": "cache_ttl_override_id",
        "cache_config": invoker_cache_config_override
    }

    await default_strategy.invoke(
        tool=tool_instance, params=params, key_provider=mock_key_provider_for_strategy,
        context=None, invoker_config=invoker_cfg
    )

    mock_cache_provider.set.assert_awaited_once()
    _args, kwargs_set = mock_cache_provider.set.call_args
    assert kwargs_set.get("ttl_seconds") == 120

@pytest.mark.asyncio()
async def test_strategy_handles_input_validation_failure(
    default_strategy: DefaultAsyncInvocationStrategy,
    mock_plugin_manager_for_strategy: PluginManager,
    mock_key_provider_for_strategy: KeyProvider
):
    tool_instance = MockToolForStrategy(input_schema={"type": "object", "properties": {"num": {"type": "integer"}}, "required": ["num"]})
    params = {"num": "not_an_integer"}
    await configure_pm_for_strategy(mock_plugin_manager_for_strategy)
    invoker_cfg = {"plugin_manager": mock_plugin_manager_for_strategy}
    result = await default_strategy.invoke(tool=tool_instance, params=params, key_provider=mock_key_provider_for_strategy, context=None, invoker_config=invoker_cfg)
    assert tool_instance._execute_count == 0
    assert isinstance(result, str)
    assert "Error executing tool 'mock_strat_tool' (InputValidationException): Input validation failed." in result

@pytest.mark.asyncio()
async def test_strategy_handles_tool_execution_error(
    default_strategy: DefaultAsyncInvocationStrategy,
    mock_plugin_manager_for_strategy: PluginManager,
    mock_key_provider_for_strategy: KeyProvider
):
    tool_instance = MockToolForStrategy()
    tool_instance.execute_should_raise = ValueError("Tool failed internally!")
    params = {"param": "valid"}
    await configure_pm_for_strategy(mock_plugin_manager_for_strategy)
    invoker_cfg = {"plugin_manager": mock_plugin_manager_for_strategy}
    result = await default_strategy.invoke(tool=tool_instance, params=params, key_provider=mock_key_provider_for_strategy, context=None, invoker_config=invoker_cfg)
    assert tool_instance._execute_count == 1
    assert isinstance(result, str)
    assert "Error executing tool 'mock_strat_tool' (ValueError): Tool failed internally!" in result

@pytest.mark.asyncio()
async def test_strategy_handles_output_transformation_failure(
    default_strategy: DefaultAsyncInvocationStrategy,
    mock_plugin_manager_for_strategy: PluginManager,
    mock_key_provider_for_strategy: KeyProvider
):
    tool_instance = MockToolForStrategy(execute_result={"raw": "data"})
    params = {"param": "valid"}
    mock_transformer = MagicMock(spec=OutputTransformer)
    mock_transformer.transform = MagicMock(side_effect=OutputTransformationException("Bad transform"))
    type(mock_transformer).plugin_id = "mock_failing_transformer_v1"
    await configure_pm_for_strategy(mock_plugin_manager_for_strategy, transformer=mock_transformer)
    invoker_cfg = {"plugin_manager": mock_plugin_manager_for_strategy, "transformer_id": "mock_failing_transformer_v1"}
    result = await default_strategy.invoke(tool=tool_instance, params=params, key_provider=mock_key_provider_for_strategy, context=None, invoker_config=invoker_cfg)
    assert tool_instance._execute_count == 1
    assert isinstance(result, str)
    assert "Error executing tool 'mock_strat_tool' (OutputTransformationException): Bad transform" in result

@pytest.mark.asyncio()
async def test_strategy_critical_component_load_failure_error_handler(
    default_strategy: DefaultAsyncInvocationStrategy,
    mock_plugin_manager_for_strategy: PluginManager,
    mock_key_provider_for_strategy: KeyProvider,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.CRITICAL, logger=STRATEGY_LOGGER_NAME)
    tool_instance = MockToolForStrategy()
    params = {"param": "any"}
    await configure_pm_for_strategy(
        mock_plugin_manager_for_strategy,
        error_handler=None,
        error_formatter=LLMErrorFormatter()
    )
    invoker_cfg = {"plugin_manager": mock_plugin_manager_for_strategy}
    result = await default_strategy.invoke(tool=tool_instance, params=params, key_provider=mock_key_provider_for_strategy, context=None, invoker_config=invoker_cfg)

    assert "Critical failure: ErrorHandler or ErrorFormatter could not be loaded" in caplog.text
    assert isinstance(result, dict)
    assert result["type"] == "StrategyConfigurationError"
    assert "Critical failure: ErrorHandler or ErrorFormatter could not be loaded by strategy." in result["message"]

@pytest.mark.asyncio()
async def test_strategy_critical_component_load_failure_error_formatter(
    default_strategy: DefaultAsyncInvocationStrategy,
    mock_plugin_manager_for_strategy: PluginManager,
    mock_key_provider_for_strategy: KeyProvider,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.CRITICAL, logger=STRATEGY_LOGGER_NAME)
    tool_instance = MockToolForStrategy()
    params = {"param": "any"}
    await configure_pm_for_strategy(
        mock_plugin_manager_for_strategy,
        error_handler=DefaultErrorHandler(),
        error_formatter=None
    )
    invoker_cfg = {"plugin_manager": mock_plugin_manager_for_strategy}
    result = await default_strategy.invoke(tool=tool_instance, params=params, key_provider=mock_key_provider_for_strategy, context=None, invoker_config=invoker_cfg)

    assert "Critical failure: ErrorHandler or ErrorFormatter could not be loaded" in caplog.text
    assert isinstance(result, dict)
    assert result["type"] == "StrategyConfigurationError"
    assert "Critical failure: ErrorHandler or ErrorFormatter could not be loaded by strategy." in result["message"]


@pytest.mark.asyncio()
async def test_strategy_validator_or_transformer_fails_to_load_warning(
    default_strategy: DefaultAsyncInvocationStrategy,
    mock_plugin_manager_for_strategy: PluginManager,
    mock_key_provider_for_strategy: KeyProvider,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.WARNING, logger=STRATEGY_LOGGER_NAME)
    tool_instance = MockToolForStrategy(execute_result="ok")
    params = {"param": "any"}
    await configure_pm_for_strategy(
        mock_plugin_manager_for_strategy,
        validator=None,
        transformer=None
    )

    invoker_cfg = {"plugin_manager": mock_plugin_manager_for_strategy}
    result = await default_strategy.invoke(tool=tool_instance, params=params, key_provider=mock_key_provider_for_strategy, context=None, invoker_config=invoker_cfg)
    assert result == "ok"

    validator_load_attempt_log_found = any(
        f"InputValidator '{DEFAULT_VALIDATOR_ID}' could not be loaded. Requested: None, Default: {DEFAULT_VALIDATOR_ID}" in rec.message
        for rec in caplog.records
    )
    transformer_load_attempt_log_found = any(
        f"OutputTransformer '{DEFAULT_TRANSFORMER_ID}' could not be loaded. Requested: None, Default: {DEFAULT_TRANSFORMER_ID}" in rec.message
        for rec in caplog.records
    )
    validator_consequence_log_found = any(
        f"InputValidator could not be loaded. Input parameters for tool '{tool_instance.identifier}' will not be validated by strategy." in rec.message
        for rec in caplog.records
    )
    transformer_consequence_log_found = any(
        f"OutputTransformer could not be loaded. Tool output for tool '{tool_instance.identifier}' will not be transformed by strategy." in rec.message
        for rec in caplog.records
    )

    assert validator_load_attempt_log_found, f"Validator load attempt warning not found. Logs: {caplog.text}"
    assert validator_consequence_log_found, f"Validator load consequence warning not found. Logs: {caplog.text}"
    assert transformer_load_attempt_log_found, f"Transformer load attempt warning not found. Logs: {caplog.text}"
    assert transformer_consequence_log_found, f"Transformer load consequence warning not found. Logs: {caplog.text}"


@pytest.mark.asyncio()
async def test_strategy_unhandled_exception_within_strategy(
    default_strategy: DefaultAsyncInvocationStrategy,
    mock_plugin_manager_for_strategy: PluginManager,
    mock_key_provider_for_strategy: KeyProvider,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR, logger=STRATEGY_LOGGER_NAME)
    tool_instance = MockToolForStrategy()
    params = {"param": "test"}
    tool_instance.get_metadata = AsyncMock(side_effect=RuntimeError("Internal strategy-level problem!"))

    await configure_pm_for_strategy(mock_plugin_manager_for_strategy, error_formatter=LLMErrorFormatter())

    invoker_cfg = {"plugin_manager": mock_plugin_manager_for_strategy}
    result = await default_strategy.invoke(tool=tool_instance, params=params, key_provider=mock_key_provider_for_strategy, context=None, invoker_config=invoker_cfg)

    assert any("Unhandled error within DefaultAsyncInvocationStrategy" in rec.message and
               "Internal strategy-level problem!" in rec.message for rec in caplog.records)

    assert isinstance(result, str)
    expected_formatted_error = "Error executing tool 'mock_strat_tool' (StrategyExecutionError): Unhandled error within DefaultAsyncInvocationStrategy for tool 'mock_strat_tool': Internal strategy-level problem!"
    assert result == expected_formatted_error


@pytest.mark.asyncio()
async def test_strategy_final_error_formatter_fails(
    default_strategy: DefaultAsyncInvocationStrategy,
    mock_plugin_manager_for_strategy: PluginManager,
    mock_key_provider_for_strategy: KeyProvider,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR) # Capture ERROR and CRITICAL
    tool_instance = MockToolForStrategy(identifier_override="formatter_fail_tool")
    params = {"param": "test"}
    tool_instance.get_metadata = AsyncMock(side_effect=RuntimeError("Simulated tool metadata error"))

    failing_formatter = LLMErrorFormatter()
    type(failing_formatter).plugin_id = DEFAULT_ERROR_FORMATTER_ID
    failing_formatter.format = MagicMock(side_effect=Exception("Formatter itself crashed!"))

    await configure_pm_for_strategy(
        mock_plugin_manager_for_strategy,
        error_formatter=failing_formatter
    )

    invoker_cfg = {"plugin_manager": mock_plugin_manager_for_strategy}
    result = await default_strategy.invoke(tool=tool_instance, params=params, key_provider=mock_key_provider_for_strategy, context=None, invoker_config=invoker_cfg)

    original_error_logged = any("Unhandled error within DefaultAsyncInvocationStrategy" in rec.message and
                                "Simulated tool metadata error" in rec.message
                                for rec in caplog.records if rec.levelno == logging.ERROR)
    formatter_failure_logged = any("Critical strategy error AND error formatter failed" in rec.message and
                                   "Formatter itself crashed!" in rec.message
                                   for rec in caplog.records if rec.levelno == logging.CRITICAL)

    assert original_error_logged, f"Original strategy error not logged as ERROR. Logs: {caplog.text}"
    assert formatter_failure_logged, f"Formatter failure during critical error handling not logged. Expected 'Formatter itself crashed!' in a CRITICAL log. Logs: {caplog.text}"

    assert isinstance(result, dict)
    assert result["type"] == "CriticalStrategyAndFormatterError"
    assert "Simulated tool metadata error" in result["message"]
    assert "Formatter itself crashed!" in result["message"]


@pytest.mark.asyncio()
async def test_strategy_cache_provider_id_is_none(
    default_strategy: DefaultAsyncInvocationStrategy,
    mock_plugin_manager_for_strategy: PluginManager,
    mock_key_provider_for_strategy: KeyProvider,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.DEBUG, logger=STRATEGY_LOGGER_NAME)
    tool_instance = MockToolForStrategy(cacheable=True, execute_result="no_cache_provider")
    params = {"param": "test"}

    invoker_cfg = {"plugin_manager": mock_plugin_manager_for_strategy, "cache_provider_id": None}
    await configure_pm_for_strategy(mock_plugin_manager_for_strategy, cache_provider=None)

    result = await default_strategy.invoke(
        tool=tool_instance, params=params, key_provider=mock_key_provider_for_strategy,
        context=None, invoker_config=invoker_cfg
    )
    assert result == "no_cache_provider"
    assert tool_instance._execute_count == 1
    assert not any("CacheProvider" in rec.message and "not loaded" in rec.message for rec in caplog.records if rec.levelno >= logging.WARNING)

    cache_provider_load_attempted = False
    for call_args_item in mock_plugin_manager_for_strategy.get_plugin_instance.call_args_list:
        requested_plugin_id = call_args_item.args[0]
        if "cache_provider" in requested_plugin_id:
            cache_provider_load_attempted = True
            break
    assert not cache_provider_load_attempted, "Strategy attempted to load a cache provider when ID was None."