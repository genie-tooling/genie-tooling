### tests/unit/test_genie_facade.py
import logging
from typing import Dict
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest
from genie_tooling.command_processors.abc import CommandProcessorPlugin
from genie_tooling.command_processors.manager import CommandProcessorManager
from genie_tooling.command_processors.types import CommandProcessorResponse
from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.resolver import PLUGIN_ID_ALIASES, ConfigResolver
from genie_tooling.conversation.impl.manager import ConversationStateManager
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.genie import FunctionToolWrapper, Genie
from genie_tooling.guardrails.manager import GuardrailManager
from genie_tooling.hitl.manager import HITLManager
from genie_tooling.invocation.invoker import ToolInvoker
from genie_tooling.llm_providers.manager import LLMProviderManager
from genie_tooling.log_adapters.abc import LogAdapter as LogAdapterPlugin
from genie_tooling.log_adapters.impl.default_adapter import DefaultLogAdapter
from genie_tooling.lookup.service import ToolLookupService
from genie_tooling.observability.manager import InteractionTracingManager
from genie_tooling.prompts.llm_output_parsers.manager import LLMOutputParserManager
from genie_tooling.prompts.manager import PromptManager
from genie_tooling.rag.manager import RAGManager
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.task_queues.manager import DistributedTaskQueueManager
from genie_tooling.token_usage.manager import TokenUsageManager
from genie_tooling.tools.manager import ToolManager


@pytest.fixture()
def mock_middleware_config_facade() -> MiddlewareConfig:
    return MiddlewareConfig(features=FeatureSettings(command_processor="llm_assisted"))


@pytest.fixture()
async def mock_key_provider_instance_facade() -> KeyProvider:
    provider = MagicMock(spec=KeyProvider)
    provider.plugin_id = "mock_kp_instance_id_for_facade"
    provider.get_key = AsyncMock(return_value="mock_key")
    return provider


@pytest.fixture()
def mock_genie_dependencies(mocker):
    deps = {
        "PluginManager": mocker.patch("genie_tooling.genie.PluginManager", spec=PluginManager),
        "ToolManager": mocker.patch("genie_tooling.genie.ToolManager", spec=ToolManager),
        "ToolInvoker": mocker.patch("genie_tooling.genie.ToolInvoker", spec=ToolInvoker),
        "RAGManager": mocker.patch("genie_tooling.genie.RAGManager", spec=RAGManager),
        "ToolLookupService": mocker.patch("genie_tooling.genie.ToolLookupService", spec=ToolLookupService),
        "LLMProviderManager": mocker.patch("genie_tooling.genie.LLMProviderManager", spec=LLMProviderManager),
        "CommandProcessorManager": mocker.patch(
            "genie_tooling.genie.CommandProcessorManager", spec=CommandProcessorManager
        ),
        "InteractionTracingManager": mocker.patch(
            "genie_tooling.genie.InteractionTracingManager", spec=InteractionTracingManager
        ),
        "HITLManager": mocker.patch("genie_tooling.genie.HITLManager", spec=HITLManager),
        "TokenUsageManager": mocker.patch("genie_tooling.genie.TokenUsageManager", spec=TokenUsageManager),
        "GuardrailManager": mocker.patch("genie_tooling.genie.GuardrailManager", spec=GuardrailManager),
        "PromptManager": mocker.patch("genie_tooling.genie.PromptManager", spec=PromptManager),
        "ConversationStateManager": mocker.patch("genie_tooling.genie.ConversationStateManager", spec=ConversationStateManager),
        "LLMOutputParserManager": mocker.patch("genie_tooling.genie.LLMOutputParserManager", spec=LLMOutputParserManager),
        "DistributedTaskQueueManager": mocker.patch(
            "genie_tooling.genie.DistributedTaskQueueManager", spec=DistributedTaskQueueManager
        ),
        "DefaultLogAdapter": mocker.patch("genie_tooling.genie.DefaultLogAdapter", spec=DefaultLogAdapter),
        "ConfigResolver": mocker.patch("genie_tooling.genie.ConfigResolver", spec=ConfigResolver),
    }

    for dep_name, class_mock in deps.items():
        # The manager instances themselves are not awaitable, so use MagicMock
        instance_mock = mocker.MagicMock()
        instance_mock.plugin_id = f"mock_{dep_name.lower()}_instance_id"
        class_mock.return_value = instance_mock

        # FIX: Explicitly configure all known async methods on the mock instances
        # to be awaitable AsyncMocks. This resolves both the TypeError and the AssertionError.
        instance_mock.setup = AsyncMock()
        instance_mock.teardown = AsyncMock()

        if dep_name == "PluginManager":
            instance_mock.discover_plugins = AsyncMock()
            instance_mock.get_plugin_instance = AsyncMock()
            instance_mock.teardown_all_plugins = AsyncMock()
        elif dep_name == "ToolManager":
            instance_mock.initialize_tools = AsyncMock()
            instance_mock.register_decorated_tools = MagicMock()  # This one is sync
        elif dep_name == "ToolInvoker":
            # FIX: This was the cause of the test failure. The `invoke` method
            # must be an `AsyncMock` to be awaited in tests.
            instance_mock.invoke = AsyncMock()
        elif dep_name == "ConfigResolver":
            instance_mock.resolve = MagicMock()  # Sync
        elif dep_name == "DefaultLogAdapter":
            instance_mock.process_event = AsyncMock()
        elif dep_name == "ToolLookupService":
            instance_mock.invalidate_all_indices = AsyncMock()
        elif dep_name == "InteractionTracingManager":
            instance_mock.trace_event = AsyncMock()
    return deps


@pytest.fixture()
async def fully_mocked_genie(
    mock_genie_dependencies: Dict,
    mock_middleware_config_facade: MiddlewareConfig,
    mock_key_provider_instance_facade: KeyProvider,
) -> Genie:
    kp_instance = await mock_key_provider_instance_facade

    real_resolver = ConfigResolver()
    resolved_config_for_test = real_resolver.resolve(mock_middleware_config_facade, kp_instance)
    assert resolved_config_for_test.default_command_processor_id == PLUGIN_ID_ALIASES["llm_assisted"]
    mock_genie_dependencies["ConfigResolver"].return_value.resolve.return_value = resolved_config_for_test

    mock_cmd_proc_plugin_instance = AsyncMock(
        spec=CommandProcessorPlugin,
        process_command=AsyncMock(return_value=CommandProcessorResponse(chosen_tool_id="mock_tool", extracted_params={"p": 1})),
    )
    mock_cmd_proc_plugin_instance.plugin_id = "mock_llm_assisted_cmd_proc_v1"

    mock_genie_dependencies["CommandProcessorManager"].return_value.get_command_processor = AsyncMock(
        return_value=mock_cmd_proc_plugin_instance
    )
    # Now we just configure the return value of the already-existing AsyncMock
    mock_genie_dependencies["ToolInvoker"].return_value.invoke.return_value = {"result": "tool executed"}
    mock_genie_dependencies["HITLManager"].return_value.is_active = True
    mock_genie_dependencies["HITLManager"].return_value.request_approval = AsyncMock(return_value={"status": "approved"})

    genie_instance = await Genie.create(config=mock_middleware_config_facade, key_provider_instance=kp_instance)
    return genie_instance


class TestFunctionToolWrapper:
    def test_init_non_callable(self):
        with pytest.raises(TypeError, match="Wrapped object must be callable."):
            FunctionToolWrapper(123, {})  # type: ignore

    def test_identifier_plugin_id_derivation(self):
        def my_func():
            pass

        metadata_with_id = {"identifier": "meta_id"}
        wrapper1 = FunctionToolWrapper(my_func, metadata_with_id)
        assert wrapper1.identifier == "meta_id"
        assert wrapper1.plugin_id == "meta_id"

        metadata_no_id = {}
        wrapper2 = FunctionToolWrapper(my_func, metadata_no_id)
        assert wrapper2.identifier == "my_func"
        assert wrapper2.plugin_id == "my_func"
        assert wrapper2._metadata["identifier"] == "my_func"
        assert wrapper2._metadata["name"] == "My Func"

    @pytest.mark.asyncio()
    async def test_get_metadata(self):
        metadata = {"test": "data"}
        wrapper = FunctionToolWrapper(lambda: None, metadata)
        assert await wrapper.get_metadata() == metadata

    @pytest.mark.asyncio()
    async def test_execute_sync_function(self, mock_key_provider_instance_facade: KeyProvider):
        kp_instance = await mock_key_provider_instance_facade

        def sync_tool(a: int, b: int) -> int:
            return a + b

        wrapper = FunctionToolWrapper(sync_tool, {})
        result = await wrapper.execute({"a": 5, "b": 3}, kp_instance)
        assert result == 8

    @pytest.mark.asyncio()
    async def test_execute_async_function(self, mock_key_provider_instance_facade: KeyProvider):
        kp_instance = await mock_key_provider_instance_facade

        async def async_tool(name: str) -> str:
            return f"Hello, {name}"

        wrapper = FunctionToolWrapper(async_tool, {})
        result = await wrapper.execute({"name": "Async"}, kp_instance)
        assert result == "Hello, Async"

    @pytest.mark.asyncio()
    async def test_setup_teardown_noop(self):
        wrapper = FunctionToolWrapper(lambda: None, {})
        await wrapper.setup()
        await wrapper.teardown()


@pytest.mark.asyncio()
class TestGenieCreate:
    async def test_create_with_llm_features(self, mock_genie_dependencies, mock_key_provider_instance_facade):
        config = MiddlewareConfig(features=FeatureSettings(llm="openai", llm_openai_model_name="gpt-4"))
        kp_instance = await mock_key_provider_instance_facade

        real_resolver = ConfigResolver()
        resolved_config_for_test = real_resolver.resolve(config, kp_instance)
        mock_genie_dependencies["ConfigResolver"].return_value.resolve.return_value = resolved_config_for_test

        genie = await Genie.create(config=config, key_provider_instance=kp_instance)
        openai_id = PLUGIN_ID_ALIASES["openai"]
        assert genie._config.default_llm_provider_id == openai_id
        assert openai_id in genie._config.llm_provider_configurations
        assert genie._config.llm_provider_configurations[openai_id]["model_name"] == "gpt-4"

    async def test_create_with_rag_features(self, mock_genie_dependencies, mock_key_provider_instance_facade):
        config = MiddlewareConfig(features=FeatureSettings(rag_embedder="openai", rag_vector_store="faiss"))
        kp_instance = await mock_key_provider_instance_facade

        real_resolver = ConfigResolver()
        resolved_config_for_test = real_resolver.resolve(config, kp_instance)
        mock_genie_dependencies["ConfigResolver"].return_value.resolve.return_value = resolved_config_for_test

        genie = await Genie.create(config=config, key_provider_instance=kp_instance)
        assert genie._config.default_rag_embedder_id == PLUGIN_ID_ALIASES["openai_embedder"]
        assert genie._config.default_rag_vector_store_id == PLUGIN_ID_ALIASES["faiss_vs"]

    async def test_create_with_tool_lookup_features(self, mock_genie_dependencies, mock_key_provider_instance_facade):
        config = MiddlewareConfig(features=FeatureSettings(tool_lookup="embedding"))
        kp_instance = await mock_key_provider_instance_facade

        real_resolver = ConfigResolver()
        resolved_config_for_test = real_resolver.resolve(config, kp_instance)
        mock_genie_dependencies["ConfigResolver"].return_value.resolve.return_value = resolved_config_for_test

        genie = await Genie.create(config=config, key_provider_instance=kp_instance)
        assert genie._config.default_tool_lookup_provider_id == PLUGIN_ID_ALIASES["embedding_lookup"]

    async def test_create_with_p1_5_features(self, mock_genie_dependencies, mock_key_provider_instance_facade):
        config = MiddlewareConfig(features=FeatureSettings(observability_tracer="otel_tracer"))
        kp_instance = await mock_key_provider_instance_facade

        real_resolver = ConfigResolver()
        resolved_config_for_test = real_resolver.resolve(config, kp_instance)
        mock_genie_dependencies["ConfigResolver"].return_value.resolve.return_value = resolved_config_for_test

        genie = await Genie.create(config=config, key_provider_instance=kp_instance)
        assert genie._config.default_observability_tracer_id == PLUGIN_ID_ALIASES["otel_tracer"]

    async def test_create_with_explicit_key_provider_instance(self, mock_genie_dependencies):
        config = MiddlewareConfig()
        mock_kp = AsyncMock(spec=KeyProvider)
        mock_kp.plugin_id = "explicit_kp_v1"

        real_resolver = ConfigResolver()
        resolved_config_for_test = real_resolver.resolve(config, mock_kp)
        mock_genie_dependencies["ConfigResolver"].return_value.resolve.return_value = resolved_config_for_test

        genie = await Genie.create(config=config, key_provider_instance=mock_kp)
        assert genie._key_provider is mock_kp
        mock_genie_dependencies["PluginManager"].return_value._plugin_instances[mock_kp.plugin_id] = mock_kp

    async def test_create_with_key_provider_id_from_config(self, mock_genie_dependencies):
        mock_kp_loaded_by_id = AsyncMock(spec=KeyProvider)
        mock_kp_loaded_by_id.plugin_id = "kp_from_id_v1"

        async def get_instance_side_effect_kp_load(plugin_id, config=None):
            if plugin_id == "kp_from_id_v1":
                return mock_kp_loaded_by_id
            if plugin_id == DefaultLogAdapter.plugin_id:
                mock_log_adapter = AsyncMock(spec=LogAdapterPlugin)
                mock_log_adapter.plugin_id = DefaultLogAdapter.plugin_id
                return mock_log_adapter
            return AsyncMock()

        mock_genie_dependencies["PluginManager"].return_value.get_plugin_instance.side_effect = get_instance_side_effect_kp_load

        config = MiddlewareConfig(key_provider_id="kp_from_id_v1")
        real_resolver = ConfigResolver()
        resolved_config_for_test = real_resolver.resolve(config, key_provider_instance=None)
        mock_genie_dependencies["ConfigResolver"].return_value.resolve.return_value = resolved_config_for_test

        genie = await Genie.create(config=config)
        assert genie._key_provider is mock_kp_loaded_by_id
        assert any(
            call_args[0] == "kp_from_id_v1"
            for call_args, _ in mock_genie_dependencies["PluginManager"].return_value.get_plugin_instance.call_args_list
        )

    async def test_create_default_environment_key_provider(self, mock_genie_dependencies):
        mock_env_kp = AsyncMock(spec=KeyProvider)
        mock_env_kp.plugin_id = PLUGIN_ID_ALIASES["env_keys"]

        async def get_instance_side_effect(plugin_id, config=None):
            if plugin_id == PLUGIN_ID_ALIASES["env_keys"]:
                return mock_env_kp
            if plugin_id == DefaultLogAdapter.plugin_id:
                mock_log_adapter = AsyncMock(spec=LogAdapterPlugin)
                mock_log_adapter.plugin_id = DefaultLogAdapter.plugin_id
                return mock_log_adapter
            return AsyncMock()

        mock_genie_dependencies["PluginManager"].return_value.get_plugin_instance.side_effect = get_instance_side_effect

        config = MiddlewareConfig()
        real_resolver = ConfigResolver()
        resolved_config_for_test = real_resolver.resolve(config, key_provider_instance=None)
        mock_genie_dependencies["ConfigResolver"].return_value.resolve.return_value = resolved_config_for_test

        genie = await Genie.create(config=config)
        assert genie._key_provider is mock_env_kp

    async def test_create_key_provider_load_fails(self, mock_genie_dependencies):
        async def get_instance_kp_fail_side_effect(plugin_id, config=None):
            if plugin_id == "failing_kp_id":
                return None
            if plugin_id == DefaultLogAdapter.plugin_id:
                mock_log_adapter = AsyncMock(spec=LogAdapterPlugin)
                mock_log_adapter.plugin_id = DefaultLogAdapter.plugin_id
                return mock_log_adapter
            return AsyncMock()

        mock_genie_dependencies["PluginManager"].return_value.get_plugin_instance.side_effect = get_instance_kp_fail_side_effect

        config = MiddlewareConfig(key_provider_id="failing_kp_id")
        real_resolver = ConfigResolver()
        resolved_config_for_test = real_resolver.resolve(config, key_provider_instance=None)
        mock_genie_dependencies["ConfigResolver"].return_value.resolve.return_value = resolved_config_for_test

        with pytest.raises(RuntimeError, match="Failed to load KeyProvider with ID 'failing_kp_id'"):
            await Genie.create(config=config)

    async def test_create_log_adapter_fallback(
        self, mock_genie_dependencies, mock_key_provider_instance_facade, caplog
    ):
        caplog.set_level(logging.WARNING)
        config = MiddlewareConfig(default_log_adapter_id="non_existent_log_adapter_v1")
        kp_instance = await mock_key_provider_instance_facade

        real_resolver = ConfigResolver()
        resolved_config_for_test = real_resolver.resolve(config, kp_instance)
        mock_genie_dependencies["ConfigResolver"].return_value.resolve.return_value = resolved_config_for_test

        fallback_default_log_adapter_mock = mock_genie_dependencies["DefaultLogAdapter"].return_value
        fallback_default_log_adapter_mock.plugin_id = DefaultLogAdapter.plugin_id

        async def get_instance_side_effect(plugin_id, config=None):
            if plugin_id == "non_existent_log_adapter_v1":
                return None
            if hasattr(kp_instance, "plugin_id") and plugin_id == kp_instance.plugin_id:
                return kp_instance
            return AsyncMock()

        mock_genie_dependencies["PluginManager"].return_value.get_plugin_instance.side_effect = get_instance_side_effect

        genie = await Genie.create(config=config, key_provider_instance=kp_instance)

        assert genie._log_adapter is not None
        assert genie._log_adapter.plugin_id == DefaultLogAdapter.plugin_id
        assert (
            "Failed to load configured LogAdapter 'non_existent_log_adapter_v1'. Falling back to DefaultLogAdapter."
            in caplog.text
        )

    async def test_create_custom_log_adapter_success(self, mock_genie_dependencies, mock_key_provider_instance_facade):
        custom_log_adapter_id = "my_custom_log_adapter_v1"
        config = MiddlewareConfig(default_log_adapter_id=custom_log_adapter_id)
        kp_instance = await mock_key_provider_instance_facade

        real_resolver = ConfigResolver()
        resolved_config_for_test = real_resolver.resolve(config, kp_instance)
        mock_genie_dependencies["ConfigResolver"].return_value.resolve.return_value = resolved_config_for_test

        mock_custom_log_adapter_instance = AsyncMock(spec=LogAdapterPlugin)
        mock_custom_log_adapter_instance.plugin_id = custom_log_adapter_id

        async def get_instance_side_effect(plugin_id, config=None):
            if plugin_id == custom_log_adapter_id:
                return mock_custom_log_adapter_instance
            if hasattr(kp_instance, "plugin_id") and plugin_id == kp_instance.plugin_id:
                return kp_instance
            return AsyncMock()

        mock_genie_dependencies["PluginManager"].return_value.get_plugin_instance.side_effect = get_instance_side_effect

        genie = await Genie.create(config=config, key_provider_instance=kp_instance)
        assert genie._log_adapter is mock_custom_log_adapter_instance


@pytest.mark.asyncio()
class TestGenieRegisterToolFunctions:
    async def test_register_tool_functions_calls_tool_manager(self, fully_mocked_genie: Genie):
        genie_instance = await fully_mocked_genie
        mock_func = MagicMock(__name__="my_tool_func")
        mock_func._tool_metadata_ = {"identifier": "my_tool_func"}
        mock_func._original_function_ = lambda: "original"

        await genie_instance.register_tool_functions([mock_func])

        # Assert that the new method on ToolManager was called
        genie_instance._tool_manager.register_decorated_tools.assert_called_once_with(
            [mock_func], genie_instance._config.auto_enable_registered_tools
        )

    async def test_register_tool_functions_invalidates_index(self, fully_mocked_genie: Genie):
        genie_instance = await fully_mocked_genie
        mock_func = MagicMock(__name__="my_tool_func")
        mock_func._tool_metadata_ = {"identifier": "my_tool_func"}
        mock_func._original_function_ = lambda: "original"

        await genie_instance.register_tool_functions([mock_func])

        # Assert that the new invalidate method on ToolLookupService was called
        genie_instance._tool_lookup_service.invalidate_all_indices.assert_awaited_once()

    async def test_register_tool_functions_tool_manager_none(self, fully_mocked_genie: Genie):
        genie_instance = await fully_mocked_genie
        genie_instance._tool_manager = None
        mock_func = MagicMock(__name__="tool_no_mgr")
        mock_func._tool_metadata_ = {"identifier": "tool_no_mgr"}
        mock_func._original_function_ = lambda: "x"
        await genie_instance.register_tool_functions([mock_func])
        genie_instance._tracing_manager.trace_event.assert_awaited_with(
            "log.error", {"message": "Genie: ToolManager not initialized."}, "Genie", ANY
        )


@pytest.mark.asyncio()
class TestGenieExecuteToolExtended:
    async def test_execute_tool_invoker_none(self, fully_mocked_genie: Genie):
        genie_instance = await fully_mocked_genie
        genie_instance._tool_invoker = None
        with pytest.raises(RuntimeError, match="ToolInvoker not initialized."):
            await genie_instance.execute_tool("some_tool")

    async def test_execute_tool_key_provider_none(self, fully_mocked_genie: Genie):
        genie_instance = await fully_mocked_genie
        genie_instance._key_provider = None
        with pytest.raises(RuntimeError, match="KeyProvider not initialized."):
            await genie_instance.execute_tool("some_tool")

    async def test_execute_tool_invoke_raises_exception(self, fully_mocked_genie: Genie):
        genie_instance = await fully_mocked_genie
        genie_instance._tool_invoker.invoke.side_effect = ValueError("Tool invocation failed")  # type: ignore
        with pytest.raises(ValueError, match="Tool invocation failed"):
            await genie_instance.execute_tool("error_tool")
        genie_instance._tracing_manager.trace_event.assert_any_call(
            "genie.execute_tool.error",
            {"tool_id": "error_tool", "error": "Tool invocation failed", "type": "ValueError"},
            "Genie",
            ANY,
        )


@pytest.mark.asyncio()
class TestGenieClose:
    async def test_close_tears_down_managers_and_plugins(self, fully_mocked_genie: Genie):
        genie_instance = await fully_mocked_genie
        managers_on_genie = [
            genie_instance._log_adapter,
            genie_instance._tracing_manager,
            genie_instance._hitl_manager,
            genie_instance._token_usage_manager,
            genie_instance._guardrail_manager,
            genie_instance._prompt_manager,
            genie_instance._conversation_manager,
            genie_instance._llm_output_parser_manager,
            genie_instance._task_queue_manager,
            genie_instance._llm_provider_manager,
            genie_instance._command_processor_manager,
            genie_instance._rag_manager,
            genie_instance._tool_lookup_service,
            genie_instance._tool_invoker,
            genie_instance._tool_manager,
        ]
        plugin_manager_mock = genie_instance._plugin_manager

        await genie_instance.close()

        for manager_mock in managers_on_genie:
            if manager_mock and hasattr(manager_mock, "teardown"):
                manager_mock.teardown.assert_awaited_once()

        if plugin_manager_mock:
            plugin_manager_mock.teardown_all_plugins.assert_awaited_once()

    async def test_close_manager_teardown_error(self, fully_mocked_genie: Genie):
        genie_instance = await fully_mocked_genie

        # Capture the tracing manager mock before it gets nulled out by close()
        tracing_manager_mock = genie_instance._tracing_manager

        tool_manager_mock_instance = genie_instance._tool_manager
        if tool_manager_mock_instance:
            tool_manager_mock_instance.teardown.side_effect = RuntimeError("ToolManager teardown failed")

        rag_manager_mock_instance = genie_instance._rag_manager

        await genie_instance.close()

        # CORRECTED: The manager name comes from type(m).__name__, which for the mock is 'MagicMock'.
        expected_error_message = f"Error tearing down manager {type(tool_manager_mock_instance).__name__}: ToolManager teardown failed"
        tracing_manager_mock.trace_event.assert_any_call(
            "log.error", {"message": expected_error_message, "exc_info": True}, "Genie", None
        )

        if rag_manager_mock_instance:
            rag_manager_mock_instance.teardown.assert_awaited_once()

    async def test_close_attributes_nulled(self, fully_mocked_genie: Genie):
        genie_instance = await fully_mocked_genie
        await genie_instance.close()
        attrs_to_check_null = [
            "_plugin_manager",
            "_key_provider",
            "_config",
            "_tool_manager",
            "_tool_invoker",
            "_rag_manager",
            "_tool_lookup_service",
            "_llm_provider_manager",
            "_command_processor_manager",
            "llm",
            "rag",
            "_log_adapter",
            "_tracing_manager",
            "_hitl_manager",
            "_token_usage_manager",
            "_guardrail_manager",
            "observability",
            "human_in_loop",
            "usage",
            "_prompt_manager",
            "prompts",
            "_conversation_manager",
            "conversation",
            "_llm_output_parser_manager",
            "_task_queue_manager", # Added
            "task_queue",
        ]
        for attr_name in attrs_to_check_null:
            assert getattr(genie_instance, attr_name, "NOT_NULLED") is None, f"Attribute {attr_name} was not nulled."


@pytest.mark.asyncio()
async def test_genie_execute_tool(fully_mocked_genie: Genie):
    genie_instance = await fully_mocked_genie
    await genie_instance.execute_tool("some_tool", param="value")
    genie_instance._tool_invoker.invoke.assert_awaited_once_with(  # type: ignore
        tool_identifier="some_tool",
        params={"param": "value"},
        key_provider=genie_instance._key_provider,
        invoker_config=ANY,
        context=ANY # Allow context to be passed
    )


@pytest.mark.asyncio()
async def test_genie_register_tool_functions_no_tool_lookup_service(fully_mocked_genie: Genie, caplog):
    genie_instance = await fully_mocked_genie
    genie_instance._tool_lookup_service = None
    caplog.set_level(logging.INFO)
    mock_func = MagicMock(__name__="test_func_no_lookup")
    mock_func._tool_metadata_ = {"identifier": "test_func_no_lookup"}
    mock_func._original_function_ = lambda: "test"
    await genie_instance.register_tool_functions([mock_func])
    assert "Genie: Invalidated tool lookup index" not in caplog.text
