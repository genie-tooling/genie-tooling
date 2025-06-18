# tests/unit/test_genie_facade.py
import logging
from typing import Dict
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from genie_tooling.command_processors.abc import CommandProcessorPlugin
from genie_tooling.command_processors.types import CommandProcessorResponse
from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.resolver import PLUGIN_ID_ALIASES, ConfigResolver
from genie_tooling.core.plugin_manager import PluginManager

# --- NEW: Import shared component types for testing ---
from genie_tooling.embedding_generators.abc import EmbeddingGeneratorPlugin
from genie_tooling.genie import FunctionToolWrapper, Genie
from genie_tooling.log_adapters.abc import LogAdapter as LogAdapterPlugin
from genie_tooling.log_adapters.impl.default_adapter import DefaultLogAdapter
from genie_tooling.redactors.impl.noop_redactor import NoOpRedactorPlugin
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.vector_stores.abc import VectorStorePlugin

# --- END NEW ---


@pytest.fixture()
def mock_middleware_config_facade() -> MiddlewareConfig:
    return MiddlewareConfig(features=FeatureSettings(command_processor="llm_assisted"))


@pytest.fixture()
def mock_key_provider_instance_facade() -> KeyProvider:
    provider = AsyncMock(spec=KeyProvider)
    provider.plugin_id = "mock_kp_instance_id_for_facade"
    provider.get_key.return_value = "mock_key"
    return provider


@pytest.fixture()
def mock_genie_dependencies(mocker):
    """
    Overhauled fixture to correctly patch manager classes.
    Each patch replaces the class, and its `return_value` is set to an AsyncMock
    instance, ensuring that when `Genie.create` instantiates them, it gets a
    fully awaitable mock object.
    """
    deps = {}
    managers_to_mock = [
        "PluginManager", "ToolManager", "ToolInvoker", "RAGManager",
        "ToolLookupService", "LLMProviderManager", "CommandProcessorManager",
        "InteractionTracingManager", "HITLManager", "TokenUsageManager",
        "GuardrailManager", "PromptManager", "ConversationStateManager",
        "LLMOutputParserManager", "DistributedTaskQueueManager", "DefaultLogAdapter",
        "ConfigResolver"
    ]

    for manager_name in managers_to_mock:
        # Create the mock instance we want to be used
        instance_mock = AsyncMock(name=f"Mock{manager_name}Instance")
        instance_mock.plugin_id = f"mock_{manager_name.lower()}_instance_id"

        # Patch the class in the 'genie' module
        class_mock = mocker.patch(f"genie_tooling.genie.{manager_name}")
        # Set the return value of the class constructor to our mock instance
        class_mock.return_value = instance_mock

        # Add attributes that are accessed during initialization to prevent AttributeErrors
        if manager_name == "PluginManager":
            instance_mock.discover_plugins = AsyncMock() # This is awaited
            instance_mock._plugin_instances = {}
            instance_mock._discovered_plugin_classes = {}
            instance_mock.list_discovered_plugin_classes = MagicMock(return_value={})
            # --- NEW: Mock for shared component loading ---
            instance_mock.get_plugin_instance = AsyncMock(return_value=AsyncMock())
            # --- END NEW ---
        if manager_name == "ToolManager":
            instance_mock.initialize_tools = AsyncMock()
        if manager_name == "ConfigResolver":
            instance_mock.resolve = MagicMock() # resolve is synchronous

        deps[manager_name] = class_mock

    return deps


@pytest.fixture()
def fully_mocked_genie(
    event_loop,
    mock_genie_dependencies: Dict,
    mock_middleware_config_facade: MiddlewareConfig,
    mock_key_provider_instance_facade: KeyProvider,
) -> Genie:
    async def setup_async():
        kp_instance = mock_key_provider_instance_facade
        real_resolver = ConfigResolver()
        resolved_config_for_test = real_resolver.resolve(mock_middleware_config_facade, kp_instance)
        mock_genie_dependencies["ConfigResolver"].return_value.resolve.return_value = resolved_config_for_test

        mock_cmd_proc_plugin_instance = AsyncMock(
            spec=CommandProcessorPlugin,
            process_command=AsyncMock(return_value=CommandProcessorResponse(chosen_tool_id="mock_tool", extracted_params={"p": 1})),
        )
        mock_cmd_proc_plugin_instance.plugin_id = "mock_llm_assisted_cmd_proc_v1"

        mock_genie_dependencies["CommandProcessorManager"].return_value.get_command_processor.return_value = mock_cmd_proc_plugin_instance
        mock_genie_dependencies["ToolInvoker"].return_value.invoke.return_value = {"result": "tool executed"}
        mock_genie_dependencies["HITLManager"].return_value.is_active = True
        mock_genie_dependencies["HITLManager"].return_value.request_approval.return_value = {"status": "approved"}

        return await Genie.create(config=mock_middleware_config_facade, key_provider_instance=kp_instance)

    return event_loop.run_until_complete(setup_async())


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
        kp_instance = mock_key_provider_instance_facade

        def sync_tool(a: int, b: int) -> int:
            return a + b

        wrapper = FunctionToolWrapper(sync_tool, {})
        result = await wrapper.execute({"a": 5, "b": 3}, kp_instance)
        assert result == 8

    @pytest.mark.asyncio()
    async def test_execute_async_function(self, mock_key_provider_instance_facade: KeyProvider):
        kp_instance = mock_key_provider_instance_facade

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
    # --- NEW TEST CASE ---
    async def test_create_instantiates_and_exposes_shared_components(self, mock_genie_dependencies):
        """
        Verify that `Genie.create` correctly instantiates shared components
        like embedders and vector stores and exposes them via accessors.
        """
        # ARRANGE
        # 1. Configure features to enable shared components
        config = MiddlewareConfig(
            features=FeatureSettings(
                rag_embedder="openai",
                rag_vector_store="chroma"
            )
        )
        # 2. Setup mock plugins to be returned by PluginManager
        mock_embedder_instance = AsyncMock(spec=EmbeddingGeneratorPlugin)
        mock_vector_store_instance = AsyncMock(spec=VectorStorePlugin)
        mock_kp_instance = AsyncMock(spec=KeyProvider)
        mock_log_adapter_instance = AsyncMock(spec=LogAdapterPlugin)

        async def get_instance_side_effect(plugin_id, config=None):
            if plugin_id == PLUGIN_ID_ALIASES["openai_embedder"]:
                return mock_embedder_instance
            if plugin_id == PLUGIN_ID_ALIASES["chroma_vs"]:
                return mock_vector_store_instance
            if plugin_id == PLUGIN_ID_ALIASES["env_keys"]:
                 return mock_kp_instance
            if plugin_id == DefaultLogAdapter.plugin_id:
                return mock_log_adapter_instance
            return AsyncMock()

        mock_genie_dependencies["PluginManager"].return_value.get_plugin_instance.side_effect = get_instance_side_effect

        # 3. Configure the resolver mock to return a resolved config
        real_resolver = ConfigResolver()
        resolved_config_for_test = real_resolver.resolve(config, key_provider_instance=mock_kp_instance)
        mock_genie_dependencies["ConfigResolver"].return_value.resolve.return_value = resolved_config_for_test

        # ACT
        genie = await Genie.create(config=config, key_provider_instance=mock_kp_instance)

        # ASSERT
        # Verify the shared components were loaded
        mock_genie_dependencies["PluginManager"].return_value.get_plugin_instance.assert_any_call(
            PLUGIN_ID_ALIASES["openai_embedder"], config=ANY
        )
        mock_genie_dependencies["PluginManager"].return_value.get_plugin_instance.assert_any_call(
            PLUGIN_ID_ALIASES["chroma_vs"], config=ANY
        )

        # Verify the private attributes are set on the Genie instance
        assert genie._default_embedder is mock_embedder_instance
        assert genie._default_vector_store is mock_vector_store_instance

        # Verify the public accessors work correctly
        assert await genie.get_default_embedder() is mock_embedder_instance
        assert await genie.get_default_vector_store() is mock_vector_store_instance

    # --- END NEW TEST CASE ---

    async def test_create_with_llm_features(self, mock_genie_dependencies, mock_key_provider_instance_facade):
        config = MiddlewareConfig(features=FeatureSettings(llm="openai", llm_openai_model_name="gpt-4"))
        kp_instance = mock_key_provider_instance_facade

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
        kp_instance = mock_key_provider_instance_facade

        real_resolver = ConfigResolver()
        resolved_config_for_test = real_resolver.resolve(config, kp_instance)
        mock_genie_dependencies["ConfigResolver"].return_value.resolve.return_value = resolved_config_for_test

        genie = await Genie.create(config=config, key_provider_instance=kp_instance)
        assert genie._config.default_rag_embedder_id == PLUGIN_ID_ALIASES["openai_embedder"]
        assert genie._config.default_rag_vector_store_id == PLUGIN_ID_ALIASES["faiss_vs"]

    async def test_create_with_tool_lookup_features(self, mock_genie_dependencies, mock_key_provider_instance_facade):
        config = MiddlewareConfig(features=FeatureSettings(tool_lookup="embedding"))
        kp_instance = mock_key_provider_instance_facade

        real_resolver = ConfigResolver()
        resolved_config_for_test = real_resolver.resolve(config, kp_instance)
        mock_genie_dependencies["ConfigResolver"].return_value.resolve.return_value = resolved_config_for_test

        genie = await Genie.create(config=config, key_provider_instance=kp_instance)
        assert genie._config.default_tool_lookup_provider_id == PLUGIN_ID_ALIASES["embedding_lookup"]

    async def test_create_with_p1_5_features(self, mock_genie_dependencies, mock_key_provider_instance_facade):
        config = MiddlewareConfig(features=FeatureSettings(observability_tracer="otel_tracer"))
        kp_instance = mock_key_provider_instance_facade

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
            call.args[0] == "kp_from_id_v1"
            for call in mock_genie_dependencies["PluginManager"].return_value.get_plugin_instance.call_args_list
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
        kp_instance = mock_key_provider_instance_facade

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
            if plugin_id == NoOpRedactorPlugin.plugin_id:
                return AsyncMock(spec=NoOpRedactorPlugin)
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
        kp_instance = mock_key_provider_instance_facade

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

    async def test_create_with_injected_plugin_manager(self, mocker):
        mock_pm = mocker.MagicMock(spec=PluginManager)
        mock_pm.discover_plugins = AsyncMock()
        mock_pm._plugin_instances = {}
        mock_pm._discovered_plugin_classes = {}
        mock_pm.list_discovered_plugin_classes = MagicMock(return_value={})

        mock_kp_instance = mocker.AsyncMock(spec=KeyProvider)
        mock_kp_instance.plugin_id = "injected_kp"

        async def mock_get_instance(plugin_id, config=None):
            if plugin_id == PLUGIN_ID_ALIASES["env_keys"]:
                return mock_kp_instance
            if plugin_id == DefaultLogAdapter.plugin_id:
                return AsyncMock(spec=LogAdapterPlugin)
            if plugin_id == NoOpRedactorPlugin.plugin_id:
                 return AsyncMock(spec=NoOpRedactorPlugin)
            return AsyncMock()

        mock_pm.get_plugin_instance = AsyncMock(side_effect=mock_get_instance)

        config = MiddlewareConfig()
        genie = await Genie.create(config=config, plugin_manager=mock_pm)

        assert genie._plugin_manager is mock_pm
        mock_pm.discover_plugins.assert_not_called()
        mock_pm.get_plugin_instance.assert_any_call(PLUGIN_ID_ALIASES["env_keys"])

    @patch("genie_tooling.genie.PluginManager")
    async def test_create_with_internal_plugin_manager_default_behavior(self, MockPluginManagerConstructor, mock_key_provider_instance_facade, mocker):
        mock_pm_instance = mocker.MagicMock(spec=PluginManager)
        mock_pm_instance.discover_plugins = AsyncMock()
        mock_pm_instance._plugin_instances = {}
        mock_pm_instance._discovered_plugin_classes = {}
        mock_pm_instance.list_discovered_plugin_classes = MagicMock(return_value={})

        async def mock_get_instance_internal(plugin_id, config=None):
            if plugin_id == PLUGIN_ID_ALIASES["env_keys"]:
                return mock_key_provider_instance_facade
            if plugin_id == DefaultLogAdapter.plugin_id:
                return AsyncMock(spec=LogAdapterPlugin)
            if plugin_id == NoOpRedactorPlugin.plugin_id:
                return AsyncMock(spec=NoOpRedactorPlugin)
            return AsyncMock()
        mock_pm_instance.get_plugin_instance = AsyncMock(side_effect=mock_get_instance_internal)

        MockPluginManagerConstructor.return_value = mock_pm_instance

        config = MiddlewareConfig()
        genie = await Genie.create(config=config)

        MockPluginManagerConstructor.assert_called_once_with(plugin_dev_dirs=config.plugin_dev_dirs)
        mock_pm_instance.discover_plugins.assert_awaited_once()
        assert genie._plugin_manager is mock_pm_instance

    async def test_create_with_injected_key_provider(self, mocker):
        mock_kp_injected = mocker.AsyncMock(spec=KeyProvider)
        mock_pm = mocker.MagicMock(spec=PluginManager)
        mock_pm.discover_plugins = AsyncMock()
        mock_pm._plugin_instances = {}
        mock_pm._discovered_plugin_classes = {}
        mock_pm.list_discovered_plugin_classes = MagicMock(return_value={})

        mock_pm.get_plugin_instance.return_value = AsyncMock(spec=LogAdapterPlugin)

        config = MiddlewareConfig()
        genie = await Genie.create(config=config, key_provider_instance=mock_kp_injected, plugin_manager=mock_pm)

        assert genie._key_provider is mock_kp_injected
        get_instance_calls = mock_pm.get_plugin_instance.call_args_list
        assert not any(call.args[0] == PLUGIN_ID_ALIASES["env_keys"] for call in get_instance_calls)


@pytest.mark.asyncio()
class TestGenieRegisterToolFunctions:
    async def test_register_tool_functions_calls_tool_manager(self, fully_mocked_genie: Genie):
        genie_instance = fully_mocked_genie
        mock_func = MagicMock(__name__="my_tool_func")
        mock_func._tool_metadata_ = {"identifier": "my_tool_func"}
        mock_func._original_function_ = lambda: "original"

        await genie_instance.register_tool_functions([mock_func])

        genie_instance._tool_manager.register_decorated_tools.assert_called_once_with(
            [mock_func], genie_instance._config.auto_enable_registered_tools
        )

    async def test_register_tool_functions_invalidates_index(self, fully_mocked_genie: Genie):
        genie_instance = fully_mocked_genie
        mock_func = MagicMock(__name__="my_tool_func")
        mock_func._tool_metadata_ = {"identifier": "my_tool_func"}
        mock_func._original_function_ = lambda: "original"
        await genie_instance.register_tool_functions([mock_func])
        genie_instance._tool_lookup_service.invalidate_all_indices.assert_awaited_once()

    async def test_register_tool_functions_tool_manager_none(self, fully_mocked_genie: Genie):
        genie_instance = fully_mocked_genie
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
        genie_instance = fully_mocked_genie
        genie_instance._tool_invoker = None
        with pytest.raises(RuntimeError, match="ToolInvoker not initialized."):
            await genie_instance.execute_tool("some_tool")

    async def test_execute_tool_key_provider_none(self, fully_mocked_genie: Genie):
        genie_instance = fully_mocked_genie
        genie_instance._key_provider = None
        with pytest.raises(RuntimeError, match="KeyProvider not initialized."):
            await genie_instance.execute_tool("some_tool")

    async def test_execute_tool_invoke_raises_exception(self, fully_mocked_genie: Genie):
        genie_instance = fully_mocked_genie
        genie_instance._tool_invoker.invoke.side_effect = ValueError("Tool invocation failed")
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
        genie_instance = fully_mocked_genie
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
        genie_instance = fully_mocked_genie
        tracing_manager_mock = genie_instance._tracing_manager

        tool_manager_mock_instance = genie_instance._tool_manager
        if tool_manager_mock_instance:
            tool_manager_mock_instance.teardown.side_effect = RuntimeError("ToolManager teardown failed")

        rag_manager_mock_instance = genie_instance._rag_manager

        await genie_instance.close()

        expected_error_message = f"Error tearing down manager {type(tool_manager_mock_instance).__name__}: ToolManager teardown failed"
        tracing_manager_mock.trace_event.assert_any_call(
            "log.error", {"message": expected_error_message, "exc_info": True}, "Genie", None
        )

        if rag_manager_mock_instance:
            rag_manager_mock_instance.teardown.assert_awaited_once()

    async def test_close_attributes_nulled(self, fully_mocked_genie: Genie):
        genie_instance = fully_mocked_genie
        await genie_instance.close()
        attrs_to_check_null = [
            "_plugin_manager", "_key_provider", "_config", "_tool_manager", "_tool_invoker",
            "_rag_manager", "_tool_lookup_service", "_llm_provider_manager",
            "_command_processor_manager", "llm", "rag", "_log_adapter", "_tracing_manager",
            "_hitl_manager", "_token_usage_manager", "_guardrail_manager", "observability",
            "human_in_loop", "usage", "_prompt_manager", "prompts", "_conversation_manager",
            "conversation", "_llm_output_parser_manager", "_task_queue_manager", "task_queue",
        ]
        for attr_name in attrs_to_check_null:
            assert getattr(genie_instance, attr_name, "NOT_NULLED") is None, f"Attribute {attr_name} was not nulled."


@pytest.mark.asyncio()
async def test_genie_execute_tool(fully_mocked_genie: Genie):
    genie_instance = fully_mocked_genie
    await genie_instance.execute_tool("some_tool", param="value")
    genie_instance._tool_invoker.invoke.assert_awaited_once_with(
        tool_identifier="some_tool",
        params={"param": "value"},
        key_provider=genie_instance._key_provider,
        invoker_config=ANY,
        context=ANY
    )


@pytest.mark.asyncio()
async def test_genie_register_tool_functions_no_tool_lookup_service(fully_mocked_genie: Genie, caplog):
    genie_instance = fully_mocked_genie
    genie_instance._tool_lookup_service = None
    caplog.set_level(logging.INFO)
    mock_func = MagicMock(__name__="test_func_no_lookup")
    mock_func._tool_metadata_ = {"identifier": "test_func_no_lookup"}
    mock_func._original_function_ = lambda: "original"
    await genie_instance.register_tool_functions([mock_func])
    assert "Genie: Invalidated tool lookup index" not in caplog.text
