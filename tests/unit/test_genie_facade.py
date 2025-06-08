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
from genie_tooling.prompts.conversation.impl.manager import ConversationStateManager
from genie_tooling.prompts.llm_output_parsers.manager import LLMOutputParserManager
from genie_tooling.prompts.manager import PromptManager
from genie_tooling.rag.manager import RAGManager
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.task_queues.manager import DistributedTaskQueueManager
from genie_tooling.token_usage.manager import TokenUsageManager
from genie_tooling.tools.manager import ToolManager


@pytest.fixture
def mock_middleware_config_facade() -> MiddlewareConfig:
    return MiddlewareConfig(features=FeatureSettings(command_processor="llm_assisted"))

@pytest.fixture
async def mock_key_provider_instance_facade() -> KeyProvider:
    provider = MagicMock(spec=KeyProvider)
    provider.plugin_id = "mock_kp_instance_id_for_facade"
    provider.get_key = AsyncMock(return_value="mock_key")
    return provider

@pytest.fixture
def mock_genie_dependencies(mocker):
    # Mock the ToolManager to have a real dictionary for _tools and a mockable method
    mock_tool_manager_instance = MagicMock(spec=ToolManager)
    mock_tool_manager_instance._tools = {}
    mock_tool_manager_instance.register_decorated_tools = MagicMock()
    mock_tool_manager_instance.initialize_tools = AsyncMock()
    mock_tool_manager_instance.teardown = AsyncMock()

    deps = {
        "PluginManager": mocker.patch("genie_tooling.genie.PluginManager", spec=PluginManager),
        "ToolManager": mocker.patch("genie_tooling.genie.ToolManager", return_value=mock_tool_manager_instance),
        "ToolInvoker": mocker.patch("genie_tooling.genie.ToolInvoker", spec=ToolInvoker),
        "RAGManager": mocker.patch("genie_tooling.genie.RAGManager", spec=RAGManager),
        "ToolLookupService": mocker.patch("genie_tooling.genie.ToolLookupService", spec=ToolLookupService),
        "LLMProviderManager": mocker.patch("genie_tooling.genie.LLMProviderManager", spec=LLMProviderManager),
        "CommandProcessorManager": mocker.patch("genie_tooling.genie.CommandProcessorManager", spec=CommandProcessorManager),
        "InteractionTracingManager": mocker.patch("genie_tooling.genie.InteractionTracingManager", spec=InteractionTracingManager),
        "HITLManager": mocker.patch("genie_tooling.genie.HITLManager", spec=HITLManager),
        "TokenUsageManager": mocker.patch("genie_tooling.genie.TokenUsageManager", spec=TokenUsageManager),
        "GuardrailManager": mocker.patch("genie_tooling.genie.GuardrailManager", spec=GuardrailManager),
        "PromptManager": mocker.patch("genie_tooling.genie.PromptManager", spec=PromptManager),
        "ConversationStateManager": mocker.patch("genie_tooling.genie.ConversationStateManager", spec=ConversationStateManager),
        "LLMOutputParserManager": mocker.patch("genie_tooling.genie.LLMOutputParserManager", spec=LLMOutputParserManager),
        "DistributedTaskQueueManager": mocker.patch("genie_tooling.genie.DistributedTaskQueueManager", spec=DistributedTaskQueueManager),
        "DefaultLogAdapter": mocker.patch("genie_tooling.genie.DefaultLogAdapter", spec=DefaultLogAdapter),
        "ConfigResolver": mocker.patch("genie_tooling.genie.ConfigResolver", spec=ConfigResolver),
    }

    # Standard mocking for other managers
    for dep_name, class_mock in deps.items():
        if dep_name == "ToolManager": continue # Already handled
        instance_mock = AsyncMock()
        instance_mock.plugin_id = f"mock_{dep_name.lower()}_instance_id"
        class_mock.return_value = instance_mock

        if not hasattr(instance_mock, "teardown") or not isinstance(instance_mock.teardown, AsyncMock):
            instance_mock.teardown = AsyncMock()

        if dep_name == "PluginManager":
            instance_mock.discover_plugins = AsyncMock()
            instance_mock.get_plugin_instance = AsyncMock()
            instance_mock.teardown_all_plugins = AsyncMock()
        elif dep_name == "ConfigResolver":
            instance_mock.resolve = MagicMock()
        elif dep_name == "DefaultLogAdapter":
            instance_mock.process_event = AsyncMock()
    return deps

@pytest.fixture
async def fully_mocked_genie(
    mock_genie_dependencies: Dict,
    mock_middleware_config_facade: MiddlewareConfig,
    mock_key_provider_instance_facade: KeyProvider
) -> Genie:
    kp_instance = await mock_key_provider_instance_facade

    real_resolver = ConfigResolver()
    resolved_config_for_test = real_resolver.resolve(
        mock_middleware_config_facade,
        kp_instance
    )
    mock_genie_dependencies["ConfigResolver"].return_value.resolve.return_value = resolved_config_for_test

    mock_cmd_proc_plugin_instance = AsyncMock(
        spec=CommandProcessorPlugin,
        process_command=AsyncMock(return_value=CommandProcessorResponse(
            chosen_tool_id="mock_tool", extracted_params={"p": 1}
        ))
    )
    mock_cmd_proc_plugin_instance.plugin_id = "mock_llm_assisted_cmd_proc_v1"

    mock_genie_dependencies["CommandProcessorManager"].return_value.get_command_processor = AsyncMock(
        return_value=mock_cmd_proc_plugin_instance
    )
    mock_genie_dependencies["ToolInvoker"].return_value.invoke = AsyncMock(return_value={"result": "tool executed"})
    mock_genie_dependencies["HITLManager"].return_value.is_active = True
    mock_genie_dependencies["HITLManager"].return_value.request_approval = AsyncMock(return_value={"status": "approved"})

    # Mock the new incremental update method on the ToolLookupService mock
    mock_genie_dependencies["ToolLookupService"].return_value.add_or_update_tools = AsyncMock()

    genie_instance = await Genie.create(
        config=mock_middleware_config_facade,
        key_provider_instance=kp_instance
    )
    return genie_instance


class TestFunctionToolWrapper:
    def test_init_non_callable(self):
        with pytest.raises(TypeError, match="Wrapped object must be callable."):
            FunctionToolWrapper(123, {}) # type: ignore

    def test_identifier_plugin_id_derivation(self):
        def my_func(): pass
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

    @pytest.mark.asyncio
    async def test_get_metadata(self):
        metadata = {"test": "data"}
        wrapper = FunctionToolWrapper(lambda: None, metadata)
        assert await wrapper.get_metadata() == metadata

    @pytest.mark.asyncio
    async def test_execute_sync_function(self, mock_key_provider_instance_facade: KeyProvider):
        kp_instance = await mock_key_provider_instance_facade
        def sync_tool(a: int, b: int) -> int:
            return a + b
        wrapper = FunctionToolWrapper(sync_tool, {})
        result = await wrapper.execute({"a": 5, "b": 3}, kp_instance)
        assert result == 8

    @pytest.mark.asyncio
    async def test_execute_async_function(self, mock_key_provider_instance_facade: KeyProvider):
        kp_instance = await mock_key_provider_instance_facade
        async def async_tool(name: str) -> str:
            return f"Hello, {name}"
        wrapper = FunctionToolWrapper(async_tool, {})
        result = await wrapper.execute({"name": "Async"}, kp_instance)
        assert result == "Hello, Async"

    @pytest.mark.asyncio
    async def test_setup_teardown_noop(self):
        wrapper = FunctionToolWrapper(lambda: None, {})
        await wrapper.setup()
        await wrapper.teardown()


@pytest.mark.asyncio
class TestGenieCreate:
    async def test_create_with_llm_features(
        self, mock_genie_dependencies, mock_key_provider_instance_facade
    ):
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

@pytest.mark.asyncio
class TestGenieRegisterToolFunctions:
    @patch('genie_tooling.tools.manager.logger')
    async def test_register_tool_functions_not_decorated(self, mock_tool_manager_logger, fully_mocked_genie: Genie):
        genie_instance = await fully_mocked_genie
        
        def non_decorated_func(): pass
        
        # Simulate the real ToolManager's logic inside the mock for this test
        def register_side_effect(functions, auto_enable):
            for func in functions:
                if not hasattr(func, "_tool_metadata_"):
                    mock_tool_manager_logger.warning(f"Function '{func.__name__}' not @tool decorated. Skipping.")
        
        genie_instance._tool_manager.register_decorated_tools.side_effect = register_side_effect

        await genie_instance.register_tool_functions([non_decorated_func])
        
        mock_tool_manager_logger.warning.assert_called_with("Function 'non_decorated_func' not @tool decorated. Skipping.")
        assert not genie_instance._tool_manager._tools

    @patch('genie_tooling.tools.manager.logger')
    async def test_register_tool_functions_skips_already_loaded_tool(self, mock_tool_manager_logger, fully_mocked_genie: Genie):
        genie_instance = await fully_mocked_genie
        
        mock_explicit_tool = MagicMock(spec=ToolPlugin)
        genie_instance._tool_manager._tools = {"my_tool_func": mock_explicit_tool}

        mock_decorated_func = MagicMock(__name__="my_tool_func")
        mock_decorated_func._tool_metadata_ = {"identifier": "my_tool_func"}
        mock_decorated_func._original_function_ = lambda: "new_version"

        def register_side_effect(functions, auto_enable):
            tool_wrapper = FunctionToolWrapper(mock_decorated_func._original_function_, mock_decorated_func._tool_metadata_)
            if tool_wrapper.identifier in genie_instance._tool_manager._tools:
                mock_tool_manager_logger.debug(f"Tool '{tool_wrapper.identifier}' from decorated function was already loaded from explicit configuration. Explicit config takes precedence.")
        
        genie_instance._tool_manager.register_decorated_tools.side_effect = register_side_effect

        await genie_instance.register_tool_functions([mock_decorated_func])

        # This is the mock that will be returned by the fallback DefaultLogAdapter() call
        # It's already set up in mock_genie_dependencies
        fallback_default_log_adapter_mock = mock_genie_dependencies["DefaultLogAdapter"].return_value
        fallback_default_log_adapter_mock.plugin_id = DefaultLogAdapter.plugin_id

        async def get_instance_side_effect(plugin_id, config=None):
            if plugin_id == "non_existent_log_adapter_v1":
                return None
            # When Genie.create tries to load DefaultLogAdapter as a fallback,
            # it directly instantiates it. The PluginManager is not involved in this specific fallback instantiation.
            # However, if the DefaultLogAdapter itself was requested by ID, this mock would catch it.
            if plugin_id == DefaultLogAdapter.plugin_id:
                 return fallback_default_log_adapter_mock # Should not be hit by the fallback logic in Genie.create
            if hasattr(kp_instance, "plugin_id") and plugin_id == kp_instance.plugin_id:
                return kp_instance
            return AsyncMock()
        mock_genie_dependencies["PluginManager"].return_value.get_plugin_instance.side_effect = get_instance_side_effect

        genie = await Genie.create(config=config, key_provider_instance=kp_instance)

        assert genie._log_adapter is not None
        assert genie._log_adapter.plugin_id == DefaultLogAdapter.plugin_id
        assert "Failed to load configured LogAdapter 'non_existent_log_adapter_v1'. Falling back to DefaultLogAdapter." in caplog.text

        def register_side_effect(functions, auto_enable):
            assert auto_enable is True
            tool_wrapper = FunctionToolWrapper(mock_func._original_function_, mock_func._tool_metadata_)
            if auto_enable:
                genie_instance._tool_manager._tools[tool_wrapper.identifier] = tool_wrapper
                mock_tool_manager_logger.info(f"Auto-enabled tool '{tool_wrapper.identifier}' from decorated function '{mock_func.__name__}'.")

        genie_instance._tool_manager.register_decorated_tools.side_effect = register_side_effect

        await genie_instance.register_tool_functions([mock_func])

        mock_custom_log_adapter_instance = AsyncMock(spec=LogAdapterPlugin)
        mock_custom_log_adapter_instance.plugin_id = custom_log_adapter_id
        async def get_instance_side_effect(plugin_id, config=None):
            if plugin_id == custom_log_adapter_id:
                return mock_custom_log_adapter_instance
            if hasattr(kp_instance, "plugin_id") and plugin_id == kp_instance.plugin_id:
                return kp_instance
            return AsyncMock()
        mock_genie_dependencies["PluginManager"].return_value.get_plugin_instance.side_effect = get_instance_side_effect

    @patch('genie_tooling.tools.manager.logger')
    async def test_register_tool_functions_auto_enable_false(self, mock_tool_manager_logger, fully_mocked_genie: Genie):
        genie_instance = await fully_mocked_genie
        genie_instance._config.auto_enable_registered_tools = False

        mock_func = MagicMock(__name__="unactivated_tool")
        mock_func._tool_metadata_ = {"identifier": "unactivated_tool"}
        mock_func._original_function_ = lambda: "action"

        def register_side_effect(functions, auto_enable):
            assert auto_enable is False
            tool_wrapper = FunctionToolWrapper(mock_func._original_function_, mock_func._tool_metadata_)
            if not auto_enable:
                mock_tool_manager_logger.warning(f"Tool '{tool_wrapper.identifier}' from decorated function '{mock_func.__name__}' was registered but is NOT active. To enable it, set `auto_enable_registered_tools=True` in MiddlewareConfig or add '{tool_wrapper.identifier}' to the `tool_configurations` dictionary.")

        genie_instance._tool_manager.register_decorated_tools.side_effect = register_side_effect

        await genie_instance.register_tool_functions([mock_func])

        mock_func_new = MagicMock(__name__="my_tool_func_new_version")
        mock_func_new._tool_metadata_ = {"identifier": "my_tool_func"}
        mock_func_new._original_function_ = lambda: "new_version"
        await genie_instance.register_tool_functions([mock_func_new])
        assert "Tool 'my_tool_func' already registered. Overwriting." in caplog.text
        assert genie_instance._tool_manager._tools["my_tool_func"]._func() == "new_version" # type: ignore


    async def test_register_tool_functions_with_tool_lookup_service(self, fully_mocked_genie: Genie):
        genie_instance = await fully_mocked_genie
        mock_func = MagicMock(__name__="another_tool")
        mock_func._tool_metadata_ = {"identifier": "another_tool"}
        mock_func._original_function_ = lambda: "tool_action"

        await genie_instance.register_tool_functions([mock_func])
        genie_instance._tool_lookup_service.invalidate_index.assert_called_once()

    async def test_register_tool_functions_tool_manager_none(self, fully_mocked_genie: Genie, caplog):
        genie_instance = await fully_mocked_genie
        caplog.set_level(logging.ERROR)
        genie_instance._tool_manager = None
        mock_func = MagicMock(__name__="tool_no_mgr")
        mock_func._tool_metadata_ = {"identifier": "tool_no_mgr"}
        mock_func._original_function_ = lambda: "x"
        await genie_instance.register_tool_functions([mock_func])
        assert "Genie: ToolManager not initialized." in caplog.text

@pytest.mark.asyncio
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
        genie_instance._tool_invoker.invoke.side_effect = ValueError("Tool invocation failed")
        with pytest.raises(ValueError, match="Tool invocation failed"):
            await genie_instance.execute_tool("error_tool")
        genie_instance._tracing_manager.trace_event.assert_any_call(
            "genie.execute_tool.error",
            {"tool_id": "error_tool", "error": "Tool invocation failed", "type": "ValueError"},
            "Genie", ANY
        )


@pytest.mark.asyncio
class TestGenieRunCommand:
    async def test_run_command_cmd_proc_mgr_none(self, fully_mocked_genie: Genie):
        genie_instance = await fully_mocked_genie
        genie_instance._command_processor_manager = None
        result = await genie_instance.run_command("test")
        assert result == {"error": "CommandProcessorManager not initialized."}

    async def test_run_command_no_default_processor(self, fully_mocked_genie: Genie):
        genie_instance = await fully_mocked_genie
        original_default_cmd_proc_id = genie_instance._config.default_command_processor_id
        genie_instance._config.default_command_processor_id = None
        try:
            result = await genie_instance.run_command("test", processor_id=None)
            assert result == {"error": "No command processor configured."}
        finally:
            genie_instance._config.default_command_processor_id = original_default_cmd_proc_id

    async def test_run_command_processor_not_found(self, fully_mocked_genie: Genie):
        genie_instance = await fully_mocked_genie
        genie_instance._command_processor_manager.get_command_processor.return_value = None
        result = await genie_instance.run_command("test", processor_id="non_existent_proc")
        assert result == {"error": "CommandProcessor 'non_existent_proc' not found."}

    async def test_run_command_processor_returns_error(self, fully_mocked_genie: Genie):
        genie_instance = await fully_mocked_genie
        mock_processor = AsyncMock(spec=CommandProcessorPlugin)
        mock_processor.process_command = AsyncMock(return_value=CommandProcessorResponse(error="Processor error", llm_thought_process="Thought error"))
        genie_instance._command_processor_manager.get_command_processor.return_value = mock_processor
        result = await genie_instance.run_command("test")
        assert result == {"error": "Processor error", "thought_process": "Thought error"}

    async def test_run_command_processor_no_tool_id(self, fully_mocked_genie: Genie):
        genie_instance = await fully_mocked_genie
        mock_processor = AsyncMock(spec=CommandProcessorPlugin)
        mock_processor.process_command = AsyncMock(return_value=CommandProcessorResponse(chosen_tool_id=None, llm_thought_process="No tool needed"))
        genie_instance._command_processor_manager.get_command_processor.return_value = mock_processor
        result = await genie_instance.run_command("test")
        assert result == {"message": "No tool selected by command processor.", "thought_process": "No tool needed"}

    async def test_run_command_hitl_inactive(self, fully_mocked_genie: Genie):
        genie_instance = await fully_mocked_genie
        genie_instance._hitl_manager.is_active = False
        await genie_instance.run_command("test")
        genie_instance._hitl_manager.request_approval.assert_not_called()
        genie_instance._tool_invoker.invoke.assert_called_once()

    async def test_run_command_execute_tool_fails_internally(self, fully_mocked_genie: Genie):
        genie_instance = await fully_mocked_genie
        genie_instance._tool_invoker.invoke.side_effect = ValueError("Internal tool execution error")
        result = await genie_instance.run_command("test")
        assert "Internal tool execution error" in result["error"]
        assert isinstance(result.get("raw_exception"), ValueError)

    async def test_run_command_unexpected_error(self, fully_mocked_genie: Genie):
        genie_instance = await fully_mocked_genie
        genie_instance._config.default_command_processor_id = "llm_assisted_cmd_proc"
        genie_instance._command_processor_manager.get_command_processor.side_effect = TypeError("Unexpected manager error")
        result = await genie_instance.run_command("test")
        assert "Unexpected error in run_command: Unexpected manager error" in result["error"]
        assert isinstance(result["raw_exception"], TypeError)


@pytest.mark.asyncio
class TestGenieClose:
    async def test_close_tears_down_managers_and_plugins(self, fully_mocked_genie: Genie):
        genie_instance = await fully_mocked_genie
        managers_on_genie = [
            genie_instance._log_adapter, genie_instance._tracing_manager, genie_instance._hitl_manager,
            genie_instance._token_usage_manager, genie_instance._guardrail_manager,
            genie_instance._prompt_manager, genie_instance._conversation_manager,
            genie_instance._llm_output_parser_manager, genie_instance._task_queue_manager,
            genie_instance._llm_provider_manager, genie_instance._command_processor_manager,
            genie_instance._rag_manager, genie_instance._tool_lookup_service,
            genie_instance._tool_invoker, genie_instance._tool_manager
        ]
        plugin_manager_mock = genie_instance._plugin_manager

        await genie_instance.close()

        for manager_mock in managers_on_genie:
            if manager_mock and hasattr(manager_mock, "teardown"):
                manager_mock.teardown.assert_awaited_once()

        if plugin_manager_mock:
            plugin_manager_mock.teardown_all_plugins.assert_awaited_once()


    async def test_close_manager_teardown_error(self, fully_mocked_genie: Genie, caplog):
        genie_instance = await fully_mocked_genie
        caplog.set_level(logging.ERROR)

        tool_manager_mock_instance = genie_instance._tool_manager
        if tool_manager_mock_instance:
            tool_manager_mock_instance.teardown.side_effect = RuntimeError("ToolManager teardown failed")

        rag_manager_mock_instance = genie_instance._rag_manager

        await genie_instance.close()
        assert "Error tearing down manager AsyncMock: ToolManager teardown failed" in caplog.text
        if rag_manager_mock_instance:
            rag_manager_mock_instance.teardown.assert_awaited_once()


    async def test_close_attributes_nulled(self, fully_mocked_genie: Genie):
        genie_instance = await fully_mocked_genie
        await genie_instance.close()
        attrs_to_null = [
            "_plugin_manager", "_key_provider", "_config", "_tool_manager", "_tool_invoker",
            "_rag_manager", "_tool_lookup_service", "_llm_provider_manager",
            "_command_processor_manager", "llm", "rag", "_log_adapter", "_tracing_manager",
            "_hitl_manager", "_token_usage_manager", "_guardrail_manager", "observability",
            "human_in_loop", "usage", "_prompt_manager", "prompts", "_conversation_manager",
            "conversation", "_llm_output_parser_manager", "_task_queue_manager", "task_queue"
        ]
        for attr_name in attrs_to_null:
            assert getattr(genie_instance, attr_name, "NOT_NULLED") is None, f"Attribute {attr_name} was not nulled."


@pytest.mark.asyncio
async def test_genie_execute_tool(fully_mocked_genie: Genie):
    genie_instance = await fully_mocked_genie
    await genie_instance.execute_tool("some_tool", param="value")
    genie_instance._tool_invoker.invoke.assert_awaited_once_with(
        tool_identifier="some_tool",
        params={"param": "value"},
        key_provider=genie_instance._key_provider,
        invoker_config=ANY
    )

@pytest.mark.asyncio
async def test_genie_register_tool_functions_no_tool_lookup_service(fully_mocked_genie: Genie, caplog):
    genie_instance = await fully_mocked_genie
    genie_instance._tool_lookup_service = None
    caplog.set_level(logging.INFO)
    mock_func = MagicMock(__name__="test_func_no_lookup")
    mock_func._tool_metadata_ = {"identifier": "test_func_no_lookup"}
    mock_func._original_function_ = lambda: "test"
    await genie_instance.register_tool_functions([mock_func])
    assert "Genie: Invalidated tool lookup index." not in caplog.text
