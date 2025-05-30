from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.resolver import PLUGIN_ID_ALIASES, ConfigResolver
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import Plugin as CorePluginType
from genie_tooling.core.types import RetrievedChunk
from genie_tooling.genie import Genie, LLMInterface, RAGInterface
from genie_tooling.invocation.invoker import ToolInvoker
from genie_tooling.llm_providers.abc import LLMProviderPlugin
from genie_tooling.lookup.service import ToolLookupService
from genie_tooling.rag.manager import RAGManager
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.manager import ToolManager

try:
    from genie_tooling.llm_providers.manager import LLMProviderManager
    from genie_tooling.llm_providers.types import (
        ChatMessage,
        LLMChatResponse,
        LLMCompletionResponse,
    )
except ImportError:
    LLMProviderManager = type("LLMProviderManager", (), {}) # type: ignore
    ChatMessage = Dict # type: ignore
    LLMChatResponse = Any # type: ignore
    LLMCompletionResponse = Any # type: ignore

try:
    from genie_tooling.command_processors.manager import CommandProcessorManager
    from genie_tooling.command_processors.types import CommandProcessorResponse
except ImportError:
    CommandProcessorManager = type("CommandProcessorManager", (), {}) # type: ignore
    CommandProcessorResponse = Any # type: ignore

class MockKeyProviderForGenie(KeyProvider, CorePluginType):
    _plugin_id_value: str; _description_value: str
    def __init__(self, keys: Dict[str, Any] = None, plugin_id="mock_genie_key_provider_v1"):
        self._plugin_id_value = plugin_id
        self._description_value = "Mock KeyProvider for Genie tests."
        self._keys = keys or {}; self.setup_called_with_config = None; self.teardown_called = False
    @property
    def plugin_id(self) -> str: return self._plugin_id_value
    @property
    def description(self) -> str: return self._description_value
    async def get_key(self, key_name: str) -> Optional[str]: return self._keys.get(key_name)
    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: self.setup_called_with_config = config
    async def teardown(self) -> None: self.teardown_called = True

@pytest.fixture
def mock_middleware_config_facade() -> MiddlewareConfig:
    return MiddlewareConfig()

@pytest.fixture
async def mock_key_provider_instance_facade() -> MockKeyProviderForGenie:
    provider = MockKeyProviderForGenie({"TEST_KEY": "test_value"})
    await provider.setup()
    return provider

@pytest.fixture
def mock_genie_dependencies(mocker):
    deps = {
        "PluginManager_cls": mocker.patch("genie_tooling.genie.PluginManager"),
        "ToolManager_cls": mocker.patch("genie_tooling.genie.ToolManager"),
        "ToolInvoker_cls": mocker.patch("genie_tooling.genie.ToolInvoker"),
        "RAGManager_cls": mocker.patch("genie_tooling.genie.RAGManager"),
        "ToolLookupService_cls": mocker.patch("genie_tooling.genie.ToolLookupService"),
        "LLMProviderManager_cls": mocker.patch("genie_tooling.genie.LLMProviderManager"),
        "CommandProcessorManager_cls": mocker.patch("genie_tooling.genie.CommandProcessorManager"),
        "ConfigResolver_cls": mocker.patch("genie_tooling.genie.ConfigResolver"),
    }
    deps["pm_inst"] = AsyncMock(spec=PluginManager)
    deps["pm_inst"]._plugin_instances = {}
    deps["pm_inst"]._discovered_plugin_classes = {}
    deps["pm_inst"].list_discovered_plugin_classes = MagicMock(return_value=deps["pm_inst"]._discovered_plugin_classes)
    deps["pm_inst"].get_plugin_instance = AsyncMock(return_value=None)
    deps["pm_inst"].discover_plugins = AsyncMock()
    deps["pm_inst"].teardown_all_plugins = AsyncMock()

    deps["tm_inst"] = AsyncMock(spec=ToolManager); deps["ToolManager_cls"].return_value = deps["tm_inst"]
    deps["ti_inst"] = AsyncMock(spec=ToolInvoker); deps["ToolInvoker_cls"].return_value = deps["ti_inst"]
    deps["ragm_inst"] = AsyncMock(spec=RAGManager); deps["ragm_inst"].retrieve_from_query = AsyncMock(); deps["RAGManager_cls"].return_value = deps["ragm_inst"]
    deps["tls_inst"] = AsyncMock(spec=ToolLookupService); deps["ToolLookupService_cls"].return_value = deps["tls_inst"]

    deps["llmpm_inst"] = AsyncMock(spec=LLMProviderManager); deps["llmpm_inst"].get_llm_provider = AsyncMock()
    deps["LLMProviderManager_cls"].return_value = deps["llmpm_inst"]

    deps["cpm_inst"] = AsyncMock(spec=CommandProcessorManager); deps["CommandProcessorManager_cls"].return_value = deps["cpm_inst"]

    deps["resolver_inst"] = MagicMock(spec=ConfigResolver)
    def generic_resolver_side_effect(user_cfg, key_provider_instance=None):
        resolved = user_cfg.model_copy(deep=True)
        if key_provider_instance and hasattr(key_provider_instance, "plugin_id"):
            resolved.key_provider_id = key_provider_instance.plugin_id
        elif resolved.key_provider_id is None:
            resolved.key_provider_id = PLUGIN_ID_ALIASES["env_keys"]

        current_features = getattr(resolved, "features", FeatureSettings())
        if resolved.default_llm_provider_id is None and current_features.llm != "none":
             llm_alias_id = PLUGIN_ID_ALIASES.get(current_features.llm)
             if llm_alias_id: resolved.default_llm_provider_id = llm_alias_id

        if resolved.default_llm_provider_id is None: resolved.default_llm_provider_id = "test_llm"
        if resolved.default_command_processor_id is None: resolved.default_command_processor_id = "test_proc"
        if resolved.default_rag_retriever_id is None: resolved.default_rag_retriever_id = "test_retriever"
        return resolved
    deps["resolver_inst"].resolve.side_effect = generic_resolver_side_effect
    deps["ConfigResolver_cls"].return_value = deps["resolver_inst"]
    return deps

@pytest.fixture
async def fully_mocked_genie(
    mock_genie_dependencies,
    mock_middleware_config_facade: MiddlewareConfig,
    mock_key_provider_instance_facade: MockKeyProviderForGenie
) -> Genie:
    resolved_kp_instance = await mock_key_provider_instance_facade

    temp_pm_mock = AsyncMock(spec=PluginManager)
    temp_pm_mock._plugin_instances = {}
    temp_pm_mock._discovered_plugin_classes = {}
    temp_pm_mock.discover_plugins = AsyncMock()
    async def temp_pm_get_instance_side_effect(plugin_id, config=None, **kwargs):
        if plugin_id == resolved_kp_instance.plugin_id:
            temp_pm_mock._plugin_instances[plugin_id] = resolved_kp_instance # type: ignore
            return resolved_kp_instance
        if plugin_id == PLUGIN_ID_ALIASES["env_keys"]:
            env_kp = MockKeyProviderForGenie(plugin_id=PLUGIN_ID_ALIASES["env_keys"])
            await env_kp.setup(config)
            temp_pm_mock._plugin_instances[plugin_id] = env_kp # type: ignore
            return env_kp
        return AsyncMock(name=f"temp_pm_mock_plugin_for_{plugin_id}")
    temp_pm_mock.get_plugin_instance.side_effect = temp_pm_get_instance_side_effect
    temp_pm_mock.list_discovered_plugin_classes = MagicMock(return_value=temp_pm_mock._discovered_plugin_classes)

    main_pm_mock = mock_genie_dependencies["pm_inst"]

    llm_provider_for_tests = AsyncMock(spec=LLMProviderPlugin)
    llm_provider_for_tests.chat = AsyncMock(return_value={"message": {"role":"assistant", "content":"Hi there!"}, "finish_reason":"stop"})
    llm_provider_for_tests.generate = AsyncMock(return_value={"text": "Generated text", "finish_reason":"stop"})

    async def main_pm_get_instance_side_effect(plugin_id, config=None, **kwargs):
        if plugin_id in main_pm_mock._plugin_instances: # type: ignore
            return main_pm_mock._plugin_instances[plugin_id] # type: ignore

        resolver_to_use_in_fixture = mock_genie_dependencies["resolver_inst"]
        config_genie_will_use = resolver_to_use_in_fixture.resolve(mock_middleware_config_facade, resolved_kp_instance)
        default_llm_id_from_resolved_config = config_genie_will_use.default_llm_provider_id or "test_llm"

        if plugin_id == default_llm_id_from_resolved_config:
            await llm_provider_for_tests.setup(config)
            main_pm_mock._plugin_instances[plugin_id] = llm_provider_for_tests # type: ignore
            return llm_provider_for_tests

        generic_mock_plugin = AsyncMock(name=f"main_pm_mock_plugin_for_{plugin_id}")
        if hasattr(generic_mock_plugin, "setup"):
             await generic_mock_plugin.setup(config)
        main_pm_mock._plugin_instances[plugin_id] = generic_mock_plugin # type: ignore
        return generic_mock_plugin
    main_pm_mock.get_plugin_instance.side_effect = main_pm_get_instance_side_effect

    mock_genie_dependencies["PluginManager_cls"].side_effect = [temp_pm_mock, main_pm_mock]

    mock_genie_dependencies["llmpm_inst"].get_llm_provider.return_value = llm_provider_for_tests

    genie_instance = await Genie.create(
        config=mock_middleware_config_facade,
        key_provider_instance=resolved_kp_instance
    )
    genie_instance.llm = LLMInterface(mock_genie_dependencies["llmpm_inst"], genie_instance._config.default_llm_provider_id)
    genie_instance.rag = RAGInterface(mock_genie_dependencies["ragm_inst"], genie_instance._config, genie_instance._key_provider)
    return genie_instance

@pytest.mark.asyncio
class TestGenieCreate:
    async def test_create_uses_config_resolver(self, mock_genie_dependencies, mock_middleware_config_facade: MiddlewareConfig, mock_key_provider_instance_facade: MockKeyProviderForGenie):
        resolver_mock = mock_genie_dependencies["resolver_inst"]
        resolver_mock.resolve.side_effect = None # Clear generic side effect for this test

        local_temp_pm_mock = AsyncMock(spec=PluginManager)
        local_temp_pm_mock._plugin_instances = {}
        local_temp_pm_mock._discovered_plugin_classes = {}
        local_temp_pm_mock.discover_plugins = AsyncMock()
        local_temp_pm_mock.list_discovered_plugin_classes = MagicMock(return_value=local_temp_pm_mock._discovered_plugin_classes)

        resolved_kp = await mock_key_provider_instance_facade
        async def temp_pm_get_instance_side_effect(plugin_id, config=None, **kwargs):
            if plugin_id == resolved_kp.plugin_id:
                local_temp_pm_mock._plugin_instances[plugin_id] = resolved_kp # type: ignore
                return resolved_kp
            return AsyncMock(name=f"temp_pm_mock_plugin_for_{plugin_id}_in_test_create_uses_resolver")
        local_temp_pm_mock.get_plugin_instance.side_effect = temp_pm_get_instance_side_effect

        main_pm_mock_for_test = mock_genie_dependencies["pm_inst"]
        main_pm_mock_for_test._plugin_instances.clear()
        main_pm_mock_for_test._discovered_plugin_classes.clear()

        mock_genie_dependencies["PluginManager_cls"].side_effect = [local_temp_pm_mock, main_pm_mock_for_test]

        expected_resolved_config_from_resolver = mock_middleware_config_facade.model_copy(deep=True)
        expected_resolved_config_from_resolver.key_provider_id = resolved_kp.plugin_id
        expected_resolved_config_from_resolver.default_llm_provider_id = "resolved_llm_id_from_test_specific"
        # Ensure features align with the expectation for default_llm_provider_id
        expected_resolved_config_from_resolver.features.llm = "openai" # Assuming 'openai' maps to a provider

        resolver_mock.resolve.return_value = expected_resolved_config_from_resolver

        genie = await Genie.create(config=mock_middleware_config_facade, key_provider_instance=resolved_kp)

        resolver_mock.resolve.assert_called_once_with(mock_middleware_config_facade, key_provider_instance=resolved_kp)

        mock_genie_dependencies["LLMProviderManager_cls"].assert_called_once_with(
            main_pm_mock_for_test, resolved_kp, expected_resolved_config_from_resolver
        )
        assert genie._config is expected_resolved_config_from_resolver

    async def test_create_with_default_key_provider_id(self, mock_genie_dependencies, mock_middleware_config_facade: MiddlewareConfig):
        pm_main_mock = mock_genie_dependencies["pm_inst"]
        pm_main_mock._plugin_instances.clear()
        pm_main_mock._discovered_plugin_classes.clear()
        resolver_mock = mock_genie_dependencies["resolver_inst"]
        resolver_mock.resolve.side_effect = None # Clear generic side effect

        config_after_resolve_for_test = mock_middleware_config_facade.model_copy(deep=True)
        config_after_resolve_for_test.key_provider_id = PLUGIN_ID_ALIASES["env_keys"]
        # If a default LLM is expected, ensure features align
        # config_after_resolve_for_test.features.llm = "ollama"
        # config_after_resolve_for_test.default_llm_provider_id = PLUGIN_ID_ALIASES["ollama"]

        resolver_mock.resolve.return_value = config_after_resolve_for_test

        mock_env_kp = MockKeyProviderForGenie(plugin_id=PLUGIN_ID_ALIASES["env_keys"])
        await mock_env_kp.setup()

        temp_pm_mock_for_kp_load = AsyncMock(spec=PluginManager)
        temp_pm_mock_for_kp_load._plugin_instances = {}
        temp_pm_mock_for_kp_load._discovered_plugin_classes = {}
        temp_pm_mock_for_kp_load.list_discovered_plugin_classes = MagicMock(return_value=temp_pm_mock_for_kp_load._discovered_plugin_classes)
        async def get_plugin_for_temp_pm(plugin_id, config=None, **kwargs):
            if plugin_id == PLUGIN_ID_ALIASES["env_keys"]:
                temp_pm_mock_for_kp_load._plugin_instances[plugin_id] = mock_env_kp # type: ignore
                return mock_env_kp
            return AsyncMock(name=f"other_plugin_temp_{plugin_id}")
        temp_pm_mock_for_kp_load.get_plugin_instance.side_effect = get_plugin_for_temp_pm
        temp_pm_mock_for_kp_load.discover_plugins = AsyncMock()

        mock_genie_dependencies["PluginManager_cls"].side_effect = [temp_pm_mock_for_kp_load, pm_main_mock]

        genie = await Genie.create(config=mock_middleware_config_facade)

        resolver_mock.resolve.assert_called_once_with(mock_middleware_config_facade, key_provider_instance=mock_env_kp)
        temp_pm_mock_for_kp_load.get_plugin_instance.assert_any_call(PLUGIN_ID_ALIASES["env_keys"])
        assert genie._key_provider is mock_env_kp

        mock_genie_dependencies["LLMProviderManager_cls"].assert_called_once_with(
            pm_main_mock, mock_env_kp, config_after_resolve_for_test
        )

@pytest.mark.asyncio
async def test_genie_execute_tool(fully_mocked_genie: Genie):
    actual_genie = await fully_mocked_genie # Await the fixture parameter
    tool_id = "calculator"; params = {"num1": 1, "num2": 2, "operation": "add"}
    await actual_genie.execute_tool(tool_id, **params)
    actual_genie._tool_invoker.invoke.assert_awaited_once_with(tool_identifier=tool_id, params=params, key_provider=actual_genie._key_provider)

@pytest.mark.asyncio
async def test_genie_run_command_selects_tool_and_executes(fully_mocked_genie: Genie):
    actual_genie = await fully_mocked_genie
    command = "Calculate 2 plus 3"
    processor_id = actual_genie._config.default_command_processor_id
    mock_processor_plugin = AsyncMock(); cmd_proc_response_success: CommandProcessorResponse = {"chosen_tool_id": "calculator_tool", "extracted_params": {"num1": 2, "num2": 3, "operation": "add"}, "llm_thought_process": "Understood calculation.", "error": None}; mock_processor_plugin.process_command = AsyncMock(return_value=cmd_proc_response_success)
    actual_genie._command_processor_manager.get_command_processor.return_value = mock_processor_plugin
    actual_genie.execute_tool = AsyncMock(return_value={"result": 5})
    result = await actual_genie.run_command(command)
    actual_genie._command_processor_manager.get_command_processor.assert_awaited_once_with(processor_id, genie_facade=actual_genie)
    mock_processor_plugin.process_command.assert_awaited_once_with(command, None)
    actual_genie.execute_tool.assert_awaited_once_with("calculator_tool", num1=2, num2=3, operation="add")
    assert result["tool_result"] == {"result": 5}; assert result["thought_process"] == "Understood calculation."

@pytest.mark.asyncio
async def test_genie_run_command_no_tool_selected(fully_mocked_genie: Genie):
    actual_genie = await fully_mocked_genie
    processor_id = actual_genie._config.default_command_processor_id
    command = "Just chatting"; cmd_proc_response_no_tool: CommandProcessorResponse = {"chosen_tool_id": None, "extracted_params": None, "llm_thought_process": "No specific tool needed for this.", "error": None}; mock_processor_plugin = AsyncMock(); mock_processor_plugin.process_command = AsyncMock(return_value=cmd_proc_response_no_tool)
    actual_genie._command_processor_manager.get_command_processor.return_value = mock_processor_plugin; actual_genie.execute_tool = AsyncMock()
    result = await actual_genie.run_command(command)
    actual_genie.execute_tool.assert_not_awaited()
    assert result["message"] == "No tool selected by command processor."; assert "tool_result" not in result

@pytest.mark.asyncio
async def test_genie_llm_chat(fully_mocked_genie: Genie):
    actual_genie = await fully_mocked_genie
    llm_provider_id_to_use = actual_genie._config.default_llm_provider_id
    messages: List[ChatMessage] = [{"role": "user", "content": "Hello LLM"}]
    response = await actual_genie.llm.chat(messages)
    actual_genie._llm_provider_manager.get_llm_provider.assert_awaited_once_with(llm_provider_id_to_use)
    llm_plugin_instance_used_in_chat = actual_genie._llm_provider_manager.get_llm_provider.return_value
    llm_plugin_instance_used_in_chat.chat.assert_awaited_once_with(messages)
    assert response["message"]["content"] == "Hi there!"

@pytest.mark.asyncio
async def test_genie_rag_search(fully_mocked_genie: Genie):
    actual_genie = await fully_mocked_genie
    retriever_id_to_use = actual_genie._config.default_rag_retriever_id
    query = "What is RAG?"; collection_name = "test_collection"; mock_retrieved_chunks: List[RetrievedChunk] = [{"id":"c1", "content":"RAG content", "score":0.9, "metadata":{}}] # type: ignore
    actual_genie._rag_manager.retrieve_from_query.return_value = mock_retrieved_chunks
    results = await actual_genie.rag.search(query, collection_name=collection_name, top_k=1)
    actual_genie._rag_manager.retrieve_from_query.assert_awaited_once()
    call_kwargs = actual_genie._rag_manager.retrieve_from_query.call_args.kwargs
    assert call_kwargs["query_text"] == query
    assert call_kwargs["retriever_id"] == retriever_id_to_use
    assert call_kwargs["top_k"] == 1
    assert call_kwargs["retriever_config"]["vector_store_config"]["collection_name"] == collection_name
    assert call_kwargs["retriever_config"]["embedder_config"]["key_provider"] == actual_genie._key_provider
    assert results == mock_retrieved_chunks

@pytest.mark.asyncio
async def test_genie_close(fully_mocked_genie: Genie):
    actual_genie = await fully_mocked_genie
    pm_mock_for_assert = actual_genie._plugin_manager; llmpm_mock_for_assert = actual_genie._llm_provider_manager; cpm_mock_for_assert = actual_genie._command_processor_manager
    assert pm_mock_for_assert is not None; assert llmpm_mock_for_assert is not None; assert cpm_mock_for_assert is not None
    await actual_genie.close()
    pm_mock_for_assert.teardown_all_plugins.assert_awaited_once(); llmpm_mock_for_assert.teardown.assert_awaited_once(); cpm_mock_for_assert.teardown.assert_awaited_once()
    assert actual_genie._plugin_manager is None; assert actual_genie._llm_provider_manager is None; assert actual_genie._command_processor_manager is None; assert actual_genie.llm is None; assert actual_genie.rag is None
