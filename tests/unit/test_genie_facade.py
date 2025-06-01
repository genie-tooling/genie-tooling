### tests/unit/test_genie_facade.py
import logging
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.resolver import PLUGIN_ID_ALIASES, ConfigResolver
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import Plugin as CorePluginType
from genie_tooling.core.types import RetrievedChunk
from genie_tooling.decorators import tool
from genie_tooling.genie import FunctionToolWrapper, Genie
from genie_tooling.guardrails.manager import GuardrailManager
from genie_tooling.hitl.manager import HITLManager
from genie_tooling.hitl.types import ApprovalResponse  # Keep for mocking

# Updated interface imports
from genie_tooling.interfaces import (
    ConversationInterface,
    HITLInterface,
    LLMInterface,
    ObservabilityInterface,
    PromptInterface,
    RAGInterface,
    UsageTrackingInterface,
)
from genie_tooling.invocation.invoker import ToolInvoker
from genie_tooling.llm_providers.abc import LLMProviderPlugin
from genie_tooling.lookup.service import ToolLookupService

# P1.5 Manager Imports for mocking
from genie_tooling.observability.manager import InteractionTracingManager
from genie_tooling.prompts.conversation.impl.manager import ConversationStateManager
from genie_tooling.prompts.llm_output_parsers.manager import LLMOutputParserManager
from genie_tooling.prompts.manager import PromptManager
from genie_tooling.rag.manager import RAGManager
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.token_usage.manager import TokenUsageManager
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
    _plugin_id_value: str
    _description_value: str
    def __init__(self, keys: Dict[str, Any] = None, plugin_id="mock_genie_key_provider_v1"):
        self._plugin_id_value = plugin_id
        self._description_value = "Mock KeyProvider for Genie tests."
        self._keys = keys or {}
        self.setup_called_with_config = None
        self.teardown_called = False
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
        # P1.5 Manager Mocks
        "InteractionTracingManager_cls": mocker.patch("genie_tooling.genie.InteractionTracingManager"),
        "HITLManager_cls": mocker.patch("genie_tooling.genie.HITLManager"),
        "TokenUsageManager_cls": mocker.patch("genie_tooling.genie.TokenUsageManager"),
        "GuardrailManager_cls": mocker.patch("genie_tooling.genie.GuardrailManager"),
        "PromptManager_cls": mocker.patch("genie_tooling.genie.PromptManager"),
        "ConversationStateManager_cls": mocker.patch("genie_tooling.genie.ConversationStateManager"),
        "LLMOutputParserManager_cls": mocker.patch("genie_tooling.genie.LLMOutputParserManager"),
    }
    deps["pm_inst"] = AsyncMock(spec=PluginManager)
    deps["pm_inst"]._plugin_instances = {}
    deps["pm_inst"]._discovered_plugin_classes = {}
    deps["pm_inst"].list_discovered_plugin_classes = MagicMock(return_value=deps["pm_inst"]._discovered_plugin_classes)
    deps["pm_inst"].get_plugin_instance = AsyncMock(return_value=None)
    deps["pm_inst"].discover_plugins = AsyncMock()
    deps["pm_inst"].teardown_all_plugins = AsyncMock()

    deps["tm_inst"] = AsyncMock(spec=ToolManager)
    deps["tm_inst"]._tools = {}
    deps["ToolManager_cls"].return_value = deps["tm_inst"]

    deps["ti_inst"] = AsyncMock(spec=ToolInvoker)
    deps["ToolInvoker_cls"].return_value = deps["ti_inst"]
    deps["ragm_inst"] = AsyncMock(spec=RAGManager)
    deps["ragm_inst"].retrieve_from_query = AsyncMock()
    deps["RAGManager_cls"].return_value = deps["ragm_inst"]
    deps["tls_inst"] = AsyncMock(spec=ToolLookupService)
    deps["ToolLookupService_cls"].return_value = deps["tls_inst"]

    deps["llmpm_inst"] = AsyncMock(spec=LLMProviderManager)
    deps["llmpm_inst"].get_llm_provider = AsyncMock()
    deps["LLMProviderManager_cls"].return_value = deps["llmpm_inst"]

    deps["cpm_inst"] = AsyncMock(spec=CommandProcessorManager)
    deps["CommandProcessorManager_cls"].return_value = deps["cpm_inst"]

    # P1.5 Manager Instances
    deps["tracing_m_inst"] = AsyncMock(spec=InteractionTracingManager)
    deps["InteractionTracingManager_cls"].return_value = deps["tracing_m_inst"]
    deps["hitl_m_inst"] = AsyncMock(spec=HITLManager)
    deps["HITLManager_cls"].return_value = deps["hitl_m_inst"]
    deps["token_usage_m_inst"] = AsyncMock(spec=TokenUsageManager)
    deps["TokenUsageManager_cls"].return_value = deps["token_usage_m_inst"]
    deps["guardrail_m_inst"] = AsyncMock(spec=GuardrailManager)
    deps["GuardrailManager_cls"].return_value = deps["guardrail_m_inst"]
    deps["prompt_m_inst"] = AsyncMock(spec=PromptManager)
    deps["PromptManager_cls"].return_value = deps["prompt_m_inst"]
    deps["convo_m_inst"] = AsyncMock(spec=ConversationStateManager)
    deps["ConversationStateManager_cls"].return_value = deps["convo_m_inst"]
    deps["llm_output_parser_m_inst"] = AsyncMock(spec=LLMOutputParserManager)
    deps["LLMOutputParserManager_cls"].return_value = deps["llm_output_parser_m_inst"]


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

        if resolved.default_rag_embedder_id is None and current_features.rag_embedder != "none":
            embed_alias = {"sentence_transformer": "st_embedder", "openai": "openai_embedder"}.get(current_features.rag_embedder)
            if embed_alias and PLUGIN_ID_ALIASES.get(embed_alias):
                resolved.default_rag_embedder_id = PLUGIN_ID_ALIASES[embed_alias]

        # Fallbacks for P1.5 defaults if not set by features
        if resolved.default_observability_tracer_id is None: resolved.default_observability_tracer_id = "test_tracer"
        if resolved.default_hitl_approver_id is None: resolved.default_hitl_approver_id = "test_hitl_approver"
        if resolved.default_token_usage_recorder_id is None: resolved.default_token_usage_recorder_id = "test_token_recorder"
        if resolved.default_prompt_registry_id is None: resolved.default_prompt_registry_id = "test_prompt_registry"
        if resolved.default_prompt_template_plugin_id is None: resolved.default_prompt_template_plugin_id = "test_prompt_template_engine"
        if resolved.default_conversation_state_provider_id is None: resolved.default_conversation_state_provider_id = "test_convo_provider"
        if resolved.default_llm_output_parser_id is None: resolved.default_llm_output_parser_id = "test_output_parser"


        if resolved.default_llm_provider_id is None: resolved.default_llm_provider_id = "test_llm"
        if resolved.default_command_processor_id is None: resolved.default_command_processor_id = "test_proc"
        if resolved.default_rag_retriever_id is None: resolved.default_rag_retriever_id = "test_retriever"
        if resolved.default_rag_embedder_id is None: resolved.default_rag_embedder_id = "test_embedder"


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
    main_pm_mock._plugin_instances = {}
    main_pm_mock._discovered_plugin_classes = {}


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
    # Re-initialize interfaces with potentially updated managers from Genie.create
    # This is crucial because the interfaces are instantiated with managers during Genie.create
    genie_instance.llm = LLMInterface(
        genie_instance._llm_provider_manager, # type: ignore
        genie_instance._config.default_llm_provider_id,
        genie_instance._llm_output_parser_manager, # type: ignore # Added
        genie_instance._tracing_manager, # type: ignore
        genie_instance._guardrail_manager, # type: ignore
        genie_instance._token_usage_manager # type: ignore
    )
    genie_instance.rag = RAGInterface(
        genie_instance._rag_manager, # type: ignore
        genie_instance._config,
        genie_instance._key_provider, # type: ignore
        genie_instance._tracing_manager # type: ignore
    )
    # P1.5 Interfaces
    genie_instance.observability = ObservabilityInterface(genie_instance._tracing_manager) # type: ignore
    genie_instance.human_in_loop = HITLInterface(genie_instance._hitl_manager) # type: ignore
    genie_instance.usage = UsageTrackingInterface(genie_instance._token_usage_manager) # type: ignore
    genie_instance.prompts = PromptInterface(genie_instance._prompt_manager) # type: ignore
    genie_instance.conversation = ConversationInterface(genie_instance._conversation_manager) # type: ignore


    # Mock HITL for tests that might trigger it
    genie_instance.human_in_loop.request_approval = AsyncMock( # type: ignore
        return_value=ApprovalResponse(request_id="mock_req_id", status="approved", approver_id="test_approver", reason=None, timestamp=0.0)
    )


    if not hasattr(genie_instance._tool_manager, "_tools"): # type: ignore
        genie_instance._tool_manager._tools = {} # type: ignore

    return genie_instance


@pytest.mark.asyncio
class TestGenieCreate:
    async def test_create_uses_config_resolver(self, mock_genie_dependencies, mock_middleware_config_facade: MiddlewareConfig, mock_key_provider_instance_facade: MockKeyProviderForGenie):
        resolver_mock = mock_genie_dependencies["resolver_inst"]
        resolver_mock.resolve.side_effect = None # Disable generic side effect for this test

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
        expected_resolved_config_from_resolver.features.llm = "openai"
        # Manually set P1.5 default IDs for this specific test config
        expected_resolved_config_from_resolver.default_prompt_registry_id = "test_registry"
        expected_resolved_config_from_resolver.default_prompt_template_plugin_id = "test_template"
        expected_resolved_config_from_resolver.default_conversation_state_provider_id = "test_convo"
        expected_resolved_config_from_resolver.default_llm_output_parser_id = "test_parser"


        resolver_mock.resolve.return_value = expected_resolved_config_from_resolver

        genie = await Genie.create(config=mock_middleware_config_facade, key_provider_instance=resolved_kp)

        resolver_mock.resolve.assert_called_once_with(mock_middleware_config_facade, key_provider_instance=resolved_kp)

        mock_genie_dependencies["LLMProviderManager_cls"].assert_called_once_with(
            main_pm_mock_for_test,
            resolved_kp,
            expected_resolved_config_from_resolver,
            mock_genie_dependencies["token_usage_m_inst"]
        )
        assert genie._config is expected_resolved_config_from_resolver

    async def test_create_with_default_key_provider_id(self, mock_genie_dependencies, mock_middleware_config_facade: MiddlewareConfig):
        pm_main_mock = mock_genie_dependencies["pm_inst"]
        pm_main_mock._plugin_instances.clear()
        pm_main_mock._discovered_plugin_classes.clear()
        resolver_mock = mock_genie_dependencies["resolver_inst"]
        resolver_mock.resolve.side_effect = None # Disable generic side effect

        config_after_resolve_for_test = mock_middleware_config_facade.model_copy(deep=True)
        config_after_resolve_for_test.key_provider_id = PLUGIN_ID_ALIASES["env_keys"]
        # Manually set P1.5 default IDs for this specific test config
        config_after_resolve_for_test.default_prompt_registry_id = "test_registry"
        config_after_resolve_for_test.default_prompt_template_plugin_id = "test_template"
        config_after_resolve_for_test.default_conversation_state_provider_id = "test_convo"
        config_after_resolve_for_test.default_llm_output_parser_id = "test_parser"


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
            pm_main_mock,
            mock_env_kp,
            config_after_resolve_for_test,
            mock_genie_dependencies["token_usage_m_inst"]
        )

    async def test_create_key_provider_id_from_config_no_instance(self, mock_genie_dependencies, mock_middleware_config_facade: MiddlewareConfig):
        mock_middleware_config_facade.key_provider_id = "user_specified_kp_id_v1"

        pm_main_mock = mock_genie_dependencies["pm_inst"]
        resolver_mock = mock_genie_dependencies["resolver_inst"]
        resolver_mock.resolve.side_effect = None

        config_after_resolve = mock_middleware_config_facade.model_copy(deep=True)
        config_after_resolve.default_prompt_registry_id = "test_registry"
        config_after_resolve.default_prompt_template_plugin_id = "test_template"
        config_after_resolve.default_conversation_state_provider_id = "test_convo"
        config_after_resolve.default_llm_output_parser_id = "test_parser"

        resolver_mock.resolve.return_value = config_after_resolve

        mock_user_kp = MockKeyProviderForGenie(plugin_id="user_specified_kp_id_v1")
        await mock_user_kp.setup()

        temp_pm_mock = AsyncMock(spec=PluginManager)
        temp_pm_mock._plugin_instances = {}
        temp_pm_mock._discovered_plugin_classes = {}
        temp_pm_mock.get_plugin_instance.return_value = mock_user_kp
        temp_pm_mock.discover_plugins = AsyncMock()
        temp_pm_mock.list_discovered_plugin_classes = MagicMock(return_value={})


        mock_genie_dependencies["PluginManager_cls"].side_effect = [temp_pm_mock, pm_main_mock]

        genie = await Genie.create(config=mock_middleware_config_facade, key_provider_instance=None)

        temp_pm_mock.get_plugin_instance.assert_called_once_with("user_specified_kp_id_v1")
        assert genie._key_provider is mock_user_kp
        resolver_mock.resolve.assert_called_once_with(mock_middleware_config_facade, key_provider_instance=mock_user_kp)

    async def test_create_key_provider_load_fails(self, mock_genie_dependencies, mock_middleware_config_facade: MiddlewareConfig):
        mock_middleware_config_facade.key_provider_id = "failing_kp_id"

        temp_pm_mock = AsyncMock(spec=PluginManager)
        temp_pm_mock._plugin_instances = {}
        temp_pm_mock._discovered_plugin_classes = {}
        temp_pm_mock.get_plugin_instance.return_value = None
        temp_pm_mock.discover_plugins = AsyncMock()
        temp_pm_mock.list_discovered_plugin_classes = MagicMock(return_value={})

        mock_genie_dependencies["PluginManager_cls"].side_effect = [temp_pm_mock, mock_genie_dependencies["pm_inst"]]

        with pytest.raises(RuntimeError, match="Failed to load KeyProvider with ID 'failing_kp_id'"):
            await Genie.create(config=mock_middleware_config_facade, key_provider_instance=None)

    async def test_create_key_provider_wrong_type(self, mock_genie_dependencies, mock_middleware_config_facade: MiddlewareConfig):
        mock_middleware_config_facade.key_provider_id = "wrong_type_kp_id"

        wrong_type_plugin = AsyncMock(spec=CorePluginType)

        temp_pm_mock = AsyncMock(spec=PluginManager)
        temp_pm_mock._plugin_instances = {}
        temp_pm_mock._discovered_plugin_classes = {}
        temp_pm_mock.get_plugin_instance.return_value = wrong_type_plugin
        temp_pm_mock.discover_plugins = AsyncMock()
        temp_pm_mock.list_discovered_plugin_classes = MagicMock(return_value={})

        mock_genie_dependencies["PluginManager_cls"].side_effect = [temp_pm_mock, mock_genie_dependencies["pm_inst"]]

        with pytest.raises(RuntimeError, match="Failed to load KeyProvider with ID 'wrong_type_kp_id'"):
            await Genie.create(config=mock_middleware_config_facade, key_provider_instance=None)

    async def test_create_kp_instance_already_in_main_pm(self, mock_genie_dependencies, mock_middleware_config_facade: MiddlewareConfig, mock_key_provider_instance_facade: MockKeyProviderForGenie):
        kp_instance = await mock_key_provider_instance_facade
        kp_instance_id = kp_instance.plugin_id

        main_pm_mock = mock_genie_dependencies["pm_inst"]
        main_pm_mock._plugin_instances = {kp_instance_id: kp_instance}
        main_pm_mock._discovered_plugin_classes = {kp_instance_id: type(kp_instance)}


        temp_pm_mock = AsyncMock(spec=PluginManager)
        temp_pm_mock._plugin_instances = {}
        temp_pm_mock._discovered_plugin_classes = {}
        temp_pm_mock.discover_plugins = AsyncMock()

        mock_genie_dependencies["PluginManager_cls"].side_effect = [temp_pm_mock, main_pm_mock]

        genie = await Genie.create(config=mock_middleware_config_facade, key_provider_instance=kp_instance)

        assert genie._key_provider is kp_instance
        temp_pm_mock.discover_plugins.assert_called_once()
        main_pm_mock.discover_plugins.assert_called_once()
        assert main_pm_mock._plugin_instances.get(kp_instance_id) is kp_instance


@pytest.mark.asyncio
async def test_genie_execute_tool(fully_mocked_genie: Genie):
    actual_genie = await fully_mocked_genie
    tool_id = "calculator"
    params = {"num1": 1, "num2": 2, "operation": "add"}

    # This is the config passed from Genie.execute_tool to ToolInvoker.invoke
    expected_invoker_config_received_by_tool_invoker = {
        "plugin_manager": actual_genie._plugin_manager,
        "guardrail_manager": actual_genie._guardrail_manager,
        "tracing_manager": actual_genie._tracing_manager,
        "correlation_id": Any, # Will be a UUID string
    }

    await actual_genie.execute_tool(tool_id, **params)

    actual_genie._tool_invoker.invoke.assert_awaited_once() # type: ignore
    call_args = actual_genie._tool_invoker.invoke.await_args # type: ignore

    assert call_args.kwargs["tool_identifier"] == tool_id
    assert call_args.kwargs["params"] == params
    assert call_args.kwargs["key_provider"] == actual_genie._key_provider

    actual_invoker_config_arg_to_tool_invoker = call_args.kwargs["invoker_config"]
    assert isinstance(actual_invoker_config_arg_to_tool_invoker["correlation_id"], str)

    # Update expected config with the actual correlation_id for exact match
    expected_invoker_config_received_by_tool_invoker["correlation_id"] = actual_invoker_config_arg_to_tool_invoker["correlation_id"]

    assert actual_invoker_config_arg_to_tool_invoker == expected_invoker_config_received_by_tool_invoker


@pytest.mark.asyncio
async def test_genie_execute_tool_invoker_none(fully_mocked_genie: Genie):
    actual_genie = await fully_mocked_genie
    actual_genie._tool_invoker = None
    with pytest.raises(RuntimeError, match="ToolInvoker not initialized."):
        await actual_genie.execute_tool("tool_id")

@pytest.mark.asyncio
async def test_genie_execute_tool_key_provider_none(fully_mocked_genie: Genie):
    actual_genie = await fully_mocked_genie
    actual_genie._key_provider = None
    with pytest.raises(RuntimeError, match="KeyProvider not initialized."):
        await actual_genie.execute_tool("tool_id")


@pytest.mark.asyncio
async def test_genie_run_command_selects_tool_and_executes(fully_mocked_genie: Genie):
    actual_genie = await fully_mocked_genie
    command = "Calculate 2 plus 3"
    processor_id = actual_genie._config.default_command_processor_id
    mock_processor_plugin = AsyncMock()
    cmd_proc_response_success: CommandProcessorResponse = {"chosen_tool_id": "calculator_tool", "extracted_params": {"num1": 2, "num2": 3, "operation": "add"}, "llm_thought_process": "Understood calculation.", "error": None}
    mock_processor_plugin.process_command = AsyncMock(return_value=cmd_proc_response_success)
    actual_genie._command_processor_manager.get_command_processor.return_value = mock_processor_plugin # type: ignore

    actual_genie.execute_tool = AsyncMock(return_value={"result": 5}) # type: ignore

    actual_genie.human_in_loop.request_approval = AsyncMock( # type: ignore
        return_value=ApprovalResponse(request_id="mock_req", status="approved", approver_id="test", reason=None, timestamp=0.0)
    )

    result = await actual_genie.run_command(command)
    actual_genie._command_processor_manager.get_command_processor.assert_awaited_once_with(processor_id, genie_facade=actual_genie) # type: ignore
    mock_processor_plugin.process_command.assert_awaited_once_with(command, None)
    actual_genie.execute_tool.assert_awaited_once_with("calculator_tool", num1=2, num2=3, operation="add") # type: ignore
    assert result["tool_result"] == {"result": 5}
    assert result["thought_process"] == "Understood calculation."

@pytest.mark.asyncio
async def test_genie_run_command_no_tool_selected(fully_mocked_genie: Genie):
    actual_genie = await fully_mocked_genie
    processor_id = actual_genie._config.default_command_processor_id
    command = "Just chatting"
    cmd_proc_response_no_tool: CommandProcessorResponse = {"chosen_tool_id": None, "extracted_params": None, "llm_thought_process": "No specific tool needed for this.", "error": None}
    mock_processor_plugin = AsyncMock()
    mock_processor_plugin.process_command = AsyncMock(return_value=cmd_proc_response_no_tool)
    actual_genie._command_processor_manager.get_command_processor.return_value = mock_processor_plugin # type: ignore
    actual_genie.execute_tool = AsyncMock() # type: ignore
    result = await actual_genie.run_command(command)
    actual_genie.execute_tool.assert_not_awaited() # type: ignore
    assert result["message"] == "No tool selected by command processor."
    assert "tool_result" not in result

@pytest.mark.asyncio
async def test_genie_run_command_processor_manager_none(fully_mocked_genie: Genie):
    actual_genie = await fully_mocked_genie
    actual_genie._command_processor_manager = None
    result = await actual_genie.run_command("test")
    assert result["error"] == "CommandProcessorManager not initialized."

@pytest.mark.asyncio
async def test_genie_run_command_no_target_processor_id(fully_mocked_genie: Genie):
    actual_genie = await fully_mocked_genie
    actual_genie._config.default_command_processor_id = None
    result = await actual_genie.run_command("test", processor_id=None)
    assert result["error"] == "No command processor configured."

@pytest.mark.asyncio
async def test_genie_run_command_processor_not_found(fully_mocked_genie: Genie):
    actual_genie = await fully_mocked_genie
    actual_genie._command_processor_manager.get_command_processor.return_value = None # type: ignore
    result = await actual_genie.run_command("test", processor_id="non_existent_proc")
    assert result["error"] == "CommandProcessor 'non_existent_proc' not found."

@pytest.mark.asyncio
async def test_genie_run_command_processor_returns_error(fully_mocked_genie: Genie):
    actual_genie = await fully_mocked_genie
    mock_processor_plugin = AsyncMock()
    mock_processor_plugin.process_command.return_value = {"error": "Processor internal error", "llm_thought_process": "Thinking failed"}
    actual_genie._command_processor_manager.get_command_processor.return_value = mock_processor_plugin # type: ignore
    result = await actual_genie.run_command("test")
    assert result["error"] == "Processor internal error"
    assert result["thought_process"] == "Thinking failed"

@pytest.mark.asyncio
async def test_genie_run_command_processor_raises_exception(fully_mocked_genie: Genie):
    actual_genie = await fully_mocked_genie
    mock_processor_plugin = AsyncMock()
    mock_processor_plugin.process_command.side_effect = ValueError("Unexpected processor crash")
    actual_genie._command_processor_manager.get_command_processor.return_value = mock_processor_plugin # type: ignore
    result = await actual_genie.run_command("test")
    assert "Unexpected error in run_command: Unexpected processor crash" in result["error"]
    assert isinstance(result["raw_exception"], ValueError)


@pytest.mark.asyncio
async def test_genie_llm_chat(fully_mocked_genie: Genie):
    actual_genie = await fully_mocked_genie
    llm_provider_id_to_use = actual_genie._config.default_llm_provider_id
    messages: List[ChatMessage] = [{"role": "user", "content": "Hello LLM"}]
    response = await actual_genie.llm.chat(messages)
    actual_genie._llm_provider_manager.get_llm_provider.assert_awaited_once_with(llm_provider_id_to_use) # type: ignore
    llm_plugin_instance_used_in_chat = actual_genie._llm_provider_manager.get_llm_provider.return_value # type: ignore
    llm_plugin_instance_used_in_chat.chat.assert_awaited_once_with(messages, stream=False)
    assert response["message"]["content"] == "Hi there!"

@pytest.mark.asyncio
async def test_llm_interface_generate_no_provider_id(fully_mocked_genie: Genie):
    actual_genie = await fully_mocked_genie
    actual_genie.llm._default_provider_id = None
    with pytest.raises(ValueError, match="No LLM provider ID specified and no default is set for generate."):
        await actual_genie.llm.generate("prompt")

@pytest.mark.asyncio
async def test_llm_interface_chat_no_provider_id(fully_mocked_genie: Genie):
    actual_genie = await fully_mocked_genie
    actual_genie.llm._default_provider_id = None
    with pytest.raises(ValueError, match="No LLM provider ID specified and no default is set for chat."):
        await actual_genie.llm.chat([{"role":"user", "content":"hi"}])

@pytest.mark.asyncio
async def test_llm_interface_generate_provider_not_found(fully_mocked_genie: Genie):
    actual_genie = await fully_mocked_genie
    actual_genie._llm_provider_manager.get_llm_provider.return_value = None # type: ignore
    with pytest.raises(RuntimeError, match="LLM Provider 'test_llm' not found or failed to load."):
        await actual_genie.llm.generate("prompt", provider_id="test_llm")


@pytest.mark.asyncio
async def test_genie_rag_search(fully_mocked_genie: Genie):
    actual_genie = await fully_mocked_genie
    retriever_id_to_use = actual_genie._config.default_rag_retriever_id
    query = "What is RAG?"
    collection_name = "test_collection"
    mock_retrieved_chunks: List[RetrievedChunk] = [{"id":"c1", "content":"RAG content", "score":0.9, "metadata":{}}] # type: ignore
    actual_genie._rag_manager.retrieve_from_query.return_value = mock_retrieved_chunks # type: ignore
    results = await actual_genie.rag.search(query, collection_name=collection_name, top_k=1)
    actual_genie._rag_manager.retrieve_from_query.assert_awaited_once() # type: ignore
    call_kwargs = actual_genie._rag_manager.retrieve_from_query.call_args.kwargs # type: ignore
    assert call_kwargs["query_text"] == query
    assert call_kwargs["retriever_id"] == retriever_id_to_use
    assert call_kwargs["top_k"] == 1
    assert call_kwargs["retriever_config"]["vector_store_config"]["collection_name"] == collection_name
    assert call_kwargs["retriever_config"]["embedder_config"]["key_provider"] == actual_genie._key_provider
    assert results == mock_retrieved_chunks

@pytest.mark.asyncio
async def test_rag_interface_index_directory_no_embedder_id(fully_mocked_genie: Genie):
    actual_genie = await fully_mocked_genie
    assert actual_genie.rag is not None
    actual_genie.rag._config.default_rag_embedder_id = None # type: ignore
    with pytest.raises(ValueError, match=r"RAG embedder ID not resolved for index_directory\."):
        await actual_genie.rag.index_directory("path")

@pytest.mark.asyncio
async def test_rag_interface_index_web_page_no_vector_store_id(fully_mocked_genie: Genie):
    actual_genie = await fully_mocked_genie
    assert actual_genie.rag is not None
    actual_genie.rag._config.default_rag_embedder_id = "test_embedder" # type: ignore
    actual_genie.rag._config.default_rag_vector_store_id = None # type: ignore
    with pytest.raises(ValueError, match=r"RAG vector store ID not resolved for index_web_page\."):
        await actual_genie.rag.index_web_page("url")


@pytest.mark.asyncio
async def test_genie_close(fully_mocked_genie: Genie):
    actual_genie = await fully_mocked_genie
    pm_mock_for_assert = actual_genie._plugin_manager
    llmpm_mock_for_assert = actual_genie._llm_provider_manager
    cpm_mock_for_assert = actual_genie._command_processor_manager
    # P1.5 Managers for assertion
    tracing_m_assert = actual_genie._tracing_manager
    hitl_m_assert = actual_genie._hitl_manager
    token_usage_m_assert = actual_genie._token_usage_manager
    guardrail_m_assert = actual_genie._guardrail_manager
    prompt_m_assert = actual_genie._prompt_manager
    convo_m_assert = actual_genie._conversation_manager
    llm_output_parser_m_assert = actual_genie._llm_output_parser_manager


    assert pm_mock_for_assert is not None
    assert llmpm_mock_for_assert is not None
    assert cpm_mock_for_assert is not None
    assert tracing_m_assert is not None
    assert hitl_m_assert is not None
    assert token_usage_m_assert is not None
    assert guardrail_m_assert is not None
    assert prompt_m_assert is not None
    assert convo_m_assert is not None
    assert llm_output_parser_m_assert is not None


    await actual_genie.close()
    pm_mock_for_assert.teardown_all_plugins.assert_awaited_once() # type: ignore
    llmpm_mock_for_assert.teardown.assert_awaited_once() # type: ignore
    cpm_mock_for_assert.teardown.assert_awaited_once() # type: ignore

    # P1.5 Manager teardown assertions
    tracing_m_assert.teardown.assert_awaited_once() # type: ignore
    hitl_m_assert.teardown.assert_awaited_once() # type: ignore
    token_usage_m_assert.teardown.assert_awaited_once() # type: ignore
    guardrail_m_assert.teardown.assert_awaited_once() # type: ignore
    prompt_m_assert.teardown.assert_awaited_once() # type: ignore
    convo_m_assert.teardown.assert_awaited_once() # type: ignore
    llm_output_parser_m_assert.teardown.assert_awaited_once() # type: ignore

    assert actual_genie._plugin_manager is None
    assert actual_genie._llm_provider_manager is None
    assert actual_genie._command_processor_manager is None
    assert actual_genie.llm is None
    assert actual_genie.rag is None
    # P1.5 Manager nullification assertions
    assert actual_genie._tracing_manager is None
    assert actual_genie._hitl_manager is None
    assert actual_genie._token_usage_manager is None
    assert actual_genie._guardrail_manager is None
    assert actual_genie._prompt_manager is None
    assert actual_genie._conversation_manager is None
    assert actual_genie._llm_output_parser_manager is None
    assert actual_genie.observability is None
    assert actual_genie.human_in_loop is None
    assert actual_genie.usage is None
    assert actual_genie.prompts is None
    assert actual_genie.conversation is None


@pytest.mark.asyncio
async def test_genie_close_manager_teardown_fails(fully_mocked_genie: Genie, caplog):
    actual_genie = await fully_mocked_genie
    actual_genie._llm_provider_manager.teardown.side_effect = RuntimeError("LLM Manager teardown failed") # type: ignore

    with caplog.at_level(logging.ERROR):
        await actual_genie.close()

    assert "Error tearing down manager" in caplog.text
    assert ": LLM Manager teardown failed" in caplog.text
    assert actual_genie._llm_provider_manager is None


@tool
async def my_decorated_async_func_for_genie_test(param_a: str) -> str:
    """A test async function to be registered."""
    return f"Processed: {param_a}"

@tool
def my_decorated_sync_func_for_genie_test(num_b: int) -> int:
    """A test sync function to be registered."""
    return num_b * 2

@pytest.mark.asyncio
async def test_genie_register_tool_functions(fully_mocked_genie: Genie, mocker, caplog):
    caplog.set_level(logging.INFO)
    genie_instance = await fully_mocked_genie

    assert genie_instance._tool_manager is not None, "ToolManager not initialized in fully_mocked_genie"
    if not hasattr(genie_instance._tool_manager, "_tools") or not isinstance(genie_instance._tool_manager._tools, dict): # type: ignore
        genie_instance._tool_manager._tools = {} # type: ignore

    original_tools_dict_copy = dict(genie_instance._tool_manager._tools) # type: ignore

    if hasattr(genie_instance, "_tool_lookup_service") and genie_instance._tool_lookup_service:
        if isinstance(genie_instance._tool_lookup_service, AsyncMock): # type: ignore
            genie_instance._tool_lookup_service.invalidate_index = MagicMock() # type: ignore
        mock_invalidate_index = genie_instance._tool_lookup_service.invalidate_index # type: ignore
    else:
        genie_instance._tool_lookup_service = MagicMock(name="MockToolLookupServiceInTest_Sync") # type: ignore
        genie_instance._tool_lookup_service.invalidate_index = MagicMock() # type: ignore
        mock_invalidate_index = genie_instance._tool_lookup_service.invalidate_index # type: ignore


    functions_to_register: List[Callable] = [
        my_decorated_async_func_for_genie_test,
        my_decorated_sync_func_for_genie_test
    ]

    await genie_instance.register_tool_functions(functions_to_register)

    assert len(genie_instance._tool_manager._tools) == len(original_tools_dict_copy) + 2, \
        f"Expected {len(original_tools_dict_copy) + 2} tools, found {len(genie_instance._tool_manager._tools)}" # type: ignore

    async_tool_wrapper = genie_instance._tool_manager._tools.get("my_decorated_async_func_for_genie_test") # type: ignore
    assert async_tool_wrapper is not None
    assert isinstance(async_tool_wrapper, FunctionToolWrapper)
    assert async_tool_wrapper.identifier == "my_decorated_async_func_for_genie_test"

    sync_tool_wrapper = genie_instance._tool_manager._tools.get("my_decorated_sync_func_for_genie_test") # type: ignore
    assert sync_tool_wrapper is not None
    assert isinstance(sync_tool_wrapper, FunctionToolWrapper)
    assert sync_tool_wrapper.identifier == "my_decorated_sync_func_for_genie_test"

    mock_invalidate_index.assert_called_once()
    assert "Genie: Invalidated tool lookup index." in caplog.text # Updated log message

    # This is the config passed from Genie.execute_tool to ToolInvoker.invoke
    expected_invoker_config_received_by_tool_invoker_reg_test = {
        "plugin_manager": genie_instance._plugin_manager,
        "guardrail_manager": genie_instance._guardrail_manager,
        "tracing_manager": genie_instance._tracing_manager,
        "correlation_id": Any, # Will be a UUID string
    }

    await genie_instance.execute_tool("my_decorated_async_func_for_genie_test", param_a="hello")

    genie_instance._tool_invoker.invoke.assert_awaited_once() # type: ignore
    call_args_reg = genie_instance._tool_invoker.invoke.await_args # type: ignore

    assert call_args_reg.kwargs["tool_identifier"] == "my_decorated_async_func_for_genie_test"
    assert call_args_reg.kwargs["params"] == {"param_a": "hello"}
    assert call_args_reg.kwargs["key_provider"] == genie_instance._key_provider

    actual_invoker_config_arg_to_tool_invoker_reg = call_args_reg.kwargs["invoker_config"]
    assert isinstance(actual_invoker_config_arg_to_tool_invoker_reg["correlation_id"], str)

    # Update expected config with the actual correlation_id for exact match
    expected_invoker_config_received_by_tool_invoker_reg_test["correlation_id"] = actual_invoker_config_arg_to_tool_invoker_reg["correlation_id"]

    assert actual_invoker_config_arg_to_tool_invoker_reg == expected_invoker_config_received_by_tool_invoker_reg_test


@pytest.mark.asyncio
async def test_genie_register_tool_functions_no_decorated_tools(fully_mocked_genie: Genie, caplog):
    genie_instance = await fully_mocked_genie
    caplog.set_level(logging.WARNING)

    def not_decorated_func(): pass

    if not hasattr(genie_instance._tool_manager, "_tools") or not isinstance(genie_instance._tool_manager._tools, dict): # type: ignore
        genie_instance._tool_manager._tools = {} # type: ignore

    original_tool_count = len(genie_instance._tool_manager._tools) # type: ignore
    await genie_instance.register_tool_functions([not_decorated_func])

    assert len(genie_instance._tool_manager._tools) == original_tool_count # type: ignore
    assert "Genie: Function 'not_decorated_func' not @tool decorated. Skipping." in caplog.text # Updated log message

@pytest.mark.asyncio
async def test_genie_register_tool_functions_tool_manager_none(fully_mocked_genie: Genie, caplog):
    genie_instance = await fully_mocked_genie
    genie_instance._tool_manager = None
    caplog.set_level(logging.ERROR)
    await genie_instance.register_tool_functions([my_decorated_sync_func_for_genie_test])
    assert "Genie: ToolManager not initialized." in caplog.text # Updated log message

@pytest.mark.asyncio
async def test_genie_register_tool_functions_empty_list(fully_mocked_genie: Genie, caplog):
    genie_instance = await fully_mocked_genie
    caplog.set_level(logging.INFO)
    if not hasattr(genie_instance._tool_manager, "_tools") or not isinstance(genie_instance._tool_manager._tools, dict): # type: ignore
        genie_instance._tool_manager._tools = {} # type: ignore
    original_tool_count = len(genie_instance._tool_manager._tools) # type: ignore
    await genie_instance.register_tool_functions([])
    assert len(genie_instance._tool_manager._tools) == original_tool_count # type: ignore
    assert "Genie: Registered 0 function-based tools." not in caplog.text # Updated log message

@pytest.mark.asyncio
async def test_genie_register_tool_functions_duplicate_tool(fully_mocked_genie: Genie, caplog):
    genie_instance = await fully_mocked_genie
    caplog.set_level(logging.WARNING)
    if not hasattr(genie_instance._tool_manager, "_tools") or not isinstance(genie_instance._tool_manager._tools, dict): # type: ignore
        genie_instance._tool_manager._tools = {} # type: ignore

    await genie_instance.register_tool_functions([my_decorated_sync_func_for_genie_test])
    original_tool_count = len(genie_instance._tool_manager._tools) # type: ignore

    await genie_instance.register_tool_functions([my_decorated_sync_func_for_genie_test])
    assert len(genie_instance._tool_manager._tools) == original_tool_count # type: ignore
    assert "Genie: Tool 'my_decorated_sync_func_for_genie_test' already registered. Overwriting." in caplog.text # Updated log message

@pytest.mark.asyncio
async def test_genie_register_tool_functions_no_tool_lookup_service(fully_mocked_genie: Genie, caplog):
    genie_instance = await fully_mocked_genie
    if not hasattr(genie_instance._tool_manager, "_tools") or not isinstance(genie_instance._tool_manager._tools, dict): # type: ignore
        genie_instance._tool_manager._tools = {} # type: ignore
    genie_instance._tool_lookup_service = None # type: ignore
    caplog.set_level(logging.INFO)

    await genie_instance.register_tool_functions([my_decorated_sync_func_for_genie_test])
    assert "Genie: Invalidated tool lookup index." not in caplog.text # Updated log message
