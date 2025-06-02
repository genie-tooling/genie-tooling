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
from genie_tooling.hitl.types import ApprovalResponse
from genie_tooling.interfaces import (
    ConversationInterface,
    HITLInterface,
    LLMInterface,
    ObservabilityInterface,
    PromptInterface,
    RAGInterface,
    TaskQueueInterface,
    UsageTrackingInterface,
)
from genie_tooling.invocation.invoker import ToolInvoker
from genie_tooling.llm_providers.abc import LLMProviderPlugin
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
    provider = MockKeyProviderForGenie({
        "TEST_KEY": "test_value",
        "OPENAI_API_KEY": "test_openai_key",
        "GOOGLE_API_KEY": "test_google_key",
        "LLAMA_CPP_API_KEY_TEST": "test_llama_key",
        "QDRANT_API_KEY_TEST": "test_qdrant_key_from_facade_fixture"
    })
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
        "InteractionTracingManager_cls": mocker.patch("genie_tooling.genie.InteractionTracingManager"),
        "HITLManager_cls": mocker.patch("genie_tooling.genie.HITLManager"),
        "TokenUsageManager_cls": mocker.patch("genie_tooling.genie.TokenUsageManager"),
        "GuardrailManager_cls": mocker.patch("genie_tooling.genie.GuardrailManager"),
        "PromptManager_cls": mocker.patch("genie_tooling.genie.PromptManager"),
        "ConversationStateManager_cls": mocker.patch("genie_tooling.genie.ConversationStateManager"),
        "LLMOutputParserManager_cls": mocker.patch("genie_tooling.genie.LLMOutputParserManager"),
        "DistributedTaskQueueManager_cls": mocker.patch("genie_tooling.genie.DistributedTaskQueueManager"),
    }

    deps["pm_instance_for_kp_loading"] = AsyncMock(spec=PluginManager)
    deps["pm_instance_for_kp_loading"].discover_plugins = AsyncMock()
    async def get_kp_side_effect(plugin_id, config=None, **kwargs):
        if plugin_id == PLUGIN_ID_ALIASES["env_keys"]:
            env_kp_mock = MockKeyProviderForGenie(plugin_id=PLUGIN_ID_ALIASES["env_keys"])
            await env_kp_mock.setup(config)
            return env_kp_mock
        specific_kp_mock = MockKeyProviderForGenie(plugin_id=plugin_id)
        await specific_kp_mock.setup(config)
        return specific_kp_mock
    deps["pm_instance_for_kp_loading"].get_plugin_instance = AsyncMock(side_effect=get_kp_side_effect)
    deps["pm_instance_for_kp_loading"].list_discovered_plugin_classes = MagicMock(return_value={})
    deps["pm_instance_for_kp_loading"]._plugin_instances = {}
    deps["pm_instance_for_kp_loading"]._discovered_plugin_classes = {}

    deps["pm_instance_main"] = AsyncMock(spec=PluginManager)
    deps["pm_instance_main"].discover_plugins = AsyncMock()
    deps["pm_instance_main"].get_plugin_instance = AsyncMock(return_value=None)
    deps["pm_instance_main"].list_discovered_plugin_classes = MagicMock(return_value={})
    deps["pm_instance_main"]._plugin_instances = {}
    deps["pm_instance_main"]._discovered_plugin_classes = {}
    deps["pm_instance_main"].teardown_all_plugins = AsyncMock()

    pm_constructor_call_sequence = [deps["pm_instance_for_kp_loading"], deps["pm_instance_main"]]

    def pm_constructor_side_effect(*args, **kwargs):
        if not pm_constructor_call_sequence:
            raise AssertionError("PluginManager constructor called more than twice.")
        return pm_constructor_call_sequence.pop(0)

    deps["PluginManager_cls"].side_effect = pm_constructor_side_effect


    deps["tm_inst"] = AsyncMock(spec=ToolManager)
    deps["tm_inst"]._tools = {}
    deps["tm_inst"].initialize_tools = AsyncMock()
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
    deps["llmpm_inst"].teardown = AsyncMock()
    deps["LLMProviderManager_cls"].return_value = deps["llmpm_inst"]

    deps["cpm_inst"] = AsyncMock(spec=CommandProcessorManager)
    deps["cpm_inst"].teardown = AsyncMock()
    deps["CommandProcessorManager_cls"].return_value = deps["cpm_inst"]

    deps["tracing_m_inst"] = AsyncMock(spec=InteractionTracingManager)
    deps["tracing_m_inst"].teardown = AsyncMock()
    deps["InteractionTracingManager_cls"].return_value = deps["tracing_m_inst"]

    deps["hitl_m_inst"] = AsyncMock(spec=HITLManager)
    deps["hitl_m_inst"].teardown = AsyncMock()
    deps["HITLManager_cls"].return_value = deps["hitl_m_inst"]

    deps["token_usage_m_inst"] = AsyncMock(spec=TokenUsageManager)
    deps["token_usage_m_inst"].teardown = AsyncMock()
    deps["TokenUsageManager_cls"].return_value = deps["token_usage_m_inst"]

    deps["guardrail_m_inst"] = AsyncMock(spec=GuardrailManager)
    deps["guardrail_m_inst"].teardown = AsyncMock()
    deps["GuardrailManager_cls"].return_value = deps["guardrail_m_inst"]

    deps["prompt_m_inst"] = AsyncMock(spec=PromptManager)
    deps["prompt_m_inst"].teardown = AsyncMock()
    deps["PromptManager_cls"].return_value = deps["prompt_m_inst"]

    deps["convo_m_inst"] = AsyncMock(spec=ConversationStateManager)
    deps["convo_m_inst"].teardown = AsyncMock()
    deps["ConversationStateManager_cls"].return_value = deps["convo_m_inst"]

    deps["llm_output_parser_m_inst"] = AsyncMock(spec=LLMOutputParserManager)
    deps["llm_output_parser_m_inst"].teardown = AsyncMock()
    deps["LLMOutputParserManager_cls"].return_value = deps["llm_output_parser_m_inst"]

    deps["task_queue_m_inst"] = AsyncMock(spec=DistributedTaskQueueManager)
    deps["task_queue_m_inst"].teardown = AsyncMock()
    deps["DistributedTaskQueueManager_cls"].return_value = deps["task_queue_m_inst"]


    deps["resolver_inst"] = MagicMock(spec=ConfigResolver)
    def generic_resolver_side_effect(user_cfg: MiddlewareConfig, key_provider_instance: Optional[KeyProvider] = None) -> MiddlewareConfig:
        resolved = user_cfg.model_copy(deep=True)
        if key_provider_instance and hasattr(key_provider_instance, "plugin_id"):
            resolved.key_provider_id = key_provider_instance.plugin_id
        elif resolved.key_provider_id is None:
            resolved.key_provider_id = PLUGIN_ID_ALIASES["env_keys"]

        features = resolved.features

        if features.llm != "none" and PLUGIN_ID_ALIASES.get(features.llm):
            llm_id = PLUGIN_ID_ALIASES[features.llm]
            resolved.default_llm_provider_id = llm_id
            conf = resolved.llm_provider_configurations.setdefault(llm_id, {})
            if features.llm == "ollama":
                conf["model_name"] = features.llm_ollama_model_name
            elif features.llm == "openai":
                conf["model_name"] = features.llm_openai_model_name
                if key_provider_instance: conf["key_provider"] = key_provider_instance
            elif features.llm == "gemini":
                conf["model_name"] = features.llm_gemini_model_name
                if key_provider_instance: conf["key_provider"] = key_provider_instance
            elif features.llm == "llama_cpp":
                conf["model_name"] = features.llm_llama_cpp_model_name
                conf["base_url"] = features.llm_llama_cpp_base_url
                if features.llm_llama_cpp_api_key_name and key_provider_instance:
                    conf["api_key_name"] = features.llm_llama_cpp_api_key_name
                    conf["key_provider"] = key_provider_instance

        if features.rag_embedder != "none":
            embed_alias = {"sentence_transformer": "st_embedder", "openai": "openai_embedder"}.get(features.rag_embedder)
            if embed_alias and PLUGIN_ID_ALIASES.get(embed_alias):
                embed_id = PLUGIN_ID_ALIASES[embed_alias]
                resolved.default_rag_embedder_id = embed_id
                conf = resolved.embedding_generator_configurations.setdefault(embed_id, {})
                if features.rag_embedder == "sentence_transformer":
                    conf["model_name"] = features.rag_embedder_st_model_name
                elif features.rag_embedder == "openai" and key_provider_instance:
                    conf["key_provider"] = key_provider_instance

        if features.rag_vector_store != "none":
            vs_alias = {"faiss": "faiss_vs", "chroma": "chroma_vs", "qdrant": "qdrant_vs"}.get(features.rag_vector_store)
            if vs_alias and PLUGIN_ID_ALIASES.get(vs_alias):
                vs_id = PLUGIN_ID_ALIASES[vs_alias]
                resolved.default_rag_vector_store_id = vs_id
                conf = resolved.vector_store_configurations.setdefault(vs_id, {})
                if features.rag_vector_store == "chroma":
                    conf["collection_name"] = features.rag_vector_store_chroma_collection_name
                    if features.rag_vector_store_chroma_path is not None:
                        conf["path"] = features.rag_vector_store_chroma_path
                elif features.rag_vector_store == "qdrant":
                    conf["collection_name"] = features.rag_vector_store_qdrant_collection_name
                    if features.rag_vector_store_qdrant_url: conf["url"] = features.rag_vector_store_qdrant_url
                    if features.rag_vector_store_qdrant_path: conf["path"] = features.rag_vector_store_qdrant_path
                    if features.rag_vector_store_qdrant_api_key_name and key_provider_instance:
                        conf["api_key_name"] = features.rag_vector_store_qdrant_api_key_name
                        conf["key_provider"] = key_provider_instance
                    if features.rag_vector_store_qdrant_embedding_dim:
                        conf["embedding_dim"] = features.rag_vector_store_qdrant_embedding_dim

        if features.tool_lookup != "none":
            lookup_id_key_part = features.tool_lookup
            lookup_id = PLUGIN_ID_ALIASES.get(f"{lookup_id_key_part}_lookup")
            if lookup_id:
                resolved.default_tool_lookup_provider_id = lookup_id
                tl_prov_cfg = resolved.tool_lookup_provider_configurations.setdefault(lookup_id, {})
                if features.tool_lookup_formatter_id_alias:
                    resolved.default_tool_indexing_formatter_id = PLUGIN_ID_ALIASES.get(features.tool_lookup_formatter_id_alias, features.tool_lookup_formatter_id_alias)

                if features.tool_lookup == "embedding":
                    embed_alias_tl = features.tool_lookup_embedder_id_alias or "st_embedder"
                    embed_id_tl = PLUGIN_ID_ALIASES.get(embed_alias_tl)
                    if embed_id_tl:
                        tl_prov_cfg["embedder_id"] = embed_id_tl
                        emb_tl_conf = tl_prov_cfg.setdefault("embedder_config", {})
                        if embed_alias_tl == "st_embedder" and features.rag_embedder_st_model_name:
                             emb_tl_conf["model_name"] = features.rag_embedder_st_model_name
                        elif embed_alias_tl == "openai_embedder" and key_provider_instance:
                            emb_tl_conf["key_provider"] = key_provider_instance

                    if features.tool_lookup_chroma_collection_name is not None:
                        tl_prov_cfg["vector_store_id"] = PLUGIN_ID_ALIASES.get("chroma_vs")
                        vs_tl_conf = tl_prov_cfg.setdefault("vector_store_config", {})
                        vs_tl_conf["collection_name"] = features.tool_lookup_chroma_collection_name
                        if features.tool_lookup_chroma_path is not None:
                            vs_tl_conf["path"] = features.tool_lookup_chroma_path

        if features.observability_tracer != "none":
            tracer_id = PLUGIN_ID_ALIASES.get(features.observability_tracer)
            if tracer_id:
                resolved.default_observability_tracer_id = tracer_id
                conf = resolved.observability_tracer_configurations.setdefault(tracer_id, {})
                if features.observability_tracer == "otel_tracer" and features.observability_otel_endpoint:
                    conf["otlp_http_endpoint"] = features.observability_otel_endpoint

        if features.hitl_approver != "none":
            approver_id = PLUGIN_ID_ALIASES.get(features.hitl_approver)
            if approver_id:
                resolved.default_hitl_approver_id = approver_id
                resolved.hitl_approver_configurations.setdefault(approver_id, {})

        if features.token_usage_recorder != "none":
            recorder_id = PLUGIN_ID_ALIASES.get(features.token_usage_recorder)
            if recorder_id:
                resolved.default_token_usage_recorder_id = recorder_id
                resolved.token_usage_recorder_configurations.setdefault(recorder_id, {})

        if features.input_guardrails:
            resolved.default_input_guardrail_ids = [PLUGIN_ID_ALIASES.get(g, g) for g in features.input_guardrails]
            for gr_id in resolved.default_input_guardrail_ids:
                resolved.guardrail_configurations.setdefault(gr_id, {})

        if features.prompt_registry != "none":
            reg_id = PLUGIN_ID_ALIASES.get(features.prompt_registry)
            if reg_id:
                resolved.default_prompt_registry_id = reg_id
                resolved.prompt_registry_configurations.setdefault(reg_id, {})

        if features.prompt_template_engine != "none":
            engine_id = PLUGIN_ID_ALIASES.get(features.prompt_template_engine)
            if engine_id:
                resolved.default_prompt_template_plugin_id = engine_id
                resolved.prompt_template_configurations.setdefault(engine_id, {})

        if features.conversation_state_provider != "none":
            convo_id = PLUGIN_ID_ALIASES.get(features.conversation_state_provider)
            if convo_id:
                resolved.default_conversation_state_provider_id = convo_id
                resolved.conversation_state_provider_configurations.setdefault(convo_id, {})

        if features.default_llm_output_parser != "none":
            parser_alias = features.default_llm_output_parser
            if parser_alias:
                parser_id = PLUGIN_ID_ALIASES.get(parser_alias)
                if parser_id:
                    resolved.default_llm_output_parser_id = parser_id
                    resolved.llm_output_parser_configurations.setdefault(parser_id, {})

        if features.task_queue != "none":
            task_q_alias = {"celery": "celery_task_queue", "rq": "rq_task_queue"}.get(features.task_queue)
            if task_q_alias and PLUGIN_ID_ALIASES.get(task_q_alias):
                task_q_id = PLUGIN_ID_ALIASES[task_q_alias]
                resolved.default_distributed_task_queue_id = task_q_id
                conf = resolved.distributed_task_queue_configurations.setdefault(task_q_id, {})
                if features.task_queue == "celery":
                    conf["celery_broker_url"] = features.task_queue_celery_broker_url
                    conf["celery_backend_url"] = features.task_queue_celery_backend_url

        if resolved.default_llm_provider_id is None: resolved.default_llm_provider_id = "test_llm_fallback"
        if resolved.default_command_processor_id is None: resolved.default_command_processor_id = "test_proc_fallback"
        if resolved.default_rag_retriever_id is None: resolved.default_rag_retriever_id = "basic_similarity_retriever_v1"
        if resolved.default_rag_embedder_id is None: resolved.default_rag_embedder_id = "test_embedder_fallback"

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

    llm_provider_for_tests = AsyncMock(spec=LLMProviderPlugin)
    llm_provider_for_tests.chat = AsyncMock(return_value={"message": {"role":"assistant", "content":"Hi there!"}, "finish_reason":"stop"})
    llm_provider_for_tests.generate = AsyncMock(return_value={"text": "Generated text", "finish_reason":"stop"})
    mock_genie_dependencies["llmpm_inst"].get_llm_provider.return_value = llm_provider_for_tests


    genie_instance = await Genie.create(
        config=mock_middleware_config_facade,
        key_provider_instance=resolved_kp_instance
    )
    genie_instance.llm = LLMInterface(
        genie_instance._llm_provider_manager, # type: ignore
        genie_instance._config.default_llm_provider_id,
        genie_instance._llm_output_parser_manager, # type: ignore
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
    genie_instance.observability = ObservabilityInterface(genie_instance._tracing_manager) # type: ignore
    genie_instance.human_in_loop = HITLInterface(genie_instance._hitl_manager) # type: ignore
    genie_instance.usage = UsageTrackingInterface(genie_instance._token_usage_manager) # type: ignore
    genie_instance.prompts = PromptInterface(genie_instance._prompt_manager) # type: ignore
    genie_instance.conversation = ConversationInterface(genie_instance._conversation_manager) # type: ignore
    genie_instance.task_queue = TaskQueueInterface(genie_instance._task_queue_manager) # type: ignore


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
        resolved_kp = await mock_key_provider_instance_facade

        main_pm_mock_for_test = mock_genie_dependencies["pm_instance_main"]

        genie = await Genie.create(config=mock_middleware_config_facade, key_provider_instance=resolved_kp)

        resolver_mock.resolve.assert_called_once_with(mock_middleware_config_facade, key_provider_instance=resolved_kp)

        config_actually_used_by_genie = genie._config

        mock_genie_dependencies["LLMProviderManager_cls"].assert_called_once_with(
            main_pm_mock_for_test,
            resolved_kp,
            config_actually_used_by_genie,
            mock_genie_dependencies["token_usage_m_inst"]
        )
        # This assertion is now valid because config_actually_used_by_genie *is* the return value
        # of the resolver_mock.resolve's side_effect.

    async def test_create_with_default_key_provider_id(self, mock_genie_dependencies, mock_middleware_config_facade: MiddlewareConfig):
        pm_for_kp_loading_mock = mock_genie_dependencies["pm_instance_for_kp_loading"]

        genie = await Genie.create(config=mock_middleware_config_facade)

        pm_for_kp_loading_mock.get_plugin_instance.assert_any_call(PLUGIN_ID_ALIASES["env_keys"])
        assert isinstance(genie._key_provider, MockKeyProviderForGenie)
        assert genie._key_provider.plugin_id == PLUGIN_ID_ALIASES["env_keys"]

        mock_genie_dependencies["resolver_inst"].resolve.assert_called_once_with(
            mock_middleware_config_facade, key_provider_instance=genie._key_provider
        )


    async def test_create_key_provider_id_from_config_no_instance(self, mock_genie_dependencies, mock_middleware_config_facade: MiddlewareConfig):
        user_kp_id = "user_specified_kp_id_v1"
        mock_middleware_config_facade.key_provider_id = user_kp_id

        pm_for_kp_loading_mock = mock_genie_dependencies["pm_instance_for_kp_loading"]

        mock_user_kp_instance = MockKeyProviderForGenie(plugin_id=user_kp_id)
        await mock_user_kp_instance.setup()

        original_get_kp_side_effect = pm_for_kp_loading_mock.get_plugin_instance.side_effect
        async def new_get_kp_side_effect(plugin_id, config=None, **kwargs):
            if plugin_id == user_kp_id:
                return mock_user_kp_instance
            if callable(original_get_kp_side_effect):
                return await original_get_kp_side_effect(plugin_id, config, **kwargs)
            return None
        pm_for_kp_loading_mock.get_plugin_instance.side_effect = new_get_kp_side_effect

        genie = await Genie.create(config=mock_middleware_config_facade, key_provider_instance=None)

        pm_for_kp_loading_mock.get_plugin_instance.assert_any_call(user_kp_id)
        assert genie._key_provider is mock_user_kp_instance
        mock_genie_dependencies["resolver_inst"].resolve.assert_called_once_with(
            mock_middleware_config_facade, key_provider_instance=mock_user_kp_instance
        )


    async def test_create_key_provider_load_fails(self, mock_genie_dependencies, mock_middleware_config_facade: MiddlewareConfig):
        failing_kp_id = "failing_kp_id"
        mock_middleware_config_facade.key_provider_id = failing_kp_id

        pm_for_kp_loading_mock = mock_genie_dependencies["pm_instance_for_kp_loading"]
        original_get_kp_side_effect = pm_for_kp_loading_mock.get_plugin_instance.side_effect
        async def new_get_kp_side_effect_fail(plugin_id, config=None, **kwargs):
            if plugin_id == failing_kp_id:
                return None
            if callable(original_get_kp_side_effect):
                return await original_get_kp_side_effect(plugin_id, config, **kwargs)
            return None
        pm_for_kp_loading_mock.get_plugin_instance.side_effect = new_get_kp_side_effect_fail

        with pytest.raises(RuntimeError, match=f"Failed to load KeyProvider with ID '{failing_kp_id}'"):
            await Genie.create(config=mock_middleware_config_facade, key_provider_instance=None)

    async def test_create_key_provider_wrong_type(self, mock_genie_dependencies, mock_middleware_config_facade: MiddlewareConfig):
        wrong_type_kp_id = "wrong_type_kp_id"
        mock_middleware_config_facade.key_provider_id = wrong_type_kp_id

        pm_for_kp_loading_mock = mock_genie_dependencies["pm_instance_for_kp_loading"]
        wrong_type_plugin_instance = AsyncMock(spec=CorePluginType)

        original_get_kp_side_effect = pm_for_kp_loading_mock.get_plugin_instance.side_effect
        async def new_get_kp_side_effect_wrong_type(plugin_id, config=None, **kwargs):
            if plugin_id == wrong_type_kp_id:
                return wrong_type_plugin_instance
            if callable(original_get_kp_side_effect):
                return await original_get_kp_side_effect(plugin_id, config, **kwargs)
            return None
        pm_for_kp_loading_mock.get_plugin_instance.side_effect = new_get_kp_side_effect_wrong_type

        with pytest.raises(RuntimeError, match=f"Failed to load KeyProvider with ID '{wrong_type_kp_id}'"):
            await Genie.create(config=mock_middleware_config_facade, key_provider_instance=None)


    async def test_create_kp_instance_already_in_main_pm(self, mock_genie_dependencies, mock_middleware_config_facade: MiddlewareConfig, mock_key_provider_instance_facade: MockKeyProviderForGenie):
        kp_instance = await mock_key_provider_instance_facade
        kp_instance_id = kp_instance.plugin_id

        pm_for_kp_loading_mock = mock_genie_dependencies["pm_instance_for_kp_loading"]
        main_pm_mock = mock_genie_dependencies["pm_instance_main"]

        main_pm_mock._plugin_instances[kp_instance_id] = kp_instance
        main_pm_mock._discovered_plugin_classes[kp_instance_id] = type(kp_instance)

        genie = await Genie.create(config=mock_middleware_config_facade, key_provider_instance=kp_instance)

        assert genie._key_provider is kp_instance
        pm_for_kp_loading_mock.discover_plugins.assert_called_once()
        main_pm_mock.discover_plugins.assert_called_once()
        assert main_pm_mock._plugin_instances.get(kp_instance_id) is kp_instance


    @pytest.mark.parametrize(
        "feature_settings, expected_llm_id_alias, expected_model_key, expected_model_value_attr",
        [
            (FeatureSettings(llm="ollama", llm_ollama_model_name="ollama-test"), "ollama", "model_name", "llm_ollama_model_name"),
            (FeatureSettings(llm="openai", llm_openai_model_name="openai-test"), "openai", "model_name", "llm_openai_model_name"),
            (FeatureSettings(llm="gemini", llm_gemini_model_name="gemini-test"), "gemini", "model_name", "llm_gemini_model_name"),
            (FeatureSettings(llm="llama_cpp", llm_llama_cpp_model_name="llama-test", llm_llama_cpp_base_url="http://llamacpp", llm_llama_cpp_api_key_name="LLAMA_CPP_API_KEY_TEST"), "llama_cpp", "model_name", "llm_llama_cpp_model_name"),
        ]
    )
    async def test_create_with_llm_features(
        self, mock_genie_dependencies, mock_key_provider_instance_facade,
        feature_settings, expected_llm_id_alias, expected_model_key, expected_model_value_attr
    ):
        config = MiddlewareConfig(features=feature_settings)
        kp_instance = await mock_key_provider_instance_facade
        genie = await Genie.create(config=config, key_provider_instance=kp_instance)

        expected_canonical_id = PLUGIN_ID_ALIASES[expected_llm_id_alias]
        assert genie._config.default_llm_provider_id == expected_canonical_id
        assert expected_canonical_id in genie._config.llm_provider_configurations
        llm_conf = genie._config.llm_provider_configurations[expected_canonical_id]
        assert llm_conf[expected_model_key] == getattr(feature_settings, expected_model_value_attr)
        if expected_llm_id_alias == "llama_cpp":
            assert llm_conf["base_url"] == feature_settings.llm_llama_cpp_base_url
        if expected_llm_id_alias in ["openai", "gemini"] or \
           (expected_llm_id_alias == "llama_cpp" and feature_settings.llm_llama_cpp_api_key_name is not None):
            assert llm_conf.get("key_provider") is kp_instance
        elif expected_llm_id_alias == "ollama":
             assert "key_provider" not in llm_conf

    async def test_create_with_rag_features(self, mock_genie_dependencies, mock_key_provider_instance_facade):
        features = FeatureSettings(
            rag_embedder="openai",
            rag_vector_store="chroma",
            rag_vector_store_chroma_path="/test/chroma",
            rag_vector_store_chroma_collection_name="test_rag"
        )
        config = MiddlewareConfig(features=features)
        kp_instance = await mock_key_provider_instance_facade
        genie = await Genie.create(config=config, key_provider_instance=kp_instance)

        openai_embed_id = PLUGIN_ID_ALIASES["openai_embedder"]
        chroma_vs_id = PLUGIN_ID_ALIASES["chroma_vs"]

        assert genie._config.default_rag_embedder_id == openai_embed_id
        assert openai_embed_id in genie._config.embedding_generator_configurations
        assert genie._config.embedding_generator_configurations[openai_embed_id].get("key_provider") is kp_instance

        assert genie._config.default_rag_vector_store_id == chroma_vs_id
        assert chroma_vs_id in genie._config.vector_store_configurations
        vs_conf = genie._config.vector_store_configurations[chroma_vs_id]
        assert vs_conf["path"] == "/test/chroma"
        assert vs_conf["collection_name"] == "test_rag"

    async def test_create_with_tool_lookup_features(self, mock_genie_dependencies, mock_key_provider_instance_facade):
        features = FeatureSettings(
            tool_lookup="embedding",
            tool_lookup_embedder_id_alias="openai_embedder",
            tool_lookup_formatter_id_alias="hr_json_formatter",
            tool_lookup_chroma_path="/lookup/chroma",
            tool_lookup_chroma_collection_name="lookup_coll"
        )
        config = MiddlewareConfig(features=features)
        kp_instance = await mock_key_provider_instance_facade
        genie = await Genie.create(config=config, key_provider_instance=kp_instance)

        embedding_lookup_id = PLUGIN_ID_ALIASES["embedding_lookup"]
        hr_json_formatter_id = PLUGIN_ID_ALIASES["hr_json_formatter"]
        openai_embedder_id = PLUGIN_ID_ALIASES["openai_embedder"]

        assert genie._config.default_tool_lookup_provider_id == embedding_lookup_id
        assert genie._config.default_tool_indexing_formatter_id == hr_json_formatter_id

        assert embedding_lookup_id in genie._config.tool_lookup_provider_configurations
        lookup_conf = genie._config.tool_lookup_provider_configurations[embedding_lookup_id]
        assert lookup_conf["embedder_id"] == openai_embedder_id
        assert lookup_conf["embedder_config"]["key_provider"] is kp_instance
        assert lookup_conf["vector_store_id"] == PLUGIN_ID_ALIASES["chroma_vs"]
        assert lookup_conf["vector_store_config"]["path"] == "/lookup/chroma"
        assert lookup_conf["vector_store_config"]["collection_name"] == "lookup_coll"

    async def test_create_with_p1_5_features(self, mock_genie_dependencies, mock_key_provider_instance_facade):
        features = FeatureSettings(
            observability_tracer="otel_tracer",
            observability_otel_endpoint="http://otel-collector:4318",
            hitl_approver="cli_hitl_approver",
            token_usage_recorder="otel_metrics_recorder",
            input_guardrails=["keyword_blocklist_guardrail"],
            prompt_registry="file_system_prompt_registry",
            conversation_state_provider="redis_convo_provider",
            default_llm_output_parser="pydantic_output_parser",
            task_queue="celery"
        )
        config = MiddlewareConfig(features=features)
        kp_instance = await mock_key_provider_instance_facade
        genie = await Genie.create(config=config, key_provider_instance=kp_instance)

        otel_tracer_id = PLUGIN_ID_ALIASES["otel_tracer"]
        cli_hitl_id = PLUGIN_ID_ALIASES["cli_hitl_approver"]
        otel_metrics_recorder_id = PLUGIN_ID_ALIASES["otel_metrics_recorder"]
        keyword_blocklist_id = PLUGIN_ID_ALIASES["keyword_blocklist_guardrail"]
        fs_prompt_reg_id = PLUGIN_ID_ALIASES["file_system_prompt_registry"]
        redis_convo_id = PLUGIN_ID_ALIASES["redis_convo_provider"]
        pydantic_parser_id = PLUGIN_ID_ALIASES["pydantic_output_parser"]
        celery_task_q_id = PLUGIN_ID_ALIASES["celery_task_queue"]


        assert genie._config.default_observability_tracer_id == otel_tracer_id
        assert otel_tracer_id in genie._config.observability_tracer_configurations
        assert genie._config.observability_tracer_configurations[otel_tracer_id]["otlp_http_endpoint"] == "http://otel-collector:4318"

        assert genie._config.default_hitl_approver_id == cli_hitl_id
        assert cli_hitl_id in genie._config.hitl_approver_configurations

        assert genie._config.default_token_usage_recorder_id == otel_metrics_recorder_id
        assert otel_metrics_recorder_id in genie._config.token_usage_recorder_configurations

        assert keyword_blocklist_id in genie._config.default_input_guardrail_ids
        assert keyword_blocklist_id in genie._config.guardrail_configurations

        assert genie._config.default_prompt_registry_id == fs_prompt_reg_id
        assert fs_prompt_reg_id in genie._config.prompt_registry_configurations

        assert genie._config.default_conversation_state_provider_id == redis_convo_id
        assert redis_convo_id in genie._config.conversation_state_provider_configurations

        assert genie._config.default_llm_output_parser_id == pydantic_parser_id
        assert pydantic_parser_id in genie._config.llm_output_parser_configurations

        assert genie._config.default_distributed_task_queue_id == celery_task_q_id
        assert celery_task_q_id in genie._config.distributed_task_queue_configurations


@pytest.mark.asyncio
async def test_genie_execute_tool(fully_mocked_genie: Genie):
    actual_genie = await fully_mocked_genie
    tool_id = "calculator"
    params = {"num1": 1, "num2": 2, "operation": "add"}

    expected_invoker_config_received_by_tool_invoker = {
        "plugin_manager": actual_genie._plugin_manager,
        "guardrail_manager": actual_genie._guardrail_manager,
        "tracing_manager": actual_genie._tracing_manager,
        "correlation_id": Any,
    }

    await actual_genie.execute_tool(tool_id, **params)

    actual_genie._tool_invoker.invoke.assert_awaited_once() # type: ignore
    call_args = actual_genie._tool_invoker.invoke.await_args # type: ignore

    assert call_args.kwargs["tool_identifier"] == tool_id
    assert call_args.kwargs["params"] == params
    assert call_args.kwargs["key_provider"] == actual_genie._key_provider

    actual_invoker_config_arg_to_tool_invoker = call_args.kwargs["invoker_config"]
    assert isinstance(actual_invoker_config_arg_to_tool_invoker["correlation_id"], str)

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
    with pytest.raises(RuntimeError, match="LLM Provider 'test_llm_fallback' not found or failed to load."):
        await actual_genie.llm.generate("prompt", provider_id="test_llm_fallback")


@pytest.mark.asyncio
async def test_genie_rag_search(fully_mocked_genie: Genie):
    actual_genie = await fully_mocked_genie
    retriever_id_to_use = actual_genie._config.default_rag_retriever_id
    assert retriever_id_to_use == "basic_similarity_retriever_v1", \
        f"Expected default_rag_retriever_id to be 'basic_similarity_retriever_v1', got {retriever_id_to_use}"

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
    tracing_m_assert = actual_genie._tracing_manager
    hitl_m_assert = actual_genie._hitl_manager
    token_usage_m_assert = actual_genie._token_usage_manager
    guardrail_m_assert = actual_genie._guardrail_manager
    prompt_m_assert = actual_genie._prompt_manager
    convo_m_assert = actual_genie._conversation_manager
    llm_output_parser_m_assert = actual_genie._llm_output_parser_manager
    task_queue_m_assert = actual_genie._task_queue_manager


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
    assert task_queue_m_assert is not None


    await actual_genie.close()
    pm_mock_for_assert.teardown_all_plugins.assert_awaited_once() # type: ignore
    llmpm_mock_for_assert.teardown.assert_awaited_once() # type: ignore
    cpm_mock_for_assert.teardown.assert_awaited_once() # type: ignore

    tracing_m_assert.teardown.assert_awaited_once() # type: ignore
    hitl_m_assert.teardown.assert_awaited_once() # type: ignore
    token_usage_m_assert.teardown.assert_awaited_once() # type: ignore
    guardrail_m_assert.teardown.assert_awaited_once() # type: ignore
    prompt_m_assert.teardown.assert_awaited_once() # type: ignore
    convo_m_assert.teardown.assert_awaited_once() # type: ignore
    llm_output_parser_m_assert.teardown.assert_awaited_once() # type: ignore
    task_queue_m_assert.teardown.assert_awaited_once() # type: ignore

    assert actual_genie._plugin_manager is None
    assert actual_genie._llm_provider_manager is None
    assert actual_genie._command_processor_manager is None
    assert actual_genie.llm is None
    assert actual_genie.rag is None
    assert actual_genie._tracing_manager is None
    assert actual_genie._hitl_manager is None
    assert actual_genie._token_usage_manager is None
    assert actual_genie._guardrail_manager is None
    assert actual_genie._prompt_manager is None
    assert actual_genie._conversation_manager is None
    assert actual_genie._llm_output_parser_manager is None
    assert actual_genie._task_queue_manager is None
    assert actual_genie.observability is None
    assert actual_genie.human_in_loop is None
    assert actual_genie.usage is None
    assert actual_genie.prompts is None
    assert actual_genie.conversation is None
    assert actual_genie.task_queue is None


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
    assert "Genie: Invalidated tool lookup index." in caplog.text

    expected_invoker_config_received_by_tool_invoker_reg_test = {
        "plugin_manager": genie_instance._plugin_manager,
        "guardrail_manager": genie_instance._guardrail_manager,
        "tracing_manager": genie_instance._tracing_manager,
        "correlation_id": Any,
    }

    await genie_instance.execute_tool("my_decorated_async_func_for_genie_test", param_a="hello")

    genie_instance._tool_invoker.invoke.assert_awaited_once() # type: ignore
    call_args_reg = genie_instance._tool_invoker.invoke.await_args # type: ignore

    assert call_args_reg.kwargs["tool_identifier"] == "my_decorated_async_func_for_genie_test"
    assert call_args_reg.kwargs["params"] == {"param_a": "hello"}
    assert call_args_reg.kwargs["key_provider"] == genie_instance._key_provider

    actual_invoker_config_arg_to_tool_invoker_reg = call_args_reg.kwargs["invoker_config"]
    assert isinstance(actual_invoker_config_arg_to_tool_invoker_reg["correlation_id"], str)

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
    assert "Genie: Function 'not_decorated_func' not @tool decorated. Skipping." in caplog.text

@pytest.mark.asyncio
async def test_genie_register_tool_functions_tool_manager_none(fully_mocked_genie: Genie, caplog):
    genie_instance = await fully_mocked_genie
    genie_instance._tool_manager = None
    caplog.set_level(logging.ERROR)
    await genie_instance.register_tool_functions([my_decorated_sync_func_for_genie_test])
    assert "Genie: ToolManager not initialized." in caplog.text

@pytest.mark.asyncio
async def test_genie_register_tool_functions_empty_list(fully_mocked_genie: Genie, caplog):
    genie_instance = await fully_mocked_genie
    caplog.set_level(logging.INFO)
    if not hasattr(genie_instance._tool_manager, "_tools") or not isinstance(genie_instance._tool_manager._tools, dict): # type: ignore
        genie_instance._tool_manager._tools = {} # type: ignore
    original_tool_count = len(genie_instance._tool_manager._tools) # type: ignore
    await genie_instance.register_tool_functions([])
    assert len(genie_instance._tool_manager._tools) == original_tool_count # type: ignore
    assert "Genie: Registered 0 function-based tools." not in caplog.text

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
    assert "Genie: Tool 'my_decorated_sync_func_for_genie_test' already registered. Overwriting." in caplog.text

@pytest.mark.asyncio
async def test_genie_register_tool_functions_no_tool_lookup_service(fully_mocked_genie: Genie, caplog):
    genie_instance = await fully_mocked_genie
    if not hasattr(genie_instance._tool_manager, "_tools") or not isinstance(genie_instance._tool_manager._tools, dict): # type: ignore
        genie_instance._tool_manager._tools = {} # type: ignore
    genie_instance._tool_lookup_service = None # type: ignore
    caplog.set_level(logging.INFO)

    await genie_instance.register_tool_functions([my_decorated_sync_func_for_genie_test])
    assert "Genie: Invalidated tool lookup index." not in caplog.text
