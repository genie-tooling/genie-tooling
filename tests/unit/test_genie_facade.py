### tests/unit/test_genie_facade.py
import logging
from typing import Dict
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest
from genie_tooling.command_processors.manager import CommandProcessorManager
from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.resolver import PLUGIN_ID_ALIASES, ConfigResolver
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.genie import Genie
from genie_tooling.guardrails.manager import GuardrailManager
from genie_tooling.hitl.manager import HITLManager
from genie_tooling.invocation.invoker import ToolInvoker
from genie_tooling.llm_providers.manager import LLMProviderManager
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
    return MiddlewareConfig()

@pytest.fixture
async def mock_key_provider_instance_facade() -> KeyProvider:
    provider = MagicMock(spec=KeyProvider)
    provider.plugin_id = "mock_kp_instance_id_for_facade"
    provider.get_key = AsyncMock(return_value="mock_key")
    return provider

@pytest.fixture
def mock_genie_dependencies(mocker):
    """Mocks all direct dependencies of Genie.create for controlled instantiation."""
    deps = {
        "PluginManager": mocker.patch("genie_tooling.genie.PluginManager", spec=PluginManager),
        "ToolManager": mocker.patch("genie_tooling.genie.ToolManager", spec=ToolManager),
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

    for dep_name, class_mock in deps.items():
        instance_mock = AsyncMock()
        class_mock.return_value = instance_mock

        if dep_name == "PluginManager":
            instance_mock.discover_plugins = AsyncMock()
            instance_mock.get_plugin_instance = AsyncMock()
        elif dep_name == "ToolManager":
            instance_mock.initialize_tools = AsyncMock()
        elif dep_name == "ConfigResolver":
            instance_mock.resolve = MagicMock()
        elif dep_name == "DefaultLogAdapter":
            instance_mock.process_event = AsyncMock()
    return deps

@pytest.fixture
async def fully_mocked_genie(
    mock_genie_dependencies: Dict,
    mock_middleware_config_facade: MiddlewareConfig,
    mock_key_provider_instance_facade: KeyProvider # This is an async fixture, pytest handles awaiting it
) -> Genie:
    """
    Creates a real Genie instance but with all its manager dependencies mocked.
    """
    kp_instance = mock_key_provider_instance_facade # Pytest has already awaited this

    real_resolver = ConfigResolver()
    resolved_config_for_test = real_resolver.resolve(
        mock_middleware_config_facade,
        kp_instance
    )
    mock_genie_dependencies["ConfigResolver"].return_value.resolve.return_value = resolved_config_for_test

    mock_genie_dependencies["ToolInvoker"].return_value.invoke = AsyncMock(return_value={"result": "tool executed"})
    mock_genie_dependencies["HITLManager"].return_value.is_active = True
    mock_genie_dependencies["HITLManager"].return_value.request_approval = AsyncMock(return_value={"status": "approved"})
    mock_genie_dependencies["ToolLookupService"].return_value.invalidate_index = MagicMock()

    genie_instance = await Genie.create(
        config=mock_middleware_config_facade,
        key_provider_instance=kp_instance
    )
    return genie_instance

@pytest.mark.asyncio
class TestGenieCreate:
    async def test_create_with_llm_features(
        self, mock_genie_dependencies, mock_key_provider_instance_facade
    ):
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

@pytest.mark.asyncio
async def test_genie_execute_tool(fully_mocked_genie: Genie):
    genie_instance = await fully_mocked_genie # CORRECTED: Await the fixture here
    await genie_instance.execute_tool("some_tool", param="value")
    genie_instance._tool_invoker.invoke.assert_awaited_once_with(
        tool_identifier="some_tool",
        params={"param": "value"},
        key_provider=genie_instance._key_provider,
        invoker_config=ANY
    )

@pytest.mark.asyncio
async def test_genie_register_tool_functions_no_tool_lookup_service(fully_mocked_genie: Genie, caplog):
    genie_instance = await fully_mocked_genie # CORRECTED: Await the fixture here
    genie_instance._tool_lookup_service = None
    caplog.set_level(logging.INFO)
    mock_func = MagicMock()
    mock_func._tool_metadata_ = {"identifier": "test_func"}
    mock_func._original_function_ = lambda: "test"
    await genie_instance.register_tool_functions([mock_func])
    assert "Genie: Invalidated tool lookup index." not in caplog.text
