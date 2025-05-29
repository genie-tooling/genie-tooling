from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest
from genie_tooling.config.models import MiddlewareConfig
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
        ChatMessage, LLMChatResponse, LLMCompletionResponse,
    )
except ImportError:
    LLMProviderManager = type("LLMProviderManager", (), {})
    ChatMessage = dict; LLMChatResponse = dict; LLMCompletionResponse = dict

try:
    from genie_tooling.command_processors.manager import CommandProcessorManager
    from genie_tooling.command_processors.types import CommandProcessorResponse
except ImportError:
    CommandProcessorManager = type("CommandProcessorManager", (), {})
    CommandProcessorResponse = dict

class MockKeyProviderForGenie(KeyProvider, CorePluginType):
    _plugin_id_value: str
    _description_value: str

    def __init__(self, keys: Dict[str, Any] = None):
        self._plugin_id_value = "mock_genie_key_provider_v1"
        self._description_value = "Mock KeyProvider for Genie tests."
        self._keys = keys or {}
        self.setup_called_with_config: Optional[Dict[str, Any]] = None
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
    return MiddlewareConfig(
        default_llm_provider_id="default_llm", default_command_processor_id="default_cmd_proc",
        default_tool_lookup_provider_id="default_lookup", default_tool_indexing_formatter_id="default_formatter",
        default_rag_retriever_id="default_retriever_from_config",
        llm_provider_configurations={"default_llm": {"model": "test"}},
        command_processor_configurations={"default_cmd_proc": {}}, plugin_dev_dirs=[],
    )

@pytest.fixture
async def mock_key_provider_instance_facade() -> MockKeyProviderForGenie:
    provider = MockKeyProviderForGenie({"TEST_KEY": "test_value"})
    await provider.setup()
    return provider

@pytest.fixture
def mock_genie_dependencies(mocker, mock_middleware_config_facade: MiddlewareConfig, mock_key_provider_instance_facade: MockKeyProviderForGenie): # mock_key_provider_instance_facade is a coroutine here
    deps = { "PluginManager_cls": mocker.patch("genie_tooling.genie.PluginManager"), "ToolManager_cls": mocker.patch("genie_tooling.genie.ToolManager"), "ToolInvoker_cls": mocker.patch("genie_tooling.genie.ToolInvoker"), "RAGManager_cls": mocker.patch("genie_tooling.genie.RAGManager"), "ToolLookupService_cls": mocker.patch("genie_tooling.genie.ToolLookupService"), "LLMProviderManager_cls": mocker.patch("genie_tooling.genie.LLMProviderManager"), "CommandProcessorManager_cls": mocker.patch("genie_tooling.genie.CommandProcessorManager"), "LLMInterface_cls": mocker.patch("genie_tooling.genie.LLMInterface"), "RAGInterface_cls": mocker.patch("genie_tooling.genie.RAGInterface"), }
    deps["pm_inst"] = AsyncMock(spec=PluginManager); deps["pm_inst"].list_discovered_plugin_classes.return_value = {}; deps["pm_inst"].get_plugin_instance = AsyncMock(return_value=None); deps["pm_inst"]._discovered_plugin_classes = {}; deps["pm_inst"]._plugin_instances = {}; deps["pm_inst"].discover_plugins = AsyncMock(); deps["pm_inst"].teardown_all_plugins = AsyncMock(); deps["PluginManager_cls"].return_value = deps["pm_inst"]
    deps["tm_inst"] = AsyncMock(spec=ToolManager); deps["ToolManager_cls"].return_value = deps["tm_inst"]; deps["ti_inst"] = AsyncMock(spec=ToolInvoker); deps["ToolInvoker_cls"].return_value = deps["ti_inst"]; deps["ragm_inst"] = AsyncMock(spec=RAGManager); deps["ragm_inst"].retrieve_from_query = AsyncMock(); deps["RAGManager_cls"].return_value = deps["ragm_inst"]; deps["tls_inst"] = AsyncMock(spec=ToolLookupService); deps["ToolLookupService_cls"].return_value = deps["tls_inst"]; deps["llmpm_inst"] = AsyncMock(spec=LLMProviderManager); deps["llmpm_inst"].teardown = AsyncMock(); deps["llmpm_inst"].get_llm_provider = AsyncMock(); deps["LLMProviderManager_cls"].return_value = deps["llmpm_inst"]; deps["cpm_inst"] = AsyncMock(spec=CommandProcessorManager); deps["cpm_inst"].teardown = AsyncMock(); deps["cpm_inst"].get_command_processor = AsyncMock(); deps["CommandProcessorManager_cls"].return_value = deps["cpm_inst"]
    # Note: mock_key_provider_instance_facade is a coroutine; if used here, it needs await.
    # However, these are class mocks, so the instance is passed later.
    mock_llm_interface_instance = LLMInterface(llm_provider_manager=deps["llmpm_inst"], default_provider_id=mock_middleware_config_facade.default_llm_provider_id)
    mock_llm_interface_instance.generate = AsyncMock(wraps=mock_llm_interface_instance.generate); mock_llm_interface_instance.chat = AsyncMock(wraps=mock_llm_interface_instance.chat); deps["LLMInterface_cls"].return_value = mock_llm_interface_instance; deps["llmi_inst"] = mock_llm_interface_instance
    # RAGInterface constructor expects a resolved KeyProvider, not a coroutine.
    # This will be handled by awaiting the fixture in fully_mocked_genie.
    # For the direct mock setup, we'd need to resolve it if RAGInterface was created here.
    # Let's assume fully_mocked_genie handles the resolution.
    # If RAGInterface was created here: resolved_kp = await mock_key_provider_instance_facade
    mock_rag_interface_instance = RAGInterface(rag_manager=deps["ragm_inst"], config=mock_middleware_config_facade, key_provider=AsyncMock(spec=KeyProvider)) # Pass a mock KP for now
    mock_rag_interface_instance.search = AsyncMock(wraps=mock_rag_interface_instance.search); mock_rag_interface_instance.index_directory = AsyncMock(wraps=mock_rag_interface_instance.index_directory); mock_rag_interface_instance.index_web_page = AsyncMock(wraps=mock_rag_interface_instance.index_web_page); deps["RAGInterface_cls"].return_value = mock_rag_interface_instance; deps["ragi_inst"] = mock_rag_interface_instance
    return deps

@pytest.fixture
async def fully_mocked_genie(mock_genie_dependencies, mock_middleware_config_facade: MiddlewareConfig, mock_key_provider_instance_facade: MockKeyProviderForGenie) -> Genie:
    created_pm_instance = mock_genie_dependencies["pm_inst"]
    resolved_kp_instance = await mock_key_provider_instance_facade # Await the async fixture
    async def get_plugin_instance_for_create(plugin_id, config=None, **kwargs):
        if plugin_id == resolved_kp_instance.plugin_id: return resolved_kp_instance
        return AsyncMock(name=f"plugin_mock_for_{plugin_id}")
    created_pm_instance.get_plugin_instance.side_effect = get_plugin_instance_for_create
    # Update RAGInterface mock if it was created with a placeholder key_provider
    mock_rag_interface_instance = RAGInterface(rag_manager=mock_genie_dependencies["ragm_inst"], config=mock_middleware_config_facade, key_provider=resolved_kp_instance)
    mock_rag_interface_instance.search = AsyncMock(wraps=mock_rag_interface_instance.search); mock_rag_interface_instance.index_directory = AsyncMock(wraps=mock_rag_interface_instance.index_directory); mock_rag_interface_instance.index_web_page = AsyncMock(wraps=mock_rag_interface_instance.index_web_page); mock_genie_dependencies["RAGInterface_cls"].return_value = mock_rag_interface_instance; mock_genie_dependencies["ragi_inst"] = mock_rag_interface_instance

    genie_instance = await Genie.create(config=mock_middleware_config_facade, key_provider_instance=resolved_kp_instance)
    return genie_instance


@pytest.mark.asyncio
class TestGenieCreate:
    async def test_create_with_key_provider_instance(self, mock_genie_dependencies, mock_middleware_config_facade: MiddlewareConfig, mock_key_provider_instance_facade: MockKeyProviderForGenie):
        pm_mock = mock_genie_dependencies["pm_inst"]
        resolved_kp_instance = await mock_key_provider_instance_facade # Await the async fixture
        genie = await Genie.create(config=mock_middleware_config_facade, key_provider_instance=resolved_kp_instance)
        assert isinstance(genie, Genie); mock_genie_dependencies["PluginManager_cls"].assert_called_once_with(plugin_dev_dirs=mock_middleware_config_facade.plugin_dev_dirs)
        pm_mock.discover_plugins.assert_awaited_once(); assert genie._key_provider is resolved_kp_instance
        assert resolved_kp_instance.plugin_id in pm_mock._discovered_plugin_classes; assert pm_mock._plugin_instances[resolved_kp_instance.plugin_id] is resolved_kp_instance
        mock_genie_dependencies["ToolManager_cls"].assert_called_once_with(plugin_manager=pm_mock)
        mock_genie_dependencies["ToolInvoker_cls"].assert_called_once_with(tool_manager=mock_genie_dependencies["tm_inst"], plugin_manager=pm_mock)
        mock_genie_dependencies["RAGManager_cls"].assert_called_once_with(plugin_manager=pm_mock)
        mock_genie_dependencies["ToolLookupService_cls"].assert_called_once_with(tool_manager=mock_genie_dependencies["tm_inst"], plugin_manager=pm_mock, default_provider_id=mock_middleware_config_facade.default_tool_lookup_provider_id, default_indexing_formatter_id=mock_middleware_config_facade.default_tool_indexing_formatter_id)
        mock_genie_dependencies["LLMProviderManager_cls"].assert_called_once_with(pm_mock, resolved_kp_instance, mock_middleware_config_facade)
        mock_genie_dependencies["CommandProcessorManager_cls"].assert_called_once_with(pm_mock, resolved_kp_instance, mock_middleware_config_facade)
        mock_genie_dependencies["LLMInterface_cls"].assert_called_once_with(mock_genie_dependencies["llmpm_inst"], mock_middleware_config_facade.default_llm_provider_id)
        rag_interface_call_args = mock_genie_dependencies["RAGInterface_cls"].call_args; assert rag_interface_call_args is not None
        assert rag_interface_call_args[0][0] is mock_genie_dependencies["ragm_inst"]; assert rag_interface_call_args[0][1] is mock_middleware_config_facade; assert rag_interface_call_args[0][2] is resolved_kp_instance
        assert genie.llm is mock_genie_dependencies["llmi_inst"]; assert genie.rag is mock_genie_dependencies["ragi_inst"]

    async def test_create_with_key_provider_id(self, mock_genie_dependencies, mock_middleware_config_facade: MiddlewareConfig):
        pm_mock = mock_genie_dependencies["pm_inst"]; resolved_kp_instance = MockKeyProviderForGenie({"id_key": "id_value"});
        async def get_plugin_instance_side_effect(plugin_id, config=None, **kwargs):
            if plugin_id == "test_kp_id_from_config": return resolved_kp_instance
            return None
        pm_mock.get_plugin_instance.side_effect = get_plugin_instance_side_effect; kp_id = "test_kp_id_from_config";
        genie = await Genie.create(config=mock_middleware_config_facade, key_provider_id=kp_id); pm_mock.get_plugin_instance.assert_any_call(kp_id);
        assert genie._key_provider is resolved_kp_instance; mock_genie_dependencies["LLMProviderManager_cls"].assert_called_once_with(pm_mock, resolved_kp_instance, mock_middleware_config_facade); mock_genie_dependencies["CommandProcessorManager_cls"].assert_called_once_with(pm_mock, resolved_kp_instance, mock_middleware_config_facade)

    async def test_create_no_key_provider_fails(self, mock_middleware_config_facade: MiddlewareConfig):
        with pytest.raises(ValueError, match="Either key_provider_instance or key_provider_id must be provided."):
            await Genie.create(config=mock_middleware_config_facade)

    async def test_create_key_provider_id_resolution_fails(self, mock_genie_dependencies, mock_middleware_config_facade: MiddlewareConfig):
        pm_mock = mock_genie_dependencies["pm_inst"]; pm_mock.get_plugin_instance.return_value = None; kp_id = "non_existent_kp_id";
        with pytest.raises(RuntimeError, match=f"Failed to load KeyProvider with ID '{kp_id}'."):
            await Genie.create(config=mock_middleware_config_facade, key_provider_id=kp_id)

@pytest.mark.asyncio
async def test_genie_execute_tool(fully_mocked_genie: Genie):
    actual_genie = await fully_mocked_genie
    tool_id = "calculator"; params = {"num1": 1, "num2": 2, "operation": "add"}
    await actual_genie.execute_tool(tool_id, **params)
    actual_genie._tool_invoker.invoke.assert_awaited_once_with(tool_identifier=tool_id, params=params, key_provider=actual_genie._key_provider)

@pytest.mark.asyncio
async def test_genie_run_command_selects_tool_and_executes(fully_mocked_genie: Genie):
    actual_genie = await fully_mocked_genie
    command = "Calculate 2 plus 3"; processor_id = actual_genie._config.default_command_processor_id
    mock_processor_plugin = AsyncMock(); cmd_proc_response_success: CommandProcessorResponse = {"chosen_tool_id": "calculator_tool", "extracted_params": {"num1": 2, "num2": 3, "operation": "add"}, "llm_thought_process": "Understood calculation.", "error": None}; mock_processor_plugin.process_command = AsyncMock(return_value=cmd_proc_response_success)
    actual_genie._command_processor_manager.get_command_processor.return_value = mock_processor_plugin; actual_genie.execute_tool = AsyncMock(return_value={"result": 5})
    result = await actual_genie.run_command(command)
    actual_genie._command_processor_manager.get_command_processor.assert_awaited_once_with(processor_id, genie_facade=actual_genie); mock_processor_plugin.process_command.assert_awaited_once_with(command=command, conversation_history=None); actual_genie.execute_tool.assert_awaited_once_with("calculator_tool", num1=2, num2=3, operation="add")
    assert result["tool_result"] == {"result": 5}; assert result["thought_process"] == "Understood calculation."

@pytest.mark.asyncio
async def test_genie_run_command_no_tool_selected(fully_mocked_genie: Genie):
    actual_genie = await fully_mocked_genie
    command = "Just chatting"; cmd_proc_response_no_tool: CommandProcessorResponse = {"chosen_tool_id": None, "extracted_params": None, "llm_thought_process": "No specific tool needed for this.", "error": None}; mock_processor_plugin = AsyncMock(); mock_processor_plugin.process_command = AsyncMock(return_value=cmd_proc_response_no_tool)
    actual_genie._command_processor_manager.get_command_processor.return_value = mock_processor_plugin; actual_genie.execute_tool = AsyncMock()
    result = await actual_genie.run_command(command)
    actual_genie.execute_tool.assert_not_awaited(); assert result["message"] == "No tool selected by command processor."; assert "tool_result" not in result

@pytest.mark.asyncio
async def test_genie_llm_chat(fully_mocked_genie: Genie):
    actual_genie = await fully_mocked_genie
    messages: List[ChatMessage] = [{"role": "user", "content": "Hello LLM"}]
    mock_llm_plugin_instance = AsyncMock(spec=LLMProviderPlugin); mock_llm_plugin_instance.chat = AsyncMock(return_value={"message": {"role":"assistant", "content":"Hi there!"}, "finish_reason":"stop"})
    actual_genie._llm_provider_manager.get_llm_provider.return_value = mock_llm_plugin_instance
    response = await actual_genie.llm.chat(messages)
    actual_genie._llm_provider_manager.get_llm_provider.assert_awaited_once_with(actual_genie._config.default_llm_provider_id); mock_llm_plugin_instance.chat.assert_awaited_once_with(messages); assert response["message"]["content"] == "Hi there!"

@pytest.mark.asyncio
async def test_genie_rag_search(fully_mocked_genie: Genie):
    actual_genie = await fully_mocked_genie
    query = "What is RAG?"; collection_name = "test_collection"; mock_retrieved_chunks: List[RetrievedChunk] = [{"id":"c1", "content":"RAG content", "score":0.9, "metadata":{}}] # type: ignore
    actual_genie._rag_manager.retrieve_from_query.return_value = mock_retrieved_chunks
    results = await actual_genie.rag.search(query, collection_name=collection_name, top_k=1)
    expected_retriever_id = actual_genie._config.default_rag_retriever_id; expected_retriever_config_passed_to_manager = {"embedder_config": {"key_provider": actual_genie._key_provider}, "vector_store_config": {"collection_name": collection_name}}
    actual_genie._rag_manager.retrieve_from_query.assert_awaited_once_with(query_text=query, retriever_id=expected_retriever_id, retriever_config=expected_retriever_config_passed_to_manager, top_k=1); assert results == mock_retrieved_chunks

@pytest.mark.asyncio
async def test_genie_close(fully_mocked_genie: Genie):
    actual_genie = await fully_mocked_genie
    pm_mock_for_assert = actual_genie._plugin_manager; llmpm_mock_for_assert = actual_genie._llm_provider_manager; cpm_mock_for_assert = actual_genie._command_processor_manager
    assert pm_mock_for_assert is not None; assert llmpm_mock_for_assert is not None; assert cpm_mock_for_assert is not None
    await actual_genie.close()
    pm_mock_for_assert.teardown_all_plugins.assert_awaited_once(); llmpm_mock_for_assert.teardown.assert_awaited_once(); cpm_mock_for_assert.teardown.assert_awaited_once()
    assert actual_genie._plugin_manager is None; assert actual_genie._llm_provider_manager is None; assert actual_genie._command_processor_manager is None; assert actual_genie.llm is None; assert actual_genie.rag is None
