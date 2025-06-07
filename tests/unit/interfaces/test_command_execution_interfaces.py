### tests/unit/interfaces/test_command_execution_interfaces.py
from typing import Any, Dict, Optional
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from genie_tooling.command_processors.abc import CommandProcessorPlugin
from genie_tooling.command_processors.manager import CommandProcessorManager
from genie_tooling.command_processors.types import CommandProcessorResponse
from genie_tooling.config.features import FeatureSettings  # Added import
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.resolver import (  # Added PLUGIN_ID_ALIASES
    PLUGIN_ID_ALIASES,
    ConfigResolver,
)
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import Plugin as CorePluginType
from genie_tooling.genie import Genie
from genie_tooling.guardrails.manager import GuardrailManager
from genie_tooling.hitl.manager import HITLManager
from genie_tooling.hitl.types import ApprovalResponse
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


class MockKeyProviderForCmdExec(KeyProvider, CorePluginType):
    _plugin_id_value: str = "mock_cmd_exec_kp_v1"
    _description_value: str = "Mock KeyProvider for command execution tests."
    def __init__(self, keys: Dict[str, Any] = None):
        self._keys = keys or {}
    @property
    def plugin_id(self) -> str: return self._plugin_id_value
    @property
    def description(self) -> str: return self._description_value
    async def get_key(self, key_name: str) -> Optional[str]: return self._keys.get(key_name)
    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass
    async def teardown(self) -> None: pass

@pytest.fixture
async def mock_key_provider_for_cmd_exec_fixt() -> MockKeyProviderForCmdExec:
    provider = MockKeyProviderForCmdExec({"TEST_KEY_CMD_EXEC": "val_cmd_exec"})
    await provider.setup()
    return provider

@pytest.fixture
def mock_middleware_config_for_cmd_exec() -> MiddlewareConfig:
    # CORRECTED: Set a default command processor via features
    return MiddlewareConfig(
        features=FeatureSettings(command_processor="llm_assisted")
    )

@pytest.fixture
async def genie_instance_for_command_tests(
    mocker,
    mock_middleware_config_for_cmd_exec: MiddlewareConfig,
    mock_key_provider_for_cmd_exec_fixt: MockKeyProviderForCmdExec
) -> Genie:
    kp_instance = mock_key_provider_for_cmd_exec_fixt

    mock_pm_instance = AsyncMock(spec=PluginManager)
    mock_pm_instance.discover_plugins = AsyncMock()
    mock_pm_instance.get_plugin_instance = AsyncMock()

    mock_tm_instance = AsyncMock(spec=ToolManager)
    mock_tm_instance.initialize_tools = AsyncMock()
    mock_tm_instance.get_tool = AsyncMock()
    mock_tm_instance.list_tools = AsyncMock(return_value=[])
    mock_tm_instance.get_formatted_tool_definition = AsyncMock()

    mock_ti_instance = AsyncMock(spec=ToolInvoker)
    mock_ti_instance.invoke = AsyncMock(return_value={"result": "tool executed by invoker"})

    mock_rag_mgr_instance = AsyncMock(spec=RAGManager)
    mock_tls_instance = AsyncMock(spec=ToolLookupService)
    mock_llm_pm_instance = AsyncMock(spec=LLMProviderManager)

    mock_cmd_proc_mgr_instance = AsyncMock(spec=CommandProcessorManager)
    mock_cmd_proc_plugin = AsyncMock(spec=CommandProcessorPlugin)
    mock_cmd_proc_plugin.process_command = AsyncMock(return_value=CommandProcessorResponse(
        chosen_tool_id="test_tool_from_proc",
        extracted_params={"arg1": "val1"},
        llm_thought_process="Processor thought..."
    ))
    # Configure get_command_processor to return our mock plugin
    # The processor_id it will be called with is now derived from features
    # by the ConfigResolver.
    mock_cmd_proc_mgr_instance.get_command_processor = AsyncMock(return_value=mock_cmd_proc_plugin)


    mock_cr_instance = MagicMock(spec=ConfigResolver)
    real_resolver = ConfigResolver()
    resolved_config_for_test = real_resolver.resolve(
        mock_middleware_config_for_cmd_exec, # This now has features.command_processor set
        kp_instance
    )
    # Ensure the resolved config actually has the default_command_processor_id
    assert resolved_config_for_test.default_command_processor_id == PLUGIN_ID_ALIASES["llm_assisted_cmd_proc"]
    mock_cr_instance.resolve.return_value = resolved_config_for_test


    mock_itm_instance = AsyncMock(spec=InteractionTracingManager)
    mock_hitl_mgr_instance = AsyncMock(spec=HITLManager)
    mock_hitl_mgr_instance.is_active = True
    mock_hitl_mgr_instance.request_approval = AsyncMock(
        return_value=ApprovalResponse(request_id="hitl_req_id_cmd_exec", status="approved")
    )

    mock_tum_instance = AsyncMock(spec=TokenUsageManager)
    mock_gm_instance = AsyncMock(spec=GuardrailManager)
    mock_prompt_mgr_instance = AsyncMock(spec=PromptManager)
    mock_csm_instance = AsyncMock(spec=ConversationStateManager)
    mock_llm_op_parser_mgr_instance = AsyncMock(spec=LLMOutputParserManager)
    mock_dtq_mgr_instance = AsyncMock(spec=DistributedTaskQueueManager)
    mock_default_log_adapter_instance = AsyncMock(spec=DefaultLogAdapter)
    mock_default_log_adapter_instance.process_event = AsyncMock()

    with patch("genie_tooling.genie.PluginManager", return_value=mock_pm_instance), \
         patch("genie_tooling.genie.ToolManager", return_value=mock_tm_instance), \
         patch("genie_tooling.genie.ToolInvoker", return_value=mock_ti_instance), \
         patch("genie_tooling.genie.RAGManager", return_value=mock_rag_mgr_instance), \
         patch("genie_tooling.genie.ToolLookupService", return_value=mock_tls_instance), \
         patch("genie_tooling.genie.LLMProviderManager", return_value=mock_llm_pm_instance), \
         patch("genie_tooling.genie.CommandProcessorManager", return_value=mock_cmd_proc_mgr_instance), \
         patch("genie_tooling.genie.ConfigResolver", return_value=mock_cr_instance), \
         patch("genie_tooling.genie.InteractionTracingManager", return_value=mock_itm_instance), \
         patch("genie_tooling.genie.HITLManager", return_value=mock_hitl_mgr_instance), \
         patch("genie_tooling.genie.TokenUsageManager", return_value=mock_tum_instance), \
         patch("genie_tooling.genie.GuardrailManager", return_value=mock_gm_instance), \
         patch("genie_tooling.genie.PromptManager", return_value=mock_prompt_mgr_instance), \
         patch("genie_tooling.genie.ConversationStateManager", return_value=mock_csm_instance), \
         patch("genie_tooling.genie.LLMOutputParserManager", return_value=mock_llm_op_parser_mgr_instance), \
         patch("genie_tooling.genie.DefaultLogAdapter", return_value=mock_default_log_adapter_instance), \
         patch("genie_tooling.genie.DistributedTaskQueueManager", return_value=mock_dtq_mgr_instance):

        genie_instance = await Genie.create(
            config=mock_middleware_config_for_cmd_exec,
            key_provider_instance=kp_instance
        )
        return genie_instance

@pytest.mark.asyncio
async def test_run_command_success_with_tool_execution_and_hitl(genie_instance_for_command_tests: Genie):
    genie = await genie_instance_for_command_tests
    command = "do something with val1"
    result = await genie.run_command(command)

    assert result.get("tool_result") == {"result": "tool executed by invoker"}
    assert result.get("thought_process") == "Processor thought..."
    # The CommandProcessorManager's get_command_processor will be called with the default ID
    # which is now set by FeatureSettings -> ConfigResolver.
    expected_processor_id = PLUGIN_ID_ALIASES["llm_assisted_cmd_proc"]
    genie._command_processor_manager.get_command_processor.assert_awaited_once_with(expected_processor_id, genie_facade=genie)
    genie._hitl_manager.request_approval.assert_awaited_once()
    genie._tool_invoker.invoke.assert_awaited_once_with(
        tool_identifier="test_tool_from_proc",
        params={"arg1": "val1"},
        key_provider=genie._key_provider,
        invoker_config=ANY
    )

@pytest.mark.asyncio
async def test_run_command_hitl_denied(genie_instance_for_command_tests: Genie):
    genie = await genie_instance_for_command_tests
    genie._hitl_manager.request_approval.return_value = ApprovalResponse(
        request_id="denied_req_cmd_exec", status="denied", reason="User denied test"
    )
    result = await genie.run_command("do something risky")
    assert "Tool execution denied by HITL: User denied test" in result.get("error", "")
    genie._tool_invoker.invoke.assert_not_called()
