### tests/unit/interfaces/test_command_execution_interfaces.py
import uuid
from typing import Any, Dict, List, Optional, cast
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from genie_tooling.command_processors.abc import CommandProcessorPlugin
from genie_tooling.command_processors.manager import CommandProcessorManager
from genie_tooling.command_processors.types import CommandProcessorResponse
from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.config.resolver import PLUGIN_ID_ALIASES, ConfigResolver
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import Plugin as CorePluginType
from genie_tooling.genie import Genie 
from genie_tooling.guardrails.manager import GuardrailManager
from genie_tooling.hitl.manager import HITLManager
from genie_tooling.hitl.types import ApprovalRequest, ApprovalResponse
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
from genie_tooling.llm_providers.manager import LLMProviderManager
from genie_tooling.llm_providers.types import ChatMessage, LLMChatResponse, LLMCompletionResponse
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


# --- Mock KeyProvider (similar to the one in test_genie_facade.py) ---
class MockKeyProviderForCmdExec(KeyProvider, CorePluginType):
    _plugin_id_value: str
    _description_value: str
    def __init__(self, keys: Dict[str, Any] = None, plugin_id="mock_cmd_exec_kp_v1"):
        self._plugin_id_value = plugin_id
        self._description_value = "Mock KeyProvider for command execution tests."
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
    return MiddlewareConfig(
        features=FeatureSettings(
            command_processor="llm_assisted", # A common default
            hitl_approver="cli_hitl_approver" # Enable HITL for some tests
        )
    )

# --- Adapted mock_genie_dependencies from test_genie_facade.py ---
@pytest.fixture
def mock_genie_dependencies_for_cmd_exec(mocker):
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
            env_kp_mock = MockKeyProviderForCmdExec(plugin_id=PLUGIN_ID_ALIASES["env_keys"])
            await env_kp_mock.setup(config)
            return env_kp_mock
        specific_kp_mock = MockKeyProviderForCmdExec(plugin_id=plugin_id)
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

    # Mock instances for managers
    deps["tm_inst"] = AsyncMock(spec=ToolManager); deps["ToolManager_cls"].return_value = deps["tm_inst"]
    deps["ti_inst"] = AsyncMock(spec=ToolInvoker); deps["ToolInvoker_cls"].return_value = deps["ti_inst"]
    deps["ragm_inst"] = AsyncMock(spec=RAGManager); deps["RAGManager_cls"].return_value = deps["ragm_inst"]
    deps["tls_inst"] = AsyncMock(spec=ToolLookupService); deps["ToolLookupService_cls"].return_value = deps["tls_inst"]
    deps["llmpm_inst"] = AsyncMock(spec=LLMProviderManager); deps["LLMProviderManager_cls"].return_value = deps["llmpm_inst"]
    deps["cpm_inst"] = AsyncMock(spec=CommandProcessorManager); deps["CommandProcessorManager_cls"].return_value = deps["cpm_inst"]
    deps["tracing_m_inst"] = AsyncMock(spec=InteractionTracingManager); deps["InteractionTracingManager_cls"].return_value = deps["tracing_m_inst"]
    deps["hitl_m_inst"] = AsyncMock(spec=HITLManager); deps["HITLManager_cls"].return_value = deps["hitl_m_inst"]
    deps["token_usage_m_inst"] = AsyncMock(spec=TokenUsageManager); deps["TokenUsageManager_cls"].return_value = deps["token_usage_m_inst"]
    deps["guardrail_m_inst"] = AsyncMock(spec=GuardrailManager); deps["GuardrailManager_cls"].return_value = deps["guardrail_m_inst"]
    deps["prompt_m_inst"] = AsyncMock(spec=PromptManager); deps["PromptManager_cls"].return_value = deps["prompt_m_inst"]
    deps["convo_m_inst"] = AsyncMock(spec=ConversationStateManager); deps["ConversationStateManager_cls"].return_value = deps["convo_m_inst"]
    deps["llm_output_parser_m_inst"] = AsyncMock(spec=LLMOutputParserManager); deps["LLMOutputParserManager_cls"].return_value = deps["llm_output_parser_m_inst"]
    deps["task_queue_m_inst"] = AsyncMock(spec=DistributedTaskQueueManager); deps["DistributedTaskQueueManager_cls"].return_value = deps["task_queue_m_inst"]

    # ConfigResolver mock (simplified from test_genie_facade)
    deps["resolver_inst"] = MagicMock(spec=ConfigResolver)
    def simplified_resolver_side_effect(user_cfg: MiddlewareConfig, key_provider_instance: Optional[KeyProvider] = None) -> MiddlewareConfig:
        resolved = user_cfg.model_copy(deep=True)
        if key_provider_instance and hasattr(key_provider_instance, "plugin_id"):
            resolved.key_provider_id = key_provider_instance.plugin_id
        elif resolved.key_provider_id is None:
            resolved.key_provider_id = PLUGIN_ID_ALIASES["env_keys"]
        # Set default command processor if not set by features, for run_command tests
        if resolved.features.command_processor == "none" and not resolved.default_command_processor_id:
            resolved.default_command_processor_id = "llm_assisted_tool_selection_processor_v1" # A common default
        elif resolved.features.command_processor != "none":
             resolved.default_command_processor_id = PLUGIN_ID_ALIASES.get(
                 f"{resolved.features.command_processor}_cmd_proc",
                 resolved.features.command_processor # Fallback if alias not found
             )
        return resolved
    deps["resolver_inst"].resolve.side_effect = simplified_resolver_side_effect
    deps["ConfigResolver_cls"].return_value = deps["resolver_inst"]
    return deps

@pytest.fixture
async def genie_instance_for_command_tests(
    mock_genie_dependencies_for_cmd_exec: Dict, 
    mock_middleware_config_for_cmd_exec: MiddlewareConfig,
    mock_key_provider_for_cmd_exec_fixt: MockKeyProviderForCmdExec
) -> Genie:
    """
    Creates a real Genie instance but with its internal managers mocked.
    This is for testing Genie's instance methods like execute_tool and run_command.
    """
    # Ensure the specific mocks for command execution are configured on the manager instances
    # that Genie.create will use.
    mock_genie_dependencies_for_cmd_exec["ti_inst"].invoke = AsyncMock(return_value={"result": "tool executed by invoker"})
    mock_genie_dependencies_for_cmd_exec["hitl_m_inst"].request_approval = AsyncMock(
        return_value=ApprovalResponse(request_id="hitl_req_id_cmd_exec", status="approved")
    )
    # Mock for CommandProcessorManager.get_command_processor
    mock_cmd_proc_plugin = AsyncMock(spec=CommandProcessorPlugin)
    mock_cmd_proc_plugin.process_command = AsyncMock(return_value=CommandProcessorResponse(
        chosen_tool_id="test_tool_from_proc",
        extracted_params={"arg1": "val1"},
        llm_thought_process="Processor thought..."
    ))
    mock_genie_dependencies_for_cmd_exec["cpm_inst"].get_command_processor = AsyncMock(return_value=mock_cmd_proc_plugin)

    # Mock for LLMProviderManager.get_llm_provider (needed if LLM-assisted processor is used)
    mock_llm_plugin = AsyncMock(spec=LLMProviderPlugin)
    mock_llm_plugin.chat = AsyncMock(return_value=LLMChatResponse(message=ChatMessage(role="assistant", content="LLM response"))) # type: ignore
    mock_genie_dependencies_for_cmd_exec["llmpm_inst"].get_llm_provider = AsyncMock(return_value=mock_llm_plugin)


    genie_instance = await Genie.create(
        config=mock_middleware_config_for_cmd_exec,
        key_provider_instance=mock_key_provider_for_cmd_exec_fixt
    )
    # The interfaces are set up by Genie.create using the mocked managers
    return genie_instance


@pytest.mark.asyncio
class TestGenieExecuteTool:
    """Tests for genie.execute_tool()"""

    async def test_execute_tool_success(
        self, genie_instance_for_command_tests: Genie 
    ):
        """Test successful tool execution."""
        genie = await genie_instance_for_command_tests 
        tool_id = "calculator"
        params = {"num1": 5, "num2": 3, "operation": "add"}

        result = await genie.execute_tool(tool_id, **params)

        assert result == {"result": "tool executed by invoker"}
        genie._tool_invoker.invoke.assert_awaited_once() # type: ignore
        call_kwargs = genie._tool_invoker.invoke.call_args.kwargs # type: ignore
        assert call_kwargs["tool_identifier"] == tool_id
        assert call_kwargs["params"] == params
        assert call_kwargs["key_provider"] is genie._key_provider
        assert isinstance(call_kwargs["invoker_config"]["correlation_id"], str)
        genie._tracing_manager.trace_event.assert_any_call("genie.execute_tool.start", ANY, "Genie", ANY) # type: ignore
        genie._tracing_manager.trace_event.assert_any_call("genie.execute_tool.success", ANY, "Genie", ANY) # type: ignore

    async def test_execute_tool_invoker_not_initialized(self, genie_instance_for_command_tests: Genie):
        """Test error if ToolInvoker is not initialized."""
        genie = await genie_instance_for_command_tests 
        genie._tool_invoker = None 
        with pytest.raises(RuntimeError, match="ToolInvoker not initialized."):
            await genie.execute_tool("any_tool")
        # The "start" trace event is NOT called if _tool_invoker is None, as the check is before the trace.
        # So, we should not assert for it here.
        # If there was a higher-level try/except in the calling code that *then* traced an error,
        # that would be different. But for this unit test, the start trace is skipped.

    async def test_execute_tool_key_provider_not_initialized(self, genie_instance_for_command_tests: Genie):
        """Test error if KeyProvider is not initialized."""
        genie = await genie_instance_for_command_tests 
        genie._key_provider = None 
        with pytest.raises(RuntimeError, match="KeyProvider not initialized."):
            await genie.execute_tool("any_tool")

    async def test_execute_tool_invoker_raises_exception(
        self, genie_instance_for_command_tests: Genie
    ):
        """Test handling of exceptions raised by the ToolInvoker."""
        genie = await genie_instance_for_command_tests 
        genie._tool_invoker.invoke.side_effect = ValueError("Invoker boom!") # type: ignore

        with pytest.raises(ValueError, match="Invoker boom!"):
            await genie.execute_tool("error_tool")
        genie._tracing_manager.trace_event.assert_any_call("genie.execute_tool.error", ANY, "Genie", ANY) # type: ignore


@pytest.mark.asyncio
class TestGenieRunCommand:
    """Tests for genie.run_command()"""

    async def test_run_command_success_with_tool_execution_and_hitl(
        self, genie_instance_for_command_tests: Genie
    ):
        """Test successful command processing, HITL approval, and tool execution."""
        genie = await genie_instance_for_command_tests 
        
        command = "do something with val1"
        result = await genie.run_command(command)

        assert result["tool_result"] == {"result": "tool executed by invoker"}
        assert result["thought_process"] == "Processor thought..."
        genie._command_processor_manager.get_command_processor.assert_awaited_once_with( # type: ignore
            genie._config.default_command_processor_id, genie_facade=genie
        )
        
        mock_processor_plugin_instance = genie._command_processor_manager.get_command_processor.return_value # type: ignore
        mock_processor_plugin_instance.process_command.assert_awaited_once_with(command, None)

        genie._hitl_manager.request_approval.assert_awaited_once() # type: ignore
        genie._tool_invoker.invoke.assert_awaited_once_with( # type: ignore
            tool_identifier="test_tool_from_proc",
            params={"arg1": "val1"},
            key_provider=genie._key_provider,
            invoker_config=ANY 
        )
        genie._tracing_manager.trace_event.assert_any_call("genie.run_command.hitl_response", {"status": "approved", "reason": None}, "Genie", ANY) # type: ignore

    async def test_run_command_hitl_denied(
        self, genie_instance_for_command_tests: Genie
    ):
        """Test scenario where HITL denies tool execution."""
        genie = await genie_instance_for_command_tests 
        genie._hitl_manager.request_approval.return_value = ApprovalResponse( # type: ignore
            request_id="denied_req_cmd_exec", status="denied", reason="User denied test"
        )
        
        genie._tool_invoker.invoke.reset_mock() # type: ignore

        result = await genie.run_command("do something risky")

        assert "Tool execution denied by HITL: User denied test" in result["error"]
        assert "hitl_decision" in result
        assert result["hitl_decision"]["status"] == "denied"
        genie._tool_invoker.invoke.assert_not_called() # type: ignore

    async def test_run_command_no_tool_selected_by_processor(
        self, genie_instance_for_command_tests: Genie
    ):
        """Test when the command processor does not select a tool."""
        genie = await genie_instance_for_command_tests 
        mock_processor_plugin_instance = genie._command_processor_manager.get_command_processor.return_value # type: ignore
        mock_processor_plugin_instance.process_command.return_value = CommandProcessorResponse(
            chosen_tool_id=None, llm_thought_process="No tool needed for this."
        )
        genie._tool_invoker.invoke.reset_mock() # type: ignore

        result = await genie.run_command("just chatting")

        assert result["message"] == "No tool selected by command processor."
        assert result["thought_process"] == "No tool needed for this."
        genie._tool_invoker.invoke.assert_not_called() # type: ignore
        genie._tracing_manager.trace_event.assert_any_call("genie.run_command.processor_result", {"chosen_tool_id": None, "has_error": False}, "Genie", ANY) # type: ignore

    async def test_run_command_processor_returns_error(
        self, genie_instance_for_command_tests: Genie
    ):
        """Test when the command processor itself returns an error."""
        genie = await genie_instance_for_command_tests 
        mock_processor_plugin_instance = genie._command_processor_manager.get_command_processor.return_value # type: ignore
        mock_processor_plugin_instance.process_command.return_value = CommandProcessorResponse(
            error="Processor internal failure", llm_thought_process="Thinking failed."
        )

        result = await genie.run_command("failing command")

        assert result["error"] == "Processor internal failure"
        assert result["thought_process"] == "Thinking failed."
        genie._tracing_manager.trace_event.assert_any_call("genie.run_command.processor_result", {"chosen_tool_id": None, "has_error": True}, "Genie", ANY) # type: ignore

    async def test_run_command_processor_not_found(
        self, genie_instance_for_command_tests: Genie
    ):
        """Test when the specified command processor plugin is not found."""
        genie = await genie_instance_for_command_tests 
        genie._command_processor_manager.get_command_processor.return_value = None # type: ignore

        result = await genie.run_command("any command", processor_id="unknown_proc_cmd_exec")
        assert result["error"] == "CommandProcessor 'unknown_proc_cmd_exec' not found."
        genie._tracing_manager.trace_event.assert_any_call("genie.run_command.error", {"error": "ProcessorNotFound", "processor_id": "unknown_proc_cmd_exec"}, "Genie", ANY) # type: ignore

    async def test_run_command_no_processor_configured(self, genie_instance_for_command_tests: Genie):
        """Test when no default or specified command processor is configured."""
        genie = await genie_instance_for_command_tests 
        genie._config.default_command_processor_id = None 

        result = await genie.run_command("any command")
        assert result["error"] == "No command processor configured."
        genie._tracing_manager.trace_event.assert_any_call("genie.run_command.error", {"error": "NoProcessorConfigured"}, "Genie", ANY) # type: ignore

    async def test_run_command_unexpected_exception(
        self, genie_instance_for_command_tests: Genie
    ):
        """Test handling of unexpected exceptions during command processing."""
        genie = await genie_instance_for_command_tests 
        genie._command_processor_manager.get_command_processor.side_effect = TypeError("Unexpected crash in get_processor") # type: ignore

        result = await genie.run_command("crash test")
        assert "Unexpected error in run_command: Unexpected crash in get_processor" in result["error"]
        assert isinstance(result["raw_exception"], TypeError)
        genie._tracing_manager.trace_event.assert_any_call("genie.run_command.error", {"error": "Unexpected crash in get_processor", "type": "TypeError"}, "Genie", ANY) # type: ignore