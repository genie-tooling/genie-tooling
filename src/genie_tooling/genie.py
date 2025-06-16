# src/genie_tooling/genie.py
from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    cast,
)

from .command_processors.types import CommandProcessorResponse
from .config.models import MiddlewareConfig
from .config.resolver import PLUGIN_ID_ALIASES, ConfigResolver
from .conversation.impl.manager import ConversationStateManager
from .core.plugin_manager import PluginManager
from .core.types import Plugin as CorePluginType
from .guardrails.manager import GuardrailManager
from .hitl.manager import HITLManager
from .hitl.types import ApprovalRequest
from .interfaces import (
    ConversationInterface,
    HITLInterface,
    LLMInterface,
    ObservabilityInterface,
    PromptInterface,
    RAGInterface,
    TaskQueueInterface,
    UsageTrackingInterface,
)
from .invocation.invoker import ToolInvoker
from .llm_providers.types import ChatMessage
from .log_adapters.abc import LogAdapter as LogAdapterPlugin
from .log_adapters.impl.default_adapter import DefaultLogAdapter
from .lookup.service import ToolLookupService
from .observability.manager import InteractionTracingManager
from .prompts.llm_output_parsers.manager import LLMOutputParserManager
from .prompts.manager import PromptManager
from .rag.manager import RAGManager
from .security.key_provider import KeyProvider
from .task_queues.manager import DistributedTaskQueueManager
from .token_usage.manager import TokenUsageManager
from .tools.abc import Tool as ToolPlugin
from .tools.manager import ToolManager

try:
    from .llm_providers.manager import LLMProviderManager
except ImportError:
    LLMProviderManager = type("LLMProviderManager", (), {})
try:
    from .command_processors.manager import CommandProcessorManager
except ImportError:
    CommandProcessorManager = type("CommandProcessorManager", (), {})

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

class FunctionToolWrapper(ToolPlugin):
    _func: Callable
    _metadata: Dict[str, Any]
    _is_async: bool
    _sig: inspect.Signature

    _sig: inspect.Signature

    @property
    def plugin_id(self) -> str:
        return self._metadata.get("identifier", self._func.__name__)
    @property
    def identifier(self) -> str:
        return self._metadata.get("identifier", self._func.__name__)
    def __init__(self, func: Callable, metadata: Dict[str, Any]):
        if not callable(func):
            raise TypeError("Wrapped object must be callable.")
        self._func = func
        self._metadata = metadata
        self._is_async = inspect.iscoroutinefunction(func)
        self._sig = inspect.signature(func)
        if "identifier" not in self._metadata or not self._metadata["identifier"]:
            self._metadata["identifier"] = self._func.__name__
        if "name" not in self._metadata or not self._metadata["name"]:
            self._metadata["name"] = self._func.__name__.replace("_", " ").title()
    async def get_metadata(self) -> Dict[str, Any]:
        return self._metadata

    async def execute(self, params: Dict[str, Any], key_provider: KeyProvider, context: Optional[Dict[str, Any]] = None) -> Any:
        tool_context = context or {}
        is_strict = tool_context.get("strict_tool_parameters", False)

        final_kwargs: Dict[str, Any]
        if is_strict:
            # Strict mode: pass all params and let Python raise TypeError on mismatch.
            final_kwargs = params.copy()
        else:
            # Lenient mode (default): filter params to only include what the function signature accepts.
            final_kwargs = {
                k: v for k, v in params.items()
                if k in self._sig.parameters
            }

        # Inject framework-provided context if the function expects it.
        if "context" in self._sig.parameters:
            final_kwargs["context"] = tool_context
        if "key_provider" in self._sig.parameters:
            final_kwargs["key_provider"] = key_provider

        if self._is_async:
            return await self._func(**final_kwargs)
            return await self._func(**final_kwargs)
        else:
            return await asyncio.get_running_loop().run_in_executor(
                None, functools.partial(self._func, **final_kwargs)
            )

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        pass
    async def teardown(self) -> None:
        pass

class Genie:
    def __init__(
        self, plugin_manager: PluginManager, key_provider: KeyProvider, config: MiddlewareConfig, tool_manager: ToolManager,
        tool_invoker: ToolInvoker, rag_manager: RAGManager, tool_lookup_service: ToolLookupService, llm_provider_manager: LLMProviderManager,
        command_processor_manager: CommandProcessorManager, log_adapter: LogAdapterPlugin, tracing_manager: InteractionTracingManager,
        hitl_manager: HITLManager, token_usage_manager: TokenUsageManager, guardrail_manager: GuardrailManager, prompt_manager: PromptManager,
        conversation_manager: ConversationStateManager, llm_output_parser_manager: LLMOutputParserManager, task_queue_manager: DistributedTaskQueueManager,
        llm_interface: LLMInterface, rag_interface: RAGInterface, observability_interface: ObservabilityInterface, hitl_interface: HITLInterface,
        usage_tracking_interface: UsageTrackingInterface, prompt_interface: PromptInterface, conversation_interface: ConversationInterface,
        task_queue_interface: TaskQueueInterface
    ):
        self._plugin_manager = plugin_manager
        self._key_provider = key_provider
        self._config = config
        self._tool_manager = tool_manager
        self._tool_invoker = tool_invoker
        self._rag_manager = rag_manager
        self._tool_lookup_service = tool_lookup_service
        self._llm_provider_manager = llm_provider_manager
        self._command_processor_manager = command_processor_manager
        self._log_adapter = log_adapter
        self._tracing_manager = tracing_manager
        self._hitl_manager = hitl_manager
        self._token_usage_manager = token_usage_manager
        self._guardrail_manager = guardrail_manager
        self._prompt_manager = prompt_manager
        self._conversation_manager = conversation_manager
        self._llm_output_parser_manager = llm_output_parser_manager
        self._task_queue_manager = task_queue_manager
        self.llm = llm_interface
        self.rag = rag_interface
        self.observability = observability_interface
        self.human_in_loop = hitl_interface
        self.usage = usage_tracking_interface
        self.prompts = prompt_interface
        self.conversation = conversation_interface
        self.task_queue = task_queue_interface
        self._config._genie_instance = self # type: ignore
        asyncio.create_task(self.observability.trace_event("log.info", {"message": "Genie facade initialized with resolved configuration."}, "Genie"))

    @classmethod
    async def create(cls, config: MiddlewareConfig, key_provider_instance: Optional[KeyProvider] = None) -> Genie:
        pm_for_kp_loading = PluginManager(plugin_dev_dirs=config.plugin_dev_dirs)
        await pm_for_kp_loading.discover_plugins()
        actual_key_provider: KeyProvider
        user_kp_id_preference = config.key_provider_id
        if key_provider_instance:
            actual_key_provider = key_provider_instance
            if isinstance(key_provider_instance, CorePluginType):
                kp_instance_plugin_id = getattr(key_provider_instance, "plugin_id", None)
                if kp_instance_plugin_id and kp_instance_plugin_id not in pm_for_kp_loading._plugin_instances:
                    pm_for_kp_loading._plugin_instances[kp_instance_plugin_id] = key_provider_instance
        else:
            kp_id_to_load_alias_or_canonical = user_kp_id_preference or PLUGIN_ID_ALIASES["env_keys"]
            kp_id_canonical_to_load = PLUGIN_ID_ALIASES.get(kp_id_to_load_alias_or_canonical, kp_id_to_load_alias_or_canonical)
            kp_any = await pm_for_kp_loading.get_plugin_instance(kp_id_canonical_to_load)
            if not kp_any or not isinstance(kp_any, KeyProvider):
                raise RuntimeError(f"Failed to load KeyProvider with ID '{kp_id_canonical_to_load}' (resolved from '{user_kp_id_preference}').")
            actual_key_provider = cast(KeyProvider, kp_any)
        logger.info(f"Using KeyProvider: {type(actual_key_provider).__name__} (ID: {getattr(actual_key_provider, 'plugin_id', 'N/A')})")
        resolver = ConfigResolver()
        resolved_config: MiddlewareConfig = resolver.resolve(config, key_provider_instance=actual_key_provider)
        pm = PluginManager(plugin_dev_dirs=resolved_config.plugin_dev_dirs)
        await pm.discover_plugins()
        if isinstance(actual_key_provider, CorePluginType):
            kp_main_pm_plugin_id = getattr(actual_key_provider, "plugin_id", None)
            if kp_main_pm_plugin_id and (kp_main_pm_plugin_id not in pm._plugin_instances or pm._plugin_instances[kp_main_pm_plugin_id] is not actual_key_provider):
                pm._plugin_instances[kp_main_pm_plugin_id] = actual_key_provider
                if hasattr(actual_key_provider, "__class__"):
                     pm._discovered_plugin_classes[kp_main_pm_plugin_id] = type(actual_key_provider)
        default_log_adapter_id = resolved_config.default_log_adapter_id or DefaultLogAdapter.plugin_id
        log_adapter_config_from_mw = resolved_config.log_adapter_configurations.get(default_log_adapter_id, {})
        log_adapter_specific_config_with_pm = {**log_adapter_config_from_mw, "plugin_manager": pm}
        log_adapter_instance_any = await pm.get_plugin_instance(default_log_adapter_id, config=log_adapter_specific_config_with_pm)
        log_adapter_instance: LogAdapterPlugin
        if log_adapter_instance_any and isinstance(log_adapter_instance_any, LogAdapterPlugin):
            log_adapter_instance = cast(LogAdapterPlugin, log_adapter_instance_any)
        else:
            logger.warning(f"Failed to load configured LogAdapter '{default_log_adapter_id}'. Falling back to DefaultLogAdapter.")
            log_adapter_instance = DefaultLogAdapter()
            await log_adapter_instance.setup({"plugin_manager": pm})
        logger.info(f"Using LogAdapter: {log_adapter_instance.plugin_id}")

        default_tracer_id_from_config = resolved_config.default_observability_tracer_id
        default_tracer_ids_list_for_manager: Optional[List[str]] = [default_tracer_id_from_config] if default_tracer_id_from_config else None
        tracing_manager = InteractionTracingManager(pm, default_tracer_ids_list_for_manager, resolved_config.observability_tracer_configurations, log_adapter_instance=log_adapter_instance)

        tool_manager = ToolManager(plugin_manager=pm, tracing_manager=tracing_manager)
        await tool_manager.initialize_tools(tool_configurations=resolved_config.tool_configurations)
        tool_invoker = ToolInvoker(tool_manager=tool_manager, plugin_manager=pm)
        rag_manager = RAGManager(plugin_manager=pm, tracing_manager=tracing_manager)
        tool_lookup_service = ToolLookupService(tool_manager=tool_manager, plugin_manager=pm, default_provider_id=resolved_config.default_tool_lookup_provider_id, default_indexing_formatter_id=resolved_config.default_tool_indexing_formatter_id, tracing_manager=tracing_manager)
        hitl_manager = HITLManager(pm, resolved_config.default_hitl_approver_id, resolved_config.hitl_approver_configurations)
        default_recorder_id_from_config = resolved_config.default_token_usage_recorder_id
        default_recorder_ids_list_for_manager: Optional[List[str]] = [default_recorder_id_from_config] if default_recorder_id_from_config else None
        token_usage_manager = TokenUsageManager(pm, default_recorder_ids_list_for_manager, resolved_config.token_usage_recorder_configurations)
        guardrail_manager = GuardrailManager(pm, resolved_config.default_input_guardrail_ids, resolved_config.default_output_guardrail_ids, resolved_config.default_tool_usage_guardrail_ids, resolved_config.guardrail_configurations)
        prompt_manager = PromptManager(pm, resolved_config.default_prompt_registry_id, resolved_config.default_prompt_template_plugin_id, resolved_config.prompt_registry_configurations, resolved_config.prompt_template_configurations, tracing_manager=tracing_manager)
        conversation_manager = ConversationStateManager(pm, resolved_config.default_conversation_state_provider_id, resolved_config.conversation_state_provider_configurations, tracing_manager=tracing_manager)
        llm_output_parser_manager = LLMOutputParserManager(pm, resolved_config.default_llm_output_parser_id, resolved_config.llm_output_parser_configurations, tracing_manager=tracing_manager)
        task_queue_manager = DistributedTaskQueueManager(pm, resolved_config.default_distributed_task_queue_id, resolved_config.distributed_task_queue_configurations, tracing_manager=tracing_manager)
        llm_provider_manager = LLMProviderManager(pm, actual_key_provider, resolved_config, token_usage_manager)
        command_processor_manager = CommandProcessorManager(pm, actual_key_provider, resolved_config)

        llm_interface = LLMInterface(llm_provider_manager, resolved_config.default_llm_provider_id, llm_output_parser_manager, tracing_manager, guardrail_manager, token_usage_manager)
        rag_interface = RAGInterface(rag_manager, resolved_config, actual_key_provider, tracing_manager)
        observability_interface = ObservabilityInterface(tracing_manager)
        hitl_interface = HITLInterface(hitl_manager)
        usage_tracking_interface = UsageTrackingInterface(token_usage_manager)
        prompt_interface = PromptInterface(prompt_manager)
        conversation_interface = ConversationInterface(conversation_manager)
        task_queue_interface = TaskQueueInterface(task_queue_manager)

        return cls(plugin_manager=pm, key_provider=actual_key_provider, config=resolved_config, tool_manager=tool_manager, tool_invoker=tool_invoker, rag_manager=rag_manager, tool_lookup_service=tool_lookup_service, llm_provider_manager=llm_provider_manager, command_processor_manager=command_processor_manager, log_adapter=log_adapter_instance, tracing_manager=tracing_manager, hitl_manager=hitl_manager, token_usage_manager=token_usage_manager, guardrail_manager=guardrail_manager, prompt_manager=prompt_manager, conversation_manager=conversation_manager, llm_output_parser_manager=llm_output_parser_manager, task_queue_manager=task_queue_manager, llm_interface=llm_interface, rag_interface=rag_interface, observability_interface=observability_interface, hitl_interface=hitl_interface, usage_tracking_interface=usage_tracking_interface, prompt_interface=prompt_interface, conversation_interface=conversation_interface, task_queue_interface=task_queue_interface)

    async def register_tool_functions(self, functions: List[Callable]) -> None:
        if not self._tool_manager:
            await self.observability.trace_event("log.error", {"message": "Genie: ToolManager not initialized."}, "Genie")
            return
        corr_id = str(uuid.uuid4())
        await self.observability.trace_event("genie.register_tool_functions.start", {"num_functions": len(functions)}, "Genie", corr_id)
        self._tool_manager.register_decorated_tools(functions, self._config.auto_enable_registered_tools)
        if self._tool_lookup_service:
            await self._tool_lookup_service.invalidate_all_indices(correlation_id=corr_id)
        await self.observability.trace_event("genie.register_tool_functions.end", {"registered_count": len(functions)}, "Genie", corr_id)

    async def execute_tool(
        self,
        tool_identifier: str,
        context: Optional[Dict[str, Any]] = None,
        **params: Any
    ) -> Any:
        if not self._tool_invoker:
            raise RuntimeError("ToolInvoker not initialized.")
        if not self._key_provider:
            raise RuntimeError("KeyProvider not initialized.")
        corr_id = str(uuid.uuid4())
        await self.observability.trace_event("genie.execute_tool.start", {"tool_id": tool_identifier, "params": params, "has_user_context": context is not None}, "Genie", corr_id)

        # Enrich context with framework-level info for the tool invocation lifecycle
        context_for_tool_invocation: Dict[str, Any] = {"genie_framework_instance": self}
        if context:
            context_for_tool_invocation.update(context)
        # Add the strict_tool_parameters flag to the context
        context_for_tool_invocation["strict_tool_parameters"] = self._config.strict_tool_parameters

        invoker_strategy_config = {"plugin_manager": self._plugin_manager, "guardrail_manager": self._guardrail_manager, "tracing_manager": self._tracing_manager, "correlation_id": corr_id}
        try:
            result = await self._tool_invoker.invoke(tool_identifier=tool_identifier, params=params, key_provider=self._key_provider, context=context_for_tool_invocation, invoker_config=invoker_strategy_config)
            await self.observability.trace_event("genie.execute_tool.success", {"tool_id": tool_identifier, "result_type": type(result).__name__}, "Genie", corr_id)
            return result
        except Exception as e:
            await self.observability.trace_event("genie.execute_tool.error", {"tool_id": tool_identifier, "error": str(e), "type": type(e).__name__}, "Genie", corr_id)
            raise

    async def run_command(
        self,
        command: str,
        processor_id: Optional[str] = None,
        conversation_history: Optional[List[ChatMessage]] = None,
        context_for_tools: Optional[Dict[str, Any]] = None
    ) -> Any:
        if not self._command_processor_manager:
            return {"error": "CommandProcessorManager not initialized."}
        corr_id = str(uuid.uuid4())
        await self.observability.trace_event("genie.run_command.start", {"command_len": len(command), "processor_id": processor_id or self._config.default_command_processor_id, "has_tool_context": context_for_tools is not None}, "Genie", corr_id)
        target_processor_id = processor_id or self._config.default_command_processor_id
        if not target_processor_id:
            await self.observability.trace_event("genie.run_command.error", {"error": "NoProcessorConfigured"}, "Genie", corr_id)
            return {"error": "No command processor configured."}
        try:
            processor_plugin = await self._command_processor_manager.get_command_processor(target_processor_id, genie_facade=self)
            if not processor_plugin:
                await self.observability.trace_event("genie.run_command.error", {"error": "ProcessorNotFound", "processor_id": target_processor_id}, "Genie", corr_id)
                return {"error": f"CommandProcessor '{target_processor_id}' not found."}
            process_command_kwargs: Dict[str, Any] = {}
            processor_sig = inspect.signature(processor_plugin.process_command)
            if "correlation_id" in processor_sig.parameters:
                process_command_kwargs["correlation_id"] = corr_id
            if "genie_instance" in processor_sig.parameters:
                 process_command_kwargs["genie_instance"] = self
            cmd_proc_response_any: Any = await processor_plugin.process_command(command, conversation_history, **process_command_kwargs)

            # Robustness check for processor response
            if not isinstance(cmd_proc_response_any, dict):
                err_msg = f"Command processor '{target_processor_id}' returned an unexpected type '{type(cmd_proc_response_any).__name__}' instead of a dictionary."
                logger.error(err_msg)
                raise TypeError(err_msg)
            cmd_proc_response: CommandProcessorResponse = cmd_proc_response_any

            error_val, thought_val = cmd_proc_response.get("error"), cmd_proc_response.get("llm_thought_process")
            chosen_tool_id, extracted_params = cmd_proc_response.get("chosen_tool_id"), cmd_proc_response.get("extracted_params")
            final_answer = cmd_proc_response.get("final_answer")
            await self.observability.trace_event("genie.run_command.processor_result", {"chosen_tool_id": chosen_tool_id, "has_error": bool(error_val), "has_final_answer": bool(final_answer)}, "Genie", corr_id)
            if error_val:
                return {"error": error_val, "thought_process": thought_val, "raw_response": cmd_proc_response.get("raw_response")}
            if final_answer:
                return {"final_answer": final_answer, "thought_process": thought_val, "raw_response": cmd_proc_response.get("raw_response")}
            if chosen_tool_id and extracted_params is not None:
                if self._hitl_manager and self._hitl_manager.is_active:
                    approval_req_data: Dict[str, Any] = {"tool_id": chosen_tool_id, "params": extracted_params}
                    hitl_context_data = {"command": command, "correlation_id": corr_id}
                    if thought_val:
                        hitl_context_data["processor_thought"] = thought_val
                    approval_req = ApprovalRequest(request_id=str(uuid.uuid4()), prompt=f"Approve execution of tool '{chosen_tool_id}' with params: {extracted_params}?", data_to_approve=approval_req_data, context=hitl_context_data)
                    await self.observability.trace_event("genie.run_command.hitl_request", {"tool_id": chosen_tool_id, "params": extracted_params}, "Genie", corr_id)
                    approval_resp = await self.human_in_loop.request_approval(approval_req)
                    await self.observability.trace_event("genie.run_command.hitl_response", {"status": approval_resp["status"], "reason": approval_resp.get("reason")}, "Genie", corr_id)
                    if approval_resp["status"] != "approved":
                        return {"error": f"Tool execution denied by HITL: {approval_resp.get('reason', 'No reason')}", "thought_process": thought_val, "hitl_decision": approval_resp}
                tool_result = await self.execute_tool(chosen_tool_id, context=context_for_tools, **extracted_params)
                return {"tool_result": tool_result, "thought_process": thought_val, "raw_response": cmd_proc_response.get("raw_response")}
            return {"message": "No tool selected by command processor.", "thought_process": thought_val, "raw_response": cmd_proc_response.get("raw_response")}
        except Exception as e:
            # Add exc_info=True to the trace event data
            await self.observability.trace_event(
                "genie.run_command.error",
                {"error": str(e), "type": type(e).__name__, "exc_info": True},
                "Genie",
                corr_id,
            )
            return {"error": f"Unexpected error in run_command: {e!s}", "raw_exception": e}

    async def close(self) -> None:
        if not self.observability: # Check if already closed
            logger.warning("Genie.close() called on an already closed instance.")
            return

        await self.observability.trace_event("log.info", {"message": "Genie: Initiating teardown..."}, "Genie")
        await self.observability.trace_event("genie.close.start", {}, "Genie", str(uuid.uuid4()))
        managers_to_teardown = [self._log_adapter, self._tracing_manager, self._hitl_manager, self._token_usage_manager, self._guardrail_manager, self._prompt_manager, self._conversation_manager, self._llm_output_parser_manager, self._task_queue_manager, self._llm_provider_manager, self._command_processor_manager, self._rag_manager, self._tool_lookup_service, self._tool_invoker, self._tool_manager]
        for m in managers_to_teardown:
            if m and hasattr(m, "teardown") and callable(m.teardown):
                try:
                    await m.teardown()
                except Exception as e_td:
                    await self.observability.trace_event("log.error", {"message": f"Error tearing down manager {type(m).__name__}: {e_td}", "exc_info": True}, "Genie")
        if self._plugin_manager:
            await self._plugin_manager.teardown_all_plugins()

        # Call the final trace event *before* nullifying the observability attribute
        await self.observability.trace_event("genie.close.end", {}, "Genie", str(uuid.uuid4()))

        attrs_to_null = ["_plugin_manager", "_key_provider", "_config", "_tool_manager", "_tool_invoker", "_rag_manager", "_tool_lookup_service", "_llm_provider_manager", "_command_processor_manager", "llm", "rag", "_log_adapter", "_tracing_manager", "_hitl_manager", "_token_usage_manager", "_guardrail_manager", "observability", "human_in_loop", "usage", "_prompt_manager", "prompts", "_conversation_manager", "conversation", "_llm_output_parser_manager", "_task_queue_manager", "task_queue"]
        for attr in attrs_to_null:
            if hasattr(self, attr):
                setattr(self, attr, None)
        logger.info("Genie: Teardown complete.")