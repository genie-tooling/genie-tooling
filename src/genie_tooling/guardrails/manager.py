"""GuardrailManager: Orchestrates GuardrailPlugins."""
import logging
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, cast

from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.tools.abc import Tool

from .abc import (
    GuardrailPlugin,
    InputGuardrailPlugin,
    OutputGuardrailPlugin,
    ToolUsageGuardrailPlugin,
)
from .types import GuardrailViolation

if TYPE_CHECKING:
    from genie_tooling.observability.manager import InteractionTracingManager

logger = logging.getLogger(__name__)


def _safe_repr(value: Any, max_chars: int = 500) -> str:
    """Truncated repr for audit previews. Never raises."""
    try:
        s = repr(value)
    except Exception as e:
        s = f"<unrepr-able: {type(value).__name__}: {e}>"
    return s if len(s) <= max_chars else s[:max_chars] + "...<truncated>"

class GuardrailManager:
    def __init__(
        self,
        plugin_manager: PluginManager,
        default_input_guardrail_ids: Optional[List[str]] = None,
        default_output_guardrail_ids: Optional[List[str]] = None,
        default_tool_usage_guardrail_ids: Optional[List[str]] = None,
        guardrail_configurations: Optional[Dict[str, Dict[str, Any]]] = None,
        tracing_manager: Optional["InteractionTracingManager"] = None,
    ):
        self._plugin_manager = plugin_manager
        self._input_guardrail_ids = default_input_guardrail_ids or []
        self._output_guardrail_ids = default_output_guardrail_ids or []
        self._tool_usage_guardrail_ids = default_tool_usage_guardrail_ids or []
        self._guardrail_configurations = guardrail_configurations or {}
        # C3: enable structured audit emission when a guardrail blocks/warns.
        # The TracingManager is optional — if absent, we degrade gracefully
        # to module logging.
        self._tracing_manager = tracing_manager

        self._active_input_guardrails: List[InputGuardrailPlugin] = []
        self._active_output_guardrails: List[OutputGuardrailPlugin] = []
        self._active_tool_usage_guardrails: List[ToolUsageGuardrailPlugin] = []
        self._initialized = False
        logger.info("GuardrailManager initialized.")

    async def _emit_decision(
        self,
        stage: str,
        guardrail_id: str,
        violation: GuardrailViolation,
        context: Optional[Dict[str, Any]],
    ) -> None:
        """Emit a structured guardrail decision record (C3).

        Joinable to a DecisionRecord via ``context["decision_id"]`` when the
        caller (e.g. cqs ContextManager) propagates the audit correlation id.
        """
        decision_id = (context or {}).get("decision_id")
        actor = (context or {}).get("user_identity")
        session_id = (context or {}).get("session_id")
        payload = {
            "stage": stage,  # "input" | "output" | "tool_usage"
            "guardrail_id": guardrail_id,
            "decision": violation.get("action"),
            "reason": violation.get("reason"),
            "decision_id": decision_id,
            "session_id": session_id,
            "actor": actor,
            # Truncated stringification of what triggered the guardrail.
            "trigger_preview": _safe_repr(context.get("attempt") if context else None),
        }
        if self._tracing_manager:
            try:
                await self._tracing_manager.trace_event(
                    "guardrail.decision",
                    payload,
                    "GuardrailManager",
                    str(uuid.uuid4()),
                )
            except Exception:  # pragma: no cover
                logger.warning("GuardrailManager: failed to emit guardrail.decision", exc_info=True)
        else:
            logger.info(
                "guardrail.decision %s (%s): %s — %s",
                stage, guardrail_id, violation.get("action"), violation.get("reason"),
            )

    async def _initialize_guardrails(self) -> None:
        if self._initialized:
            return

        async def _load_guardrails(ids: List[str], expected_type: Type[GuardrailPlugin]) -> List[GuardrailPlugin]:
            loaded_plugins: List[GuardrailPlugin] = []
            for gid in ids:
                config = self._guardrail_configurations.get(gid, {})
                try:
                    instance_any = await self._plugin_manager.get_plugin_instance(gid, config=config)
                    if instance_any and isinstance(instance_any, expected_type):
                        loaded_plugins.append(cast(GuardrailPlugin, instance_any))
                        logger.info(f"Activated {expected_type.__name__}: {gid}")
                    elif instance_any:
                        logger.warning(f"Plugin '{gid}' loaded but is not a valid {expected_type.__name__}.")
                    else:
                        logger.warning(f"{expected_type.__name__} '{gid}' not found or failed to load.")
                except Exception as e:
                    logger.error(f"Error loading {expected_type.__name__} '{gid}': {e}", exc_info=True)
            return loaded_plugins

        self._active_input_guardrails = await _load_guardrails(self._input_guardrail_ids, InputGuardrailPlugin) # type: ignore
        self._active_output_guardrails = await _load_guardrails(self._output_guardrail_ids, OutputGuardrailPlugin) # type: ignore
        self._active_tool_usage_guardrails = await _load_guardrails(self._tool_usage_guardrail_ids, ToolUsageGuardrailPlugin) # type: ignore
        self._initialized = True

    async def check_input_guardrails(self, data: Any, context: Optional[Dict[str, Any]] = None) -> GuardrailViolation:
        if not self._initialized:
            await self._initialize_guardrails()
        for guardrail in self._active_input_guardrails:
            violation = await guardrail.check_input(data, context)
            if violation.get("action") != "allow":
                await self._emit_decision(
                    "input",
                    getattr(guardrail, "plugin_id", "unknown"),
                    violation,
                    {**(context or {}), "attempt": _safe_repr(data, 256)},
                )
                return violation
        return GuardrailViolation(action="allow", reason="All input guardrails passed.")

    async def check_output_guardrails(self, data: Any, context: Optional[Dict[str, Any]] = None) -> GuardrailViolation:
        if not self._initialized:
            await self._initialize_guardrails()
        for guardrail in self._active_output_guardrails:
            violation = await guardrail.check_output(data, context)
            if violation.get("action") != "allow":
                await self._emit_decision(
                    "output",
                    getattr(guardrail, "plugin_id", "unknown"),
                    violation,
                    {**(context or {}), "attempt": _safe_repr(data, 256)},
                )
                return violation
        return GuardrailViolation(action="allow", reason="All output guardrails passed.")

    async def check_tool_usage_guardrails(self, tool: Tool, params: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> GuardrailViolation:
        if not self._initialized:
            await self._initialize_guardrails()
        for guardrail in self._active_tool_usage_guardrails:
            violation = await guardrail.check_tool_usage(tool, params, context)
            if violation.get("action") != "allow":
                await self._emit_decision(
                    "tool_usage",
                    getattr(guardrail, "plugin_id", "unknown"),
                    violation,
                    {
                        **(context or {}),
                        "attempt": _safe_repr(
                            {"tool_id": getattr(tool, "identifier", "?"), "params": params}, 256
                        ),
                    },
                )
                return violation
        return GuardrailViolation(action="allow", reason="All tool usage guardrails passed.")

    async def teardown(self) -> None:
        logger.info("GuardrailManager tearing down active guardrails...")
        all_active_guardrails = self._active_input_guardrails + self._active_output_guardrails + self._active_tool_usage_guardrails
        unique_guardrail_instances: Dict[str, GuardrailPlugin] = {}
        for g_instance in all_active_guardrails:
            if g_instance.plugin_id not in unique_guardrail_instances:
                 unique_guardrail_instances[g_instance.plugin_id] = g_instance

        for guardrail_instance in unique_guardrail_instances.values():
            try:
                await guardrail_instance.teardown()
            except Exception as e:
                logger.error(f"Error tearing down guardrail '{guardrail_instance.plugin_id}': {e}", exc_info=True)

        self._active_input_guardrails.clear()
        self._active_output_guardrails.clear()
        self._active_tool_usage_guardrails.clear()
        self._initialized = False
        logger.info("GuardrailManager teardown complete.")
