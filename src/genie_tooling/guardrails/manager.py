### src/genie_tooling/guardrails/manager.py
"""GuardrailManager: Orchestrates GuardrailPlugins."""
import logging
from typing import Any, Dict, List, Optional, Type, Union, cast

from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.tools.abc import Tool

from .abc import (
    GuardrailPlugin,
    InputGuardrailPlugin,
    OutputGuardrailPlugin,
    ToolUsageGuardrailPlugin,
)
from .types import GuardrailAction, GuardrailViolation

logger = logging.getLogger(__name__)

class GuardrailManager:
    def __init__(
        self,
        plugin_manager: PluginManager,
        default_input_guardrail_ids: Optional[List[str]] = None,
        default_output_guardrail_ids: Optional[List[str]] = None,
        default_tool_usage_guardrail_ids: Optional[List[str]] = None,
        guardrail_configurations: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        self._plugin_manager = plugin_manager
        self._input_guardrail_ids = default_input_guardrail_ids or []
        self._output_guardrail_ids = default_output_guardrail_ids or []
        self._tool_usage_guardrail_ids = default_tool_usage_guardrail_ids or []
        self._guardrail_configurations = guardrail_configurations or {}
        
        self._active_input_guardrails: List[InputGuardrailPlugin] = []
        self._active_output_guardrails: List[OutputGuardrailPlugin] = []
        self._active_tool_usage_guardrails: List[ToolUsageGuardrailPlugin] = []
        self._initialized = False
        logger.info("GuardrailManager initialized.")

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
        if not self._initialized: await self._initialize_guardrails()
        for guardrail in self._active_input_guardrails:
            violation = await guardrail.check_input(data, context)
            # Use dictionary key access with .get() for safety
            if violation.get("action") != "allow": 
                return violation
        return GuardrailViolation(action="allow", reason="All input guardrails passed.")

    async def check_output_guardrails(self, data: Any, context: Optional[Dict[str, Any]] = None) -> GuardrailViolation:
        if not self._initialized: await self._initialize_guardrails()
        for guardrail in self._active_output_guardrails:
            violation = await guardrail.check_output(data, context)
            # Use dictionary key access with .get() for safety
            if violation.get("action") != "allow": 
                return violation
        return GuardrailViolation(action="allow", reason="All output guardrails passed.")

    async def check_tool_usage_guardrails(self, tool: Tool, params: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> GuardrailViolation:
        if not self._initialized: await self._initialize_guardrails()
        for guardrail in self._active_tool_usage_guardrails:
            violation = await guardrail.check_tool_usage(tool, params, context)
            # Use dictionary key access with .get() for safety
            if violation.get("action") != "allow": 
                return violation
        return GuardrailViolation(action="allow", reason="All tool usage guardrails passed.")

    async def teardown(self) -> None:
        logger.info("GuardrailManager tearing down active guardrails...")
        all_active_guardrails = self._active_input_guardrails + self._active_output_guardrails + self._active_tool_usage_guardrails
        # Ensure unique_guardrails contains actual plugin instances, not just IDs, for teardown
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