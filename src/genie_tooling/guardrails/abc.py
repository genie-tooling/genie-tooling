"""Abstract Base Class/Protocols for Guardrail Plugins."""
import logging
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from genie_tooling.core.types import Plugin
from genie_tooling.tools.abc import Tool  # For ToolUsageGuardrailPlugin

from .types import GuardrailAction, GuardrailViolation

logger = logging.getLogger(__name__)

@runtime_checkable
class GuardrailPlugin(Plugin, Protocol):
    """Base protocol for all guardrail plugins."""
    plugin_id: str # From Plugin
    description: str # Human-readable description
    default_action: GuardrailAction = "allow" # Default action if check passes or is inconclusive

@runtime_checkable
class InputGuardrailPlugin(GuardrailPlugin, Protocol):
    """Protocol for guardrails that check input data (e.g., prompts, user messages)."""
    async def check_input(self, data: Any, context: Optional[Dict[str, Any]] = None) -> GuardrailViolation:
        """
        Checks input data.
        Returns:
            GuardrailViolation: Contains action (allow, block, warn) and reason.
        """
        logger.warning(f"InputGuardrailPlugin '{self.plugin_id}' check_input method not fully implemented.")
        return GuardrailViolation(action=self.default_action, reason="Not implemented")

@runtime_checkable
class OutputGuardrailPlugin(GuardrailPlugin, Protocol):
    """Protocol for guardrails that check output data (e.g., LLM responses, tool results)."""
    async def check_output(self, data: Any, context: Optional[Dict[str, Any]] = None) -> GuardrailViolation:
        """
        Checks output data.
        Returns:
            GuardrailViolation: Contains action (allow, block, warn) and reason.
        """
        logger.warning(f"OutputGuardrailPlugin '{self.plugin_id}' check_output method not fully implemented.")
        return GuardrailViolation(action=self.default_action, reason="Not implemented")

@runtime_checkable
class ToolUsageGuardrailPlugin(GuardrailPlugin, Protocol):
    """Protocol for guardrails that check tool usage attempts."""
    async def check_tool_usage(self, tool: Tool, params: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> GuardrailViolation:
        """
        Checks if a tool usage attempt is permissible.
        Returns:
            GuardrailViolation: Contains action (allow, block, warn) and reason.
        """
        logger.warning(f"ToolUsageGuardrailPlugin '{self.plugin_id}' check_tool_usage method not fully implemented.")
        return GuardrailViolation(action=self.default_action, reason="Not implemented")
