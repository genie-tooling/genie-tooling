"""Abstract Base Class/Protocol for DefinitionFormatter Plugins."""
from typing import Any, Dict, Protocol, runtime_checkable

from genie_tooling.core.types import Plugin


@runtime_checkable
class DefinitionFormatter(Plugin, Protocol):
    """
    Protocol for a plugin that formats a tool's metadata
    into a specific structure (e.g., for LLMs, for human readability).
    """
    # plugin_id: str (from Plugin protocol)
    formatter_id: str # Specific ID for the format it produces (e.g., "openai_functions_v1", "human_json_v1")
    description: str # Human-readable description of the output format this formatter generates

    def format(self, tool_metadata: Dict[str, Any]) -> Any:
        """
        Takes the comprehensive metadata from Tool.get_metadata()
        and transforms it into the specific output format.

        Args:
            tool_metadata: The raw metadata dictionary from a Tool instance.

        Returns:
            The formatted definition (e.g., a dict for JSON, a string for text).
            The type `Any` allows flexibility for various output formats.
        """
        ...
