# src/genie_tooling/definition_formatters/impl/human_readable_json.py
"""HumanReadableJSONFormatter: Produces verbose JSON for developers."""
from typing import Any, Dict, Optional

# Updated import path for DefinitionFormatter
from genie_tooling.definition_formatters.abc import DefinitionFormatter


class HumanReadableJSONFormatter(DefinitionFormatter):
    """
    Formats tool definitions as a comprehensive, well-structured JSON object.

    This formatter is primarily intended for human consumption, such as in
    developer tools, UIs, or for generating documentation, as it preserves
    the full detail of the tool's metadata.
    """
    plugin_id: str = "human_readable_json_formatter_plugin_v1" # Plugin's own ID
    formatter_id: str = "human_json_v1" # ID of the format it produces
    description: str = "Formats tool definitions as verbose, well-structured JSON, suitable for developers and documentation."

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the HumanReadableJSONFormatter.

        This plugin currently has no configurable options.

        Args:
            config: Configuration dictionary (not currently used).
        """
        pass

    def format(self, tool_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns the metadata largely as-is, with keys ordered for consistency.

        It includes all standard metadata fields, setting defaults for clarity
        (e.g., `cacheable: False`) if they are not present in the input.
        """
        # Example: Ensure a specific order of top-level keys for consistency
        ordered_metadata = {
            "identifier": tool_metadata.get("identifier"),
            "name": tool_metadata.get("name"),
            "version": tool_metadata.get("version"),
            "description_human": tool_metadata.get("description_human"),
            "description_llm": tool_metadata.get("description_llm"),
            "input_schema": tool_metadata.get("input_schema"),
            "output_schema": tool_metadata.get("output_schema"),
            "key_requirements": tool_metadata.get("key_requirements"),
            "tags": tool_metadata.get("tags"),
            "cacheable": tool_metadata.get("cacheable", False), # Include defaults for clarity
            "cache_ttl_seconds": tool_metadata.get("cache_ttl_seconds"),
        }
        # Add any other keys present in tool_metadata that are not explicitly ordered
        for key, value in tool_metadata.items():
            if key not in ordered_metadata:
                ordered_metadata[key] = value

        return ordered_metadata

    async def teardown(self) -> None:
        """No specific teardown needed."""
        pass
