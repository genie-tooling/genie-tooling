"""HumanReadableJSONFormatter: Produces verbose JSON for developers."""
from typing import Any, Dict, Optional

# Updated import path for DefinitionFormatter
from genie_tooling.definition_formatters.abc import DefinitionFormatter


class HumanReadableJSONFormatter(DefinitionFormatter):
    plugin_id: str = "human_readable_json_formatter_plugin_v1" # Plugin's own ID
    formatter_id: str = "human_json_v1" # ID of the format it produces
    description: str = "Formats tool definitions as verbose, well-structured JSON, suitable for developers and documentation."

    def format(self, tool_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns the metadata largely as-is, potentially with some ordering or minor cleanup
        to enhance human readability if desired. For JSON, returning the dict itself
        is often sufficient.
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

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """No specific setup needed for this simple formatter."""
        pass

    async def teardown(self) -> None:
        """No specific teardown needed."""
        pass
