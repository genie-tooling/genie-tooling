# src/genie_tooling/definition_formatters/impl/compact_text.py
import logging
from typing import Any, Dict, List, Optional

# Updated import path for DefinitionFormatter
from genie_tooling.definition_formatters.abc import DefinitionFormatter

logger = logging.getLogger(__name__)

class CompactTextFormatter(DefinitionFormatter):
    """
    Formats tool definitions into a compact, single-line text string.
    This format is highly token-efficient and is well-suited for use in
    LLM prompts where context window size is a concern, or for indexing
    tool definitions for keyword or semantic search.
    """
    plugin_id: str = "compact_text_formatter_plugin_v1"
    formatter_id: str = "llm_compact_text_v1" # Used as default for lookup indexing
    description: str = "Formats tool definitions into a compact, token-efficient text string, suitable for lookup indexing or simple LLM prompts."

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the CompactTextFormatter.

        This plugin currently has no configurable options.

        Args:
            config: Configuration dictionary (not currently used).
        """
        pass

    def _format_params(self, input_schema: Dict[str, Any]) -> str:
        """Formats parameters concisely."""
        if not input_schema or not isinstance(input_schema.get("properties"), dict):
            return "no parameters"

        params_list: List[str] = []
        props = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        for name, prop_schema in props.items():
            if not isinstance(prop_schema, dict):
                continue

            param_type = prop_schema.get("type", "any")
            is_required = "req" if name in required else "opt"
            desc = prop_schema.get("description", "")
            # Shorten description for compactness
            short_desc = f" ({desc[:30]}...)" if desc and len(desc) > 30 else (f" ({desc})" if desc else "")

            enum_values = prop_schema.get("enum")
            enum_str = ""
            if enum_values and isinstance(enum_values, list):
                enum_str = f", enum[{','.join(map(str, enum_values[:3]))}{',...' if len(enum_values) > 3 else ''}]"

            params_list.append(f"{name}({param_type}, {is_required}{enum_str}){short_desc}")

        return "; ".join(params_list) if params_list else "no parameters"


    def format(self, tool_metadata: Dict[str, Any]) -> str:
        """
        Generates a compact string like:
        "ToolName: <name> ; ToolID: <id> ; Purpose: <llm_description> ; Args: <param_summary> ; Tags: <tags>"
        """
        name = tool_metadata.get("name", tool_metadata.get("identifier", "UnknownTool"))
        identifier = tool_metadata.get("identifier", "unknown_id")
        # Prioritize LLM description, then human, then a default.
        description = tool_metadata.get("description_llm", tool_metadata.get("description_human", "No description available."))
        # Truncate long descriptions
        description = description[:200] + "..." if len(description) > 200 else description

        input_schema = tool_metadata.get("input_schema", {})
        params_summary = self._format_params(input_schema)

        tags = tool_metadata.get("tags", [])
        tags_str = f" | Tags: {', '.join(tags)}" if tags else ""

        # Construct the compact string
        # Using a clear, parsable (by LLM or simple regex) format
        compact_repr = (
            f"ToolName: {name} ; "
            f"ToolID: {identifier} ; "
            f"Purpose: {description} ; "
            f"Args: {params_summary}"
            f"{tags_str}"
        )
        logger.debug(f"CompactTextFormatter: Generated for '{identifier}': '{compact_repr[:100]}...'")
        return compact_repr.strip()

    async def teardown(self) -> None:
        pass