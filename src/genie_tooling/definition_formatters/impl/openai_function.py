# src/genie_tooling/definition_formatters/impl/openai_function.py
import logging
from typing import Any, Dict, Optional

# Updated import path for DefinitionFormatter
from genie_tooling.definition_formatters.abc import DefinitionFormatter

logger = logging.getLogger(__name__)

class OpenAIFunctionFormatter(DefinitionFormatter):
    """
    Formats tool definitions into the JSON structure expected by the
    OpenAI Chat Completions API for function calling.
    """
    plugin_id: str = "openai_function_formatter_plugin_v1"
    formatter_id: str = "llm_openai_functions_v1"
    description: str = "Formats tool definitions into the JSON structure expected by OpenAI's function calling API."

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the OpenAIFunctionFormatter.

        This plugin currently has no configurable options.

        Args:
            config: Configuration dictionary (not currently used).
        """
        pass

    def _clean_openapi_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cleans a JSON schema to be compatible with OpenAI's function parameter schema.
        OpenAI has some restrictions (e.g., no `additionalProperties: true` at the root,
        description fields are crucial).
        """
        if not isinstance(schema, dict):
            logger.warning(f"Input schema is not a dict, returning as is: {type(schema)}")
            return schema if isinstance(schema, dict) else {} # Should not happen if schema is valid JSON Schema

        cleaned_schema = schema.copy()

        # OpenAI expects 'type: object' for parameters root
        if "type" not in cleaned_schema:
            cleaned_schema["type"] = "object"
        elif cleaned_schema["type"] != "object":
            logger.warning(f"OpenAI function parameters schema root type is '{cleaned_schema['type']}', should be 'object'. Adjusting.")
            # This might be too aggressive if the original schema was for a simple type.
            # However, function parameters are typically objects.
            # If it was e.g. a string, it should be wrapped: {"type":"object", "properties": {"param_name": {"type":"string"}}}
            # This formatter assumes the input_schema is already structured as an object with properties.
            cleaned_schema["type"] = "object"


        if "properties" not in cleaned_schema:
            cleaned_schema["properties"] = {}
            # If the original schema was, for example, just {"type": "string"},
            # OpenAI needs it as {"type": "object", "properties": {"value": {"type": "string"}}}
            # This formatter does not automatically create such a wrapper; expects input_schema to be an object.


        # Ensure descriptions for properties if missing (OpenAI relies on them)
        if "properties" in cleaned_schema and isinstance(cleaned_schema["properties"], dict):
            for prop_name, prop_schema in cleaned_schema["properties"].items():
                if isinstance(prop_schema, dict) and "description" not in prop_schema:
                    prop_schema["description"] = f"Parameter '{prop_name}'." # Add a default description

        # OpenAI generally doesn't like `additionalProperties` being explicitly `True` at the root of parameters.
        # If it's not 'false', it's often better to remove it or ensure it's false.
        # However, within nested objects, it might be acceptable.
        # For simplicity, if present at root and True, we'll remove it.
        if cleaned_schema.get("additionalProperties") is True:
            del cleaned_schema["additionalProperties"]
            logger.debug("Removed 'additionalProperties: true' from root of OpenAI parameters schema.")

        return cleaned_schema

    def format(self, tool_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transforms tool metadata into the JSON schema format expected by OpenAI.
        """
        tool_name = tool_metadata.get("identifier", tool_metadata.get("name"))
        if not tool_name:
            logger.error("Tool metadata missing 'identifier' or 'name'. Cannot format for OpenAI.")
            # Return a structure indicating error, or raise an exception
            return {
                "error": "Missing tool name/identifier in metadata",
                "metadata_provided": tool_metadata
            }

        # OpenAI tool names should be alphanumeric with underscores, max 64 chars.
        # Replace spaces and hyphens with underscores, remove other special chars.
        safe_tool_name = "".join(c if c.isalnum() or c == "_" else "_" for c in tool_name.replace("-", "_").replace(" ", "_"))
        if len(safe_tool_name) > 64:
            safe_tool_name = safe_tool_name[:64]
            logger.warning(f"Tool name '{tool_name}' truncated to '{safe_tool_name}' for OpenAI compatibility.")
        if not safe_tool_name: # If all chars were special
            safe_tool_name = "unnamed_tool"


        description = tool_metadata.get("description_llm", tool_metadata.get("description_human"))
        if not description:
            description = f"Executes the '{safe_tool_name}' tool." # Default if no description
            logger.warning(f"Tool '{safe_tool_name}' missing LLM/human description. Using default.")


        input_schema = tool_metadata.get("input_schema", {"type": "object", "properties": {}})
        cleaned_parameters_schema = self._clean_openapi_schema(input_schema)

        openai_function_definition = {
            "type": "function",
            "function": {
                "name": safe_tool_name,
                "description": description,
                "parameters": cleaned_parameters_schema,
            }
        }
        return openai_function_definition

    async def teardown(self) -> None:
        pass