
# src/genie_tooling/command_processors/impl/simple_keyword_processor.py
import asyncio # For potential async input if used in an async context
import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from genie_tooling.command_processors.abc import CommandProcessorPlugin
from genie_tooling.command_processors.types import CommandProcessorResponse
from genie_tooling.llm_providers.types import ChatMessage # For type hint consistency
from genie_tooling.tools.abc import Tool

if TYPE_CHECKING:
    from genie_tooling.genie import Genie

logger = logging.getLogger(__name__)

class SimpleKeywordToolSelectorProcessorPlugin(CommandProcessorPlugin):
    plugin_id: str = "simple_keyword_processor_v1"
    description: str = "Selects tools based on simple keyword matching. Prompts user for parameters."

    _genie: Optional['Genie'] = None
    _keyword_tool_map: Dict[str, str] = {} # keyword -> tool_identifier
    _keyword_priority: List[str] = [] # Order in which to check keywords

    async def setup(self, config: Optional[Dict[str, Any]], genie_facade: 'Genie') -> None:
        await super().setup(config, genie_facade)
        self._genie = genie_facade
        
        cfg = config or {}
        # Load keyword mapping from config
        # Expected config structure:
        # {
        #   "keyword_map": { 
        #     "calculate": "calculator_tool", "math": "calculator_tool",
        #     "weather": "open_weather_map_tool"
        #   },
        #   "keyword_priority": ["calculate", "weather", "math"] // Optional priority
        # }
        self._keyword_tool_map = cfg.get("keyword_map", {})
        self._keyword_priority = cfg.get("keyword_priority", list(self._keyword_tool_map.keys()))

        if not self._keyword_tool_map:
            logger.warning(f"{self.plugin_id}: No keyword map provided in configuration. This processor will not select any tools.")
        else:
            logger.info(f"{self.plugin_id}: Initialized with {len(self._keyword_tool_map)} keyword mappings. Priority: {self._keyword_priority}")

    async def _prompt_for_param(self, param_name: str, param_schema: Dict[str, Any]) -> Any:
        """Synchronously prompts user for a parameter value and attempts basic type coercion."""
        prompt_message = f"  Enter value for '{param_name}' ({param_schema.get('type', 'any')})"
        if "description" in param_schema:
            prompt_message += f" - {param_schema['description']}"
        if "enum" in param_schema:
            prompt_message += f" (choices: {', '.join(map(str, param_schema['enum']))})"
        if "default" in param_schema:
            prompt_message += f" (default: {param_schema['default']})"
        
        prompt_message += ": "
        
        user_input_str = await asyncio.to_thread(input, prompt_message) # Run input in thread

        if not user_input_str and "default" in param_schema:
            return param_schema["default"]
        if not user_input_str: # If no default and no input for required
            # This simple prompter doesn't re-prompt; a more robust one would.
            raise ValueError(f"Required parameter '{param_name}' was not provided.")

        param_type = param_schema.get("type")
        try:
            if param_type == "integer":
                return int(user_input_str)
            elif param_type == "number":
                return float(user_input_str)
            elif param_type == "boolean":
                return user_input_str.lower() in ["true", "yes", "1", "y"]
            # For string, array, object, no simple coercion here, return as string
            return user_input_str 
        except ValueError:
            logger.warning(f"Could not coerce input '{user_input_str}' to type '{param_type}' for param '{param_name}'. Returning as string.")
            return user_input_str # Fallback to string


    async def process_command(
        self,
        command: str,
        conversation_history: Optional[List[ChatMessage]] = None
    ) -> CommandProcessorResponse:
        if not self._genie:
            return {"error": f"{self.plugin_id} not properly set up with Genie facade."}
        if not self._keyword_tool_map:
            return {"llm_thought_process": "No keywords configured for matching.", "error": "No tools selectable by keyword."}

        command_lower = command.lower()
        chosen_tool_id: Optional[str] = None
        
        # Check based on priority if available, otherwise iterate map
        keywords_to_check = self._keyword_priority or list(self._keyword_tool_map.keys())

        for keyword in keywords_to_check:
            # Use regex for whole word matching to avoid partial matches (e.g., "cat" in "caterpillar")
            # \b is word boundary
            if re.search(r"\b" + re.escape(keyword.lower()) + r"\b", command_lower):
                chosen_tool_id = self._keyword_tool_map.get(keyword)
                if chosen_tool_id:
                    logger.info(f"{self.plugin_id}: Matched keyword '{keyword}', selected tool '{chosen_tool_id}'.")
                    break 
        
        if not chosen_tool_id:
            logger.info(f"{self.plugin_id}: No keyword match found in command: '{command}'.")
            return {"llm_thought_process": "No matching keywords found for any configured tool."}

        # Get tool schema to prompt for params
        tool_instance: Optional[Tool] = await self._genie._tool_manager.get_tool(chosen_tool_id) # type: ignore
        if not tool_instance:
            logger.error(f"{self.plugin_id}: Selected tool '{chosen_tool_id}' not found in ToolManager.")
            return {"error": f"Internal error: Tool '{chosen_tool_id}' not found after keyword match."}

        try:
            metadata = await tool_instance.get_metadata()
            input_schema = metadata.get("input_schema", {})
            properties = input_schema.get("properties", {})
            required_params = input_schema.get("required", [])
            
            extracted_params: Dict[str, Any] = {}
            print(f"\n[Agent Action] Tool '{metadata.get('name', chosen_tool_id)}' selected. Please provide parameters:")

            for param_name, param_schema in properties.items():
                if param_name in required_params or \
                   input(f"  Provide optional parameter '{param_name}' ({param_schema.get('type', 'any')})? (y/N): ").lower() == 'y':
                    try:
                        extracted_params[param_name] = await self._prompt_for_param(param_name, param_schema)
                    except ValueError as e_param: # Catch error from _prompt_for_param if required value not given
                        return {"error": str(e_param), "llm_thought_process": f"Failed to get parameter '{param_name}' for tool '{chosen_tool_id}'."}

            return {
                "chosen_tool_id": chosen_tool_id,
                "extracted_params": extracted_params,
                "llm_thought_process": f"Selected tool '{chosen_tool_id}' based on keyword match. Prompted user for parameters."
            }

        except Exception as e:
            logger.error(f"{self.plugin_id}: Error while getting schema or prompting for tool '{chosen_tool_id}': {e}", exc_info=True)
            return {"error": f"Error processing tool '{chosen_tool_id}': {str(e)}"}