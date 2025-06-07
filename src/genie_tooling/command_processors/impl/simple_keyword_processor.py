import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

from genie_tooling.command_processors.abc import CommandProcessorPlugin
from genie_tooling.command_processors.types import CommandProcessorResponse
from genie_tooling.llm_providers.types import ChatMessage
from genie_tooling.tools.abc import Tool

if TYPE_CHECKING:
    from genie_tooling.genie import Genie

logger = logging.getLogger(__name__)

class SimpleKeywordToolSelectorProcessorPlugin(CommandProcessorPlugin):
    plugin_id: str = "simple_keyword_processor_v1"
    description: str = "Selects tools based on simple keyword matching. Prompts user for parameters."
    _genie: Optional["Genie"] = None; _keyword_tool_map: Dict[str, str] = {}; _keyword_priority: List[str] = []

    async def setup(self, config: Optional[Dict[str, Any]]) -> None:
        await super().setup(config); cfg = config or {}
        self._genie = cfg.get("genie_facade")
        if not self._genie: logger.info(f"{self.plugin_id}: Genie facade not found in config during this setup. This is expected if being discovered by global PluginManager. Operational instance will be configured by CommandProcessorManager."); # Don't return
        self._keyword_tool_map = cfg.get("keyword_map", {}); self._keyword_priority = cfg.get("keyword_priority", list(self._keyword_tool_map.keys()))
        if not self._keyword_tool_map: logger.warning(f"{self.plugin_id}: No keyword map provided in configuration. This processor will not select any tools.")
        else: logger.info(f"{self.plugin_id}: Initialized with {len(self._keyword_tool_map)} keyword mappings. Priority: {self._keyword_priority}")

    async def _prompt_for_param(self, param_name: str, param_schema: Dict[str, Any]) -> Any:
        prompt_message = f"  Enter value for '{param_name}' ({param_schema.get('type', 'any')})"
        if "description" in param_schema: prompt_message += f" - {param_schema['description']}"
        if "enum" in param_schema: prompt_message += f" (choices: {', '.join(map(str, param_schema['enum']))})"
        if "default" in param_schema: prompt_message += f" (default: {param_schema['default']})"
        prompt_message += ": "; user_input_str = await asyncio.to_thread(input, prompt_message)
        if not user_input_str and "default" in param_schema: return param_schema["default"]
        if not user_input_str: raise ValueError(f"Required parameter '{param_name}' was not provided.")
        param_type = param_schema.get("type")
        try:
            if param_type == "integer": return int(user_input_str)
            if param_type == "number": return float(user_input_str)
            if param_type == "boolean": return user_input_str.lower() in ["true", "yes", "1", "y"]
            return user_input_str
        except ValueError: logger.warning(f"Could not coerce input '{user_input_str}' to type '{param_type}' for param '{param_name}'. Returning as string."); return user_input_str

    async def process_command(self, command: str, conversation_history: Optional[List[ChatMessage]] = None) -> CommandProcessorResponse:
        if not self._genie: return {"error": f"{self.plugin_id} not properly set up (Genie facade missing)."}
        if not self._keyword_tool_map: return {"llm_thought_process": "No keywords configured for matching.", "error": "No tools selectable by keyword."}
        command_lower = command.lower(); chosen_tool_id: Optional[str] = None; keywords_to_check = self._keyword_priority or list(self._keyword_tool_map.keys())
        for keyword in keywords_to_check:
            if keyword.lower() in command_lower: chosen_tool_id = self._keyword_tool_map.get(keyword); logger.info(f"{self.plugin_id}: Matched keyword '{keyword}', selected tool '{chosen_tool_id}'."); break
        if not chosen_tool_id: logger.info(f"{self.plugin_id}: No keyword match found in command: '{command}'."); return {"llm_thought_process": "No matching keywords found for any configured tool."}
        if not hasattr(self._genie, "_tool_manager") or not self._genie._tool_manager: logger.error(f"{self.plugin_id}: ToolManager not available via Genie facade."); return {"error": "Internal error: ToolManager not accessible."} # type: ignore
        tool_instance: Optional[Tool] = await self._genie._tool_manager.get_tool(chosen_tool_id) # type: ignore
        if not tool_instance: logger.error(f"{self.plugin_id}: Selected tool '{chosen_tool_id}' not found in ToolManager."); return {"error": f"Internal error: Tool '{chosen_tool_id}' not found after keyword match."}
        try:
            metadata = await tool_instance.get_metadata(); input_schema = metadata.get("input_schema", {}); properties = input_schema.get("properties", {}); required_params = input_schema.get("required", []); extracted_params: Dict[str, Any] = {}
            logger.debug(f"\n[Processor: {self.plugin_id}] Tool '{metadata.get('name', chosen_tool_id)}' selected. Please provide parameters:")
            for param_name, param_schema_val in properties.items():
                param_schema_dict = cast(Dict[str, Any], param_schema_val)
                if param_name in required_params or (await asyncio.to_thread(input, f"  Provide optional parameter '{param_name}' ({param_schema_dict.get('type', 'any')})? (y/N): ")).lower() == "y":
                    try: extracted_params[param_name] = await self._prompt_for_param(param_name, param_schema_dict)
                    except ValueError as e_param: return {"error": str(e_param), "llm_thought_process": f"Failed to get parameter '{param_name}' for tool '{chosen_tool_id}'."}
            return {"chosen_tool_id": chosen_tool_id, "extracted_params": extracted_params, "llm_thought_process": f"Selected tool '{chosen_tool_id}' based on keyword match. Prompted user for parameters."}
        except Exception as e: logger.error(f"{self.plugin_id}: Error while getting schema or prompting for tool '{chosen_tool_id}': {e}", exc_info=True); return {"error": f"Error processing tool '{chosen_tool_id}': {str(e)}"}
