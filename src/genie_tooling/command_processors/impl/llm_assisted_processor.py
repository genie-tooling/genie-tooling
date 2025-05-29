import asyncio
import json
import logging
import re  # For more robust JSON extraction
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from genie_tooling.command_processors.abc import CommandProcessorPlugin
from genie_tooling.command_processors.types import CommandProcessorResponse
from genie_tooling.llm_providers.types import ChatMessage

if TYPE_CHECKING:
    from genie_tooling.genie import Genie

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT_TEMPLATE = """
You are an expert AI assistant responsible for selecting the most appropriate tool and extracting its parameters based on a user's command and conversation history.
You will be provided with a list of available tools, their descriptions, and their input schemas.

Your task is to:
1. Analyze the user's command and any relevant conversation history.
2. Choose the single most appropriate tool_id from the provided list. If no tool is suitable, choose null.
3. If a tool is chosen, extract all necessary parameters for that tool from the user's command or conversation. Adhere strictly to the tool's input schema for parameter names and types.
4. Provide a brief "thought" process explaining your choice and parameter extraction. This thought process can be enclosed in <think>...</think> tags BEFORE the JSON block.

Respond with a JSON object matching the following schema. The JSON block MUST start with {{ and end with }}.
{{
  "type": "object",
  "properties": {{
    "thought": {{ "type": "string", "description": "Your reasoning for the tool choice and parameter extraction (can be same as in <think> tags or a summary)." }},
    "tool_id": {{ "type": ["string", "null"], "description": "The ID of the chosen tool, or null if no tool is appropriate." }},
    "params": {{ "type": ["object", "null"], "description": "An object containing extracted parameters for the chosen tool, or null." }}
  }},
  "required": ["thought", "tool_id", "params"]
}}

Available Tools:
---
{tool_definitions_string}
---
"""

class LLMAssistedToolSelectionProcessorPlugin(CommandProcessorPlugin):
    plugin_id: str = "llm_assisted_tool_selection_processor_v1"
    description: str = "Uses an LLM to select a tool and extract parameters."

    _genie: Optional["Genie"] = None
    _llm_provider_id: Optional[str] = None
    _tool_formatter_id: str = "compact_text_formatter_plugin_v1" # Default to the plugin_id
    _tool_lookup_top_k: Optional[int] = None
    _system_prompt_template: str = DEFAULT_SYSTEM_PROMPT_TEMPLATE
    _max_llm_retries: int = 1

    async def setup(self, config: Optional[Dict[str, Any]]) -> None:
        await super().setup(config)
        cfg = config or {}
        self._genie = cfg.get("genie_facade")
        if not self._genie:
             logger.error(f"{self.plugin_id}: Genie facade not found in config or is invalid. This processor cannot function.")
             return

        self._llm_provider_id = cfg.get("llm_provider_id")
        self._tool_formatter_id = cfg.get("tool_formatter_id", self._tool_formatter_id)
        self._tool_lookup_top_k = cfg.get("tool_lookup_top_k")
        if self._tool_lookup_top_k is not None: self._tool_lookup_top_k = int(self._tool_lookup_top_k)
        self._system_prompt_template = cfg.get("system_prompt_template", self._system_prompt_template)
        self._max_llm_retries = int(cfg.get("max_llm_retries", self._max_llm_retries))
        logger.info(f"{self.plugin_id}: Initialized. LLM Provider (if specified): {self._llm_provider_id}, "
                    f"Tool Formatter Plugin ID: {self._tool_formatter_id}, Lookup Top K: {self._tool_lookup_top_k}")

    async def _get_tool_definitions_string(self, command: str) -> Tuple[str, List[str]]:
        if not self._genie: return "Error: Genie facade not available.", []

        tool_ids_to_format: List[str] = []
        all_available_tools = await self._genie._tool_manager.list_tools(enabled_only=True) # type: ignore

        if self._tool_lookup_top_k and self._tool_lookup_top_k > 0 and hasattr(self._genie, "_tool_lookup_service") and self._genie._tool_lookup_service is not None: # type: ignore
            try:
                # ToolLookupService needs the indexing_formatter_id from MiddlewareConfig for its reindex.
                # We pass the tool_formatter_id (plugin_id of formatter) for formatting definitions here.
                indexing_formatter_plugin_id = self._genie._config.default_tool_indexing_formatter_id # type: ignore
                ranked_results = await self._genie._tool_lookup_service.find_tools( # type: ignore
                    command,
                    top_k=self._tool_lookup_top_k,
                    indexing_formatter_id_override=indexing_formatter_plugin_id
                )
                tool_ids_to_format = [r.tool_identifier for r in ranked_results]
                if not tool_ids_to_format:
                    logger.debug(f"{self.plugin_id}: Tool lookup returned no results for command, using all tools.")
                    tool_ids_to_format = [t.identifier for t in all_available_tools]
                else:
                    logger.debug(f"{self.plugin_id}: Using {len(tool_ids_to_format)} tools from lookup service: {tool_ids_to_format}")
            except Exception as e_lookup:
                logger.warning(f"{self.plugin_id}: Error during tool lookup: {e_lookup}. Falling back to all tools.")
                tool_ids_to_format = [t.identifier for t in all_available_tools]
        else:
            tool_ids_to_format = [t.identifier for t in all_available_tools]
            logger.debug(f"{self.plugin_id}: Tool lookup not used or not available. Using all {len(tool_ids_to_format)} tools.")

        if not tool_ids_to_format:
            return "No tools available.", []

        formatted_definitions = []
        for tool_id in tool_ids_to_format:
            # Pass the plugin_id of the formatter
            formatted_def = await self._genie._tool_manager.get_formatted_tool_definition(tool_id, self._tool_formatter_id) # type: ignore
            if formatted_def:
                if isinstance(formatted_def, dict):
                    formatted_definitions.append(json.dumps(formatted_def, indent=2))
                else:
                    formatted_definitions.append(str(formatted_def))
            else:
                logger.warning(f"{self.plugin_id}: Failed to get formatted definition for tool '{tool_id}' using formatter plugin ID '{self._tool_formatter_id}'.")

        return "\n\n".join(formatted_definitions) if formatted_definitions else "No tool definitions could be formatted.", tool_ids_to_format

    def _extract_json_block(self, text: str) -> Optional[str]:
        """Extracts the first valid JSON object string from text that might contain other data."""
        # Regex to find a string that starts with { and ends with }, attempting to match balanced braces.
        # This is a common pattern but might not be perfectly robust for all edge cases of malformed LLM output.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            potential_json = match.group(0)
            try:
                # Validate if it's actually parseable JSON
                json.loads(potential_json)
                return potential_json
            except json.JSONDecodeError:
                logger.debug(f"Found a block between {{ and }}, but it's not valid JSON: {potential_json[:100]}...")

        # Fallback if no curly-brace block is found or if it's invalid
        # Try to find JSON after common LLM preamble like "Here's the JSON:"
        json_keywords = ["json", "```json"]
        for keyword in json_keywords:
            if keyword in text:
                # Attempt to find the start of the JSON after the keyword
                json_start_index = text.find(keyword) + len(keyword)
                # Find the first '{' after the keyword
                first_brace_index = text.find("{", json_start_index)
                if first_brace_index != -1:
                    # Try to find the matching '}'
                    # This is a simplified way, a proper parser would be more robust
                    # For now, we'll just take the substring and try to parse
                    substring_after_brace = text[first_brace_index:]
                    # Try to find the last '}' that forms a valid JSON object
                    # This is tricky. A simpler approach is to hope the LLM terminates JSON correctly.
                    # For now, we'll rely on the regex above or a simple strip if it starts with {
                    if substring_after_brace.strip().startswith("{") and substring_after_brace.strip().endswith("}"):
                         try:
                            json.loads(substring_after_brace.strip())
                            return substring_after_brace.strip()
                         except json.JSONDecodeError:
                            pass # Continue to next keyword or return None
        return None


    async def process_command(
        self,
        command: str,
        conversation_history: Optional[List[ChatMessage]] = None
    ) -> CommandProcessorResponse:
        if not self._genie:
            return {"error": f"{self.plugin_id} not properly set up (Genie facade missing)."}

        tool_definitions_str, candidate_tool_ids = await self._get_tool_definitions_string(command)

        if "Error: Genie facade not available." in tool_definitions_str:
             return {"error": tool_definitions_str }
        if not candidate_tool_ids and "No tools available" not in tool_definitions_str :
             return {"error": "Failed to get any tool definitions for the LLM."}
        if not candidate_tool_ids :
             logger.info(f"{self.plugin_id}: No candidate tools to present to LLM. Definitions string: '{tool_definitions_str[:100]}...'")
             return {"llm_thought_process": "No tools are available or could be formatted for selection.", "error": "No tools processable."}

        system_prompt = self._system_prompt_template.format(tool_definitions_string=tool_definitions_str)
        messages: List[ChatMessage] = [{"role": "system", "content": system_prompt}]
        if conversation_history: messages.extend(conversation_history)
        messages.append({"role": "user", "content": command})

        for attempt in range(self._max_llm_retries + 1):
            try:
                logger.debug(f"{self.plugin_id}: Attempt {attempt+1}: Sending request to LLM for tool selection.")
                llm_response = await self._genie.llm.chat(messages=messages, provider_id=self._llm_provider_id)

                if not isinstance(llm_response, dict) or not isinstance(llm_response.get("message"), dict):
                    logger.error(f"{self.plugin_id}: LLM response or its 'message' field is not a dictionary. Response: {llm_response}")
                    if attempt < self._max_llm_retries: await asyncio.sleep(0.5 * (attempt + 1)); continue
                    return {"error": "Invalid LLM response structure.", "raw_response": llm_response}

                response_content = llm_response["message"].get("content")
                if not response_content or not isinstance(response_content, str):
                    logger.warning(f"{self.plugin_id}: LLM returned empty or non-string content. Content: {response_content}. Raw: {llm_response.get('raw_response')}")
                    if attempt < self._max_llm_retries: await asyncio.sleep(0.5 * (attempt + 1)); continue
                    return {"error": "LLM returned empty or invalid content for tool selection.", "raw_response": llm_response.get("raw_response")}

                json_str_from_llm = self._extract_json_block(response_content)
                if not json_str_from_llm:
                    logger.warning(f"{self.plugin_id}: Could not extract a JSON block from LLM response. Content: '{response_content}'")
                    if attempt < self._max_llm_retries: await asyncio.sleep(0.5 * (attempt + 1)); continue
                    return {"error": "LLM response did not contain a recognizable JSON block.", "raw_response": response_content}

                parsed_llm_output: Dict[str, Any]
                try:
                    parsed_llm_output = json.loads(json_str_from_llm)
                    if not isinstance(parsed_llm_output, dict):
                        raise json.JSONDecodeError("Parsed content is not a dictionary.", json_str_from_llm, 0)
                except json.JSONDecodeError as e_json_dec:
                    logger.warning(f"{self.plugin_id}: Failed to parse extracted JSON from LLM: {e_json_dec}. Extracted JSON: '{json_str_from_llm}'")
                    if attempt < self._max_llm_retries: await asyncio.sleep(0.5 * (attempt + 1)); continue
                    return {"error": f"Extracted JSON from LLM was invalid: {e_json_dec}", "raw_response": response_content}

                thought = parsed_llm_output.get("thought", "No thought process provided by LLM.")
                chosen_tool_id = parsed_llm_output.get("tool_id")
                extracted_params = parsed_llm_output.get("params")

                if chosen_tool_id and chosen_tool_id not in candidate_tool_ids:
                    logger.warning(f"{self.plugin_id}: LLM chose tool '{chosen_tool_id}' which was not in the candidate list ({candidate_tool_ids}). Treating as no tool chosen.")
                    chosen_tool_id = None
                    extracted_params = None
                    thought += " (Note: LLM hallucinated a tool_id not in the provided list. Corrected to no tool.)"

                if chosen_tool_id and not isinstance(extracted_params, (dict, type(None))):
                    logger.warning(f"{self.plugin_id}: LLM returned invalid 'params' type for tool '{chosen_tool_id}'. Expected dict or null, got {type(extracted_params)}.")
                    if attempt < self._max_llm_retries: await asyncio.sleep(0.5 * (attempt + 1)); continue
                    extracted_params = None # Or {} if preferred for chosen_tool_id but bad params
                    thought += " (Note: LLM returned invalid parameter format. Parameters ignored.)"

                return {
                    "chosen_tool_id": chosen_tool_id,
                    "extracted_params": extracted_params if isinstance(extracted_params, dict) else {},
                    "llm_thought_process": thought,
                    "raw_response": llm_response.get("raw_response")
                }
            except Exception as e_llm_call:
                logger.error(f"{self.plugin_id}: Error during LLM call for tool selection (attempt {attempt+1}): {e_llm_call}", exc_info=True)
                if attempt < self._max_llm_retries: await asyncio.sleep(1 * (attempt + 1)); continue
                return {"error": f"Failed to process command with LLM after multiple retries: {str(e_llm_call)}"}

        return {"error": "LLM processing failed after all retries."}
