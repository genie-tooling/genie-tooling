### src/genie_tooling/command_processors/impl/llm_assisted_processor.py

# src/genie_tooling/command_processors/impl/llm_assisted_processor.py
import asyncio
import json
import logging
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
4. Provide a brief "thought" process explaining your choice and parameter extraction.

Respond ONLY with a JSON object matching the following schema:
{{
  "type": "object",
  "properties": {{
    "thought": {{ "type": "string", "description": "Your reasoning for the tool choice and parameter extraction." }},
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
    _tool_formatter_id: str = "llm_compact_text_v1" # Compact format for LLM context
    _tool_lookup_top_k: Optional[int] = None # How many tools to pre-filter using lookup
    _system_prompt_template: str = DEFAULT_SYSTEM_PROMPT_TEMPLATE
    _max_llm_retries: int = 1 # Retries for JSON parsing or LLM errors

    async def setup(self, config: Optional[Dict[str, Any]], genie_facade: "Genie") -> None:
        await super().setup(config, genie_facade)
        self._genie = genie_facade

        cfg = config or {}
        self._llm_provider_id = cfg.get("llm_provider_id") # If None, Genie's default LLM will be used
        self._tool_formatter_id = cfg.get("tool_formatter_id", self._tool_formatter_id)
        self._tool_lookup_top_k = cfg.get("tool_lookup_top_k") # e.g., 5 or 10
        self._system_prompt_template = cfg.get("system_prompt_template", self._system_prompt_template)
        self._max_llm_retries = int(cfg.get("max_llm_retries", self._max_llm_retries))

        logger.info(f"{self.plugin_id}: Initialized. LLM Provider (if specified): {self._llm_provider_id}, "
                    f"Tool Formatter: {self._tool_formatter_id}, Lookup Top K: {self._tool_lookup_top_k}")

    async def _get_tool_definitions_string(self, command: str) -> Tuple[str, List[str]]:
        """Gets formatted tool definitions, potentially filtered by lookup."""
        if not self._genie: return "", []

        tool_ids_to_format: List[str] = []
        # Ensure _tool_manager and _tool_lookup_service are accessed correctly if Genie structure changes
        all_available_tools = await self._genie._tool_manager.list_tools(enabled_only=True) # type: ignore

        if self._tool_lookup_top_k and self._tool_lookup_top_k > 0 and hasattr(self._genie, "_tool_lookup_service"):
            try:
                ranked_results = await self._genie._tool_lookup_service.find_tools(command, top_k=self._tool_lookup_top_k) # type: ignore
                tool_ids_to_format = [r.tool_identifier for r in ranked_results]
                if not tool_ids_to_format: # Lookup returned nothing, fall back to all tools
                    logger.debug(f"{self.plugin_id}: Tool lookup returned no results for command, using all tools.")
                    tool_ids_to_format = [t.identifier for t in all_available_tools]
                else:
                    logger.debug(f"{self.plugin_id}: Using {len(tool_ids_to_format)} tools from lookup service.")
            except Exception as e_lookup:
                logger.warning(f"{self.plugin_id}: Error during tool lookup: {e_lookup}. Falling back to all tools.")
                tool_ids_to_format = [t.identifier for t in all_available_tools]
        else:
            tool_ids_to_format = [t.identifier for t in all_available_tools]

        if not tool_ids_to_format:
            return "No tools available.", []

        formatted_definitions = []
        for tool_id in tool_ids_to_format:
            formatted_def = await self._genie._tool_manager.get_formatted_tool_definition(tool_id, self._tool_formatter_id) # type: ignore
            if formatted_def:
                # Ensure string representation for the prompt
                if isinstance(formatted_def, dict):
                    formatted_definitions.append(json.dumps(formatted_def, indent=2))
                else:
                    formatted_definitions.append(str(formatted_def))

        return "\n\n".join(formatted_definitions) if formatted_definitions else "No tool definitions could be formatted.", tool_ids_to_format


    async def process_command(
        self,
        command: str,
        conversation_history: Optional[List[ChatMessage]] = None
    ) -> CommandProcessorResponse:
        if not self._genie:
            return {"error": f"{self.plugin_id} not properly set up."}

        tool_definitions_str, candidate_tool_ids = await self._get_tool_definitions_string(command)
        if not candidate_tool_ids and "No tools available" not in tool_definitions_str: # Check if formatting failed
            return {"error": "Failed to get any tool definitions for the LLM."}
        if not candidate_tool_ids: # No tools at all
             return {"llm_thought_process": "No tools are available in the system.", "error": "No tools available."}


        system_prompt = self._system_prompt_template.format(tool_definitions_string=tool_definitions_str)

        messages: List[ChatMessage] = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": command})

        for attempt in range(self._max_llm_retries + 1):
            try:
                logger.debug(f"{self.plugin_id}: Attempt {attempt+1}: Sending request to LLM for tool selection.")
                llm_response = await self._genie.llm.chat(messages=messages, provider_id=self._llm_provider_id)

                # Ensure llm_response and llm_response["message"] are dictionaries
                if not isinstance(llm_response, dict) or not isinstance(llm_response.get("message"), dict):
                    logger.error(f"{self.plugin_id}: LLM response or its 'message' field is not a dictionary. Response: {llm_response}")
                    if attempt < self._max_llm_retries: await asyncio.sleep(0.5 * (attempt + 1)); continue
                    return {"error": "Invalid LLM response structure.", "raw_response": llm_response}

                response_content = llm_response["message"].get("content")
                if not response_content or not isinstance(response_content, str):
                    logger.warning(f"{self.plugin_id}: LLM returned empty or non-string content. Content: {response_content}. Raw: {llm_response.get('raw_response')}")
                    if attempt < self._max_llm_retries: await asyncio.sleep(0.5 * (attempt + 1)); continue
                    return {"error": "LLM returned empty or invalid content for tool selection.", "raw_response": llm_response.get("raw_response")}

                parsed_llm_output: Dict[str, Any]
                try:
                    # Basic cleaning for common markdown code block
                    cleaned_content = response_content.strip()
                    if cleaned_content.startswith("```json"):
                        cleaned_content = cleaned_content[7:]
                    elif cleaned_content.startswith("```"): # Handle case where 'json' hint is missing
                        cleaned_content = cleaned_content[3:]

                    if cleaned_content.endswith("```"):
                        cleaned_content = cleaned_content[:-3]

                    parsed_llm_output = json.loads(cleaned_content.strip())
                    if not isinstance(parsed_llm_output, dict):
                        raise json.JSONDecodeError("Parsed content is not a dictionary.", cleaned_content, 0)

                except json.JSONDecodeError as e_json_dec:
                    logger.warning(f"{self.plugin_id}: Failed to parse LLM JSON output: {e_json_dec}. Content: '{response_content}'")
                    if attempt < self._max_llm_retries: await asyncio.sleep(0.5 * (attempt + 1)); continue
                    return {"error": f"LLM output was not valid JSON: {e_json_dec}", "raw_response": response_content}

                thought = parsed_llm_output.get("thought", "No thought process provided.")
                chosen_tool_id = parsed_llm_output.get("tool_id")
                extracted_params = parsed_llm_output.get("params")

                if chosen_tool_id and chosen_tool_id not in candidate_tool_ids:
                    logger.warning(f"{self.plugin_id}: LLM chose tool '{chosen_tool_id}' which was not in the candidate list ({candidate_tool_ids}). Treating as no tool chosen.")
                    chosen_tool_id = None
                    extracted_params = None # Clear params if tool ID is invalid
                    thought += " (Note: LLM hallucinated a tool_id not in the provided list. Corrected to no tool.)"


                # Basic validation of parameters (more robust would use tool's input_schema)
                if chosen_tool_id and not isinstance(extracted_params, (dict, type(None))):
                    logger.warning(f"{self.plugin_id}: LLM returned invalid 'params' type for tool '{chosen_tool_id}'. Expected dict or null, got {type(extracted_params)}.")
                    if attempt < self._max_llm_retries: await asyncio.sleep(0.5 * (attempt + 1)); continue
                    # Fallback: treat as if params were not extracted
                    extracted_params = None # Ensure it's None, not an invalid type
                    thought += " (Note: LLM returned invalid parameter format. Parameters ignored.)"


                return {
                    "chosen_tool_id": chosen_tool_id,
                    "extracted_params": extracted_params if isinstance(extracted_params, dict) else {}, # Ensure dict if tool chosen, else empty dict
                    "llm_thought_process": thought,
                    "raw_response": llm_response.get("raw_response")
                }

            except Exception as e_llm_call:
                logger.error(f"{self.plugin_id}: Error during LLM call for tool selection (attempt {attempt+1}): {e_llm_call}", exc_info=True)
                if attempt < self._max_llm_retries: await asyncio.sleep(1 * (attempt + 1)); continue # Longer backoff
                return {"error": f"Failed to process command with LLM after multiple retries: {str(e_llm_call)}"}

        # This part should ideally not be reached if the loop handles all retries
        return {"error": "LLM processing failed after all retries."}
