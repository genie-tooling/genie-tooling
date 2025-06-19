# src/genie_tooling/command_processors/impl/llm_assisted_processor.py

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from genie_tooling.command_processors.abc import CommandProcessorPlugin
from genie_tooling.command_processors.types import CommandProcessorResponse
from genie_tooling.llm_providers.types import ChatMessage
from genie_tooling.utils.json_parser_utils import extract_json_block

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
    """
    A command processor that uses a Large Language Model (LLM) to intelligently
    select a tool and extract its parameters based on a natural language command.
    """
    plugin_id: str = "llm_assisted_tool_selection_processor_v1"
    description: str = "Uses an LLM to select a tool and extract parameters."
    _genie: Optional["Genie"] = None
    _llm_provider_id: Optional[str] = None
    _tool_formatter_id: str = "compact_text_formatter_plugin_v1"
    _tool_lookup_top_k: Optional[int] = None
    _system_prompt_template: str = DEFAULT_SYSTEM_PROMPT_TEMPLATE
    _max_llm_retries: int = 1

    async def setup(self, config: Optional[Dict[str, Any]]) -> None:
        """
        Initializes the LLM-assisted processor with its configuration.
        """
        await super().setup(config)
        cfg = config or {}
        # Store the facade instance if provided during initial setup
        self._genie = cfg.get("genie_facade")
        if not self._genie:
            logger.info(f"{self.plugin_id}: Genie facade not found in config during setup. This is expected if being discovered by the global PluginManager. It must be provided before process_command is called.")

        self._llm_provider_id = cfg.get("llm_provider_id")
        self._tool_formatter_id = cfg.get("tool_formatter_id", self._tool_formatter_id)
        self._tool_lookup_top_k = cfg.get("tool_lookup_top_k")
        if self._tool_lookup_top_k is not None:
            self._tool_lookup_top_k = int(self._tool_lookup_top_k)
        self._system_prompt_template = cfg.get("system_prompt_template", self._system_prompt_template)
        self._max_llm_retries = int(cfg.get("max_llm_retries", self._max_llm_retries))
        logger.info(f"{self.plugin_id}: Initialized. LLM Provider (if specified): {self._llm_provider_id}, Tool Formatter Plugin ID: {self._tool_formatter_id}, Lookup Top K: {self._tool_lookup_top_k}")

    async def _get_tool_definitions_string(self, genie: "Genie", command: str, correlation_id: Optional[str]) -> Tuple[str, List[str]]:
        if not genie:
            return "Error: Genie facade not available.", []
        tool_ids_to_format: List[str] = []
        all_available_tools = await genie._tool_manager.list_tools(enabled_only=True) # type: ignore
        if self._tool_lookup_top_k and self._tool_lookup_top_k > 0 and hasattr(genie, "_tool_lookup_service") and genie._tool_lookup_service is not None: # type: ignore
            try:
                await genie.observability.trace_event("command_processor.tool_lookup.start", {"query": command, "top_k": self._tool_lookup_top_k}, "LLMAssistedToolSelectionProcessor", correlation_id)
                indexing_formatter_plugin_id = genie._config.default_tool_indexing_formatter_id # type: ignore
                ranked_results = await genie._tool_lookup_service.find_tools(command, top_k=self._tool_lookup_top_k, indexing_formatter_id_override=indexing_formatter_plugin_id) # type: ignore
                tool_ids_to_format = [r.tool_identifier for r in ranked_results]
                await genie.observability.trace_event("command_processor.tool_lookup.end", {"results": [r.to_dict() for r in ranked_results]}, "LLMAssistedToolSelectionProcessor", correlation_id)
                if not tool_ids_to_format:
                    await genie.observability.trace_event("log.debug", {"message": "Tool lookup returned no results for command, using all tools."}, "LLMAssistedToolSelectionProcessor", correlation_id)
                    tool_ids_to_format = [t.identifier for t in all_available_tools]
                else:
                    await genie.observability.trace_event("log.debug", {"message": f"Using {len(tool_ids_to_format)} tools from lookup service: {tool_ids_to_format}"}, "LLMAssistedToolSelectionProcessor", correlation_id)
            except Exception as e_lookup:
                await genie.observability.trace_event("log.warning", {"message": f"Error during tool lookup: {e_lookup}. Falling back to all tools."}, "LLMAssistedToolSelectionProcessor", correlation_id)
                await genie.observability.trace_event("command_processor.tool_lookup.error", {"error": str(e_lookup)}, "LLMAssistedToolSelectionProcessor", correlation_id)
                tool_ids_to_format = [t.identifier for t in all_available_tools]
        else:
            tool_ids_to_format = [t.identifier for t in all_available_tools]
            await genie.observability.trace_event("log.debug", {"message": f"Tool lookup not used or not available. Using all {len(tool_ids_to_format)} tools."}, "LLMAssistedToolSelectionProcessor", correlation_id)
        if not tool_ids_to_format:
            return "No tools available.", []
        formatted_definitions = []
        for tool_id in tool_ids_to_format:
            formatted_def = await genie._tool_manager.get_formatted_tool_definition(tool_id, self._tool_formatter_id) # type: ignore
            if formatted_def:
                if isinstance(formatted_def, dict):
                    formatted_definitions.append(json.dumps(formatted_def, indent=2))
                else:
                    formatted_definitions.append(str(formatted_def))
            else:
                await genie.observability.trace_event("log.warning", {"message": f"Failed to get formatted definition for tool '{tool_id}' using formatter plugin ID '{self._tool_formatter_id}'."}, "LLMAssistedToolSelectionProcessor", correlation_id)
        return "\n\n".join(formatted_definitions) if formatted_definitions else "No tool definitions could be formatted.", tool_ids_to_format

    # --- REFACTOR: Internal method removed, will call utility directly ---
    # async def _extract_json_block(...)

    async def process_command(
        self,
        command: str,
        conversation_history: Optional[List[ChatMessage]] = None,
        correlation_id: Optional[str] = None,
        genie_instance: Optional["Genie"] = None,
    ) -> CommandProcessorResponse:
        genie_to_use = genie_instance or self._genie
        if not genie_to_use:
            return {"error": f"{self.plugin_id} not properly set up (Genie facade missing)."}

        tool_definitions_str, candidate_tool_ids = await self._get_tool_definitions_string(genie_to_use, command, correlation_id)

        if "Error: Genie facade not available." in tool_definitions_str:
             return {"error": tool_definitions_str}
        if not candidate_tool_ids and "No tools available" not in tool_definitions_str :
             return {"error": "Failed to get any tool definitions for the LLM."}
        if not candidate_tool_ids :
             await genie_to_use.observability.trace_event("log.info", {"message": f"No candidate tools to present to LLM. Definitions string: '{tool_definitions_str[:100]}...'"}, "LLMAssistedToolSelectionProcessor", correlation_id)
             return {"llm_thought_process": "No tools are available or could be formatted for selection.", "error": "No tools processable."}

        system_prompt = self._system_prompt_template.format(tool_definitions_string=tool_definitions_str)
        await genie_to_use.observability.trace_event("command_processor.llm_assisted.prompt_context_ready", {"tool_definitions": tool_definitions_str, "system_prompt_template_used": self._system_prompt_template}, "LLMAssistedToolSelectionProcessor", correlation_id)
        messages: List[ChatMessage] = [{"role": "system", "content": system_prompt}]
        if conversation_history:
             messages.extend(conversation_history)
        messages.append({"role": "user", "content": command})

        for attempt in range(self._max_llm_retries + 1):
            try:
                await genie_to_use.observability.trace_event("log.debug", {"message": f"Attempt {attempt+1}: Sending request to LLM for tool selection."}, "LLMAssistedToolSelectionProcessor", correlation_id)
                llm_response = await genie_to_use.llm.chat(messages=messages, provider_id=self._llm_provider_id)

                if not isinstance(llm_response, dict) or not isinstance(llm_response.get("message"), dict):
                    await genie_to_use.observability.trace_event("log.error", {"message": f"LLM response or its 'message' field is not a dictionary. Response: {llm_response}"}, "LLMAssistedToolSelectionProcessor", correlation_id)
                    if attempt < self._max_llm_retries:
                        await genie_to_use.observability.trace_event("command_processor.llm_assisted.retry", {"attempt": attempt + 1, "reason": "InvalidLLMResponseStructure"}, "LLMAssistedToolSelectionProcessor", correlation_id)
                        await asyncio.sleep(0.5 * (attempt + 1))
                        continue
                    else:
                        return {"error": "Invalid LLM response structure.", "raw_response": llm_response}

                response_content = llm_response["message"].get("content")
                if not response_content or not isinstance(response_content, str):
                    await genie_to_use.observability.trace_event("log.warning", {"message": f"LLM returned empty or non-string content. Content: {response_content}. Raw: {llm_response.get('raw_response')}"}, "LLMAssistedToolSelectionProcessor", correlation_id)
                    if attempt < self._max_llm_retries:
                        await genie_to_use.observability.trace_event("command_processor.llm_assisted.retry", {"attempt": attempt + 1, "reason": "EmptyLLMContent"}, "LLMAssistedToolSelectionProcessor", correlation_id)
                        await asyncio.sleep(0.5 * (attempt + 1))
                        continue
                    else:
                        return {"error": "LLM returned empty or invalid content for tool selection.", "raw_response": llm_response.get("raw_response")}

                # --- REFACTOR: Call the new utility function ---
                json_str_from_llm = extract_json_block(response_content)
                # --- END REFACTOR ---

                if not json_str_from_llm:
                    await genie_to_use.observability.trace_event("log.warning", {"message": f"Could not extract a JSON block from LLM response. Content: '{response_content}'"}, "LLMAssistedToolSelectionProcessor", correlation_id)
                    if attempt < self._max_llm_retries:
                        await genie_to_use.observability.trace_event("command_processor.llm_assisted.retry", {"attempt": attempt + 1, "reason": "NoJSONBlockFound"}, "LLMAssistedToolSelectionProcessor", correlation_id)
                        await asyncio.sleep(0.5 * (attempt + 1))
                        continue
                    else:
                        return {"error": "LLM response did not contain a recognizable JSON block.", "raw_response": response_content}

                parsed_llm_output: Dict[str, Any]
                try:
                    parsed_llm_output = json.loads(json_str_from_llm)
                    if not isinstance(parsed_llm_output, dict):
                        await genie_to_use.observability.trace_event("log.warning", {"message": f"Parsed JSON from LLM is not a dictionary. Type: {type(parsed_llm_output)}. Extracted JSON: '{json_str_from_llm}'"}, "LLMAssistedToolSelectionProcessor", correlation_id)
                        if attempt < self._max_llm_retries:
                            await genie_to_use.observability.trace_event("command_processor.llm_assisted.retry", {"attempt": attempt + 1, "reason": "ParsedJSONNotDict"}, "LLMAssistedToolSelectionProcessor", correlation_id)
                            await asyncio.sleep(0.5 * (attempt + 1))
                            continue
                        else:
                            return {"error": "Parsed JSON from LLM was not a dictionary.", "raw_response": response_content}
                except json.JSONDecodeError as e_json_dec:
                    await genie_to_use.observability.trace_event("log.warning", {"message": f"Failed to parse extracted JSON from LLM: {e_json_dec}. Extracted JSON: '{json_str_from_llm}'"}, "LLMAssistedToolSelectionProcessor", correlation_id)
                    if attempt < self._max_llm_retries:
                        await genie_to_use.observability.trace_event("command_processor.llm_assisted.retry", {"attempt": attempt + 1, "reason": "InvalidJSON"}, "LLMAssistedToolSelectionProcessor", correlation_id)
                        await asyncio.sleep(0.5 * (attempt + 1))
                        continue
                    else:
                        return {"error": f"Extracted JSON from LLM was invalid: {e_json_dec}", "raw_response": response_content}

                await genie_to_use.observability.trace_event("command_processor.llm_assisted.result", {"parsed_output": parsed_llm_output, "raw_content": response_content}, "LLMAssistedToolSelectionProcessor", correlation_id)
                thought = parsed_llm_output.get("thought", "No thought process provided by LLM.")
                chosen_tool_id = parsed_llm_output.get("tool_id")
                extracted_params_raw = parsed_llm_output.get("params")
                extracted_params: Dict[str, Any] = {}

                if chosen_tool_id:
                    if isinstance(extracted_params_raw, dict):
                        extracted_params = extracted_params_raw
                    elif extracted_params_raw is not None:
                        await genie_to_use.observability.trace_event("log.warning", {"message": f"LLM returned invalid 'params' type for tool '{chosen_tool_id}'. Expected dict or null, got {type(extracted_params_raw)}. Params will be empty."}, "LLMAssistedToolSelectionProcessor", correlation_id)
                        thought += " (Note: LLM returned invalid parameter format. Parameters ignored.)"

                    if chosen_tool_id not in candidate_tool_ids:
                        await genie_to_use.observability.trace_event("log.warning", {"message": f"LLM chose tool '{chosen_tool_id}' which was not in the candidate list ({candidate_tool_ids}). Treating as no tool chosen."}, "LLMAssistedToolSelectionProcessor", correlation_id)
                        chosen_tool_id = None
                        extracted_params = {}
                        thought += " (Note: LLM hallucinated a tool_id not in the provided list. Corrected to no tool.)"

                if not chosen_tool_id:
                    extracted_params = {}

                return {
                    "chosen_tool_id": chosen_tool_id,
                    "extracted_params": extracted_params,
                    "llm_thought_process": thought,
                    "raw_response": llm_response.get("raw_response")
                }
            except Exception as e_llm_call:
                await genie_to_use.observability.trace_event("log.error", {"message": f"Error during LLM call for tool selection (attempt {attempt+1}): {e_llm_call}", "exc_info": True}, "LLMAssistedToolSelectionProcessor", correlation_id)
                await genie_to_use.observability.trace_event("command_processor.llm_assisted.error", {"attempt": attempt + 1, "error": str(e_llm_call)}, "LLMAssistedToolSelectionProcessor", correlation_id)
                if attempt < self._max_llm_retries:
                    await asyncio.sleep(1 * (attempt + 1))
                    continue
                else:
                    return {"error": f"Failed to process command with LLM after multiple retries: {e_llm_call!s}"}

        return {"error": "LLM processing failed after all retries."}
