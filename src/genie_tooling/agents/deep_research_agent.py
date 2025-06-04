import asyncio
import json
import logging
import re
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, TypedDict, Type

from pydantic import BaseModel, Field as PydanticField

from genie_tooling.agents.base_agent import BaseAgent
from genie_tooling.agents.types import AgentOutput
from genie_tooling.llm_providers.types import ChatMessage
from genie_tooling.lookup.types import RankedToolResult

if TYPE_CHECKING:
    from genie_tooling.genie import Genie

logger = logging.getLogger(__name__)

# --- Prompt IDs ---
PLANNER_PROMPT_ID = "deep_research_planner_v1"
TOOL_SELECTOR_PROMPT_ID = "deep_research_tool_selector_v1"
RAG_EXTRACTOR_PROMPT_ID = "deep_research_rag_extractor_v1"
WEB_EXTRACTOR_PROMPT_ID = "deep_research_web_extractor_v1"
ARXIV_EXTRACTOR_PROMPT_ID = "deep_research_arxiv_extractor_v1"
CONTROLLER_PROMPT_ID = "deep_research_controller_v1"
ITERATIVE_SYNTHESIZER_PROMPT_ID = "deep_research_iterative_synthesizer_v1"
FINAL_SYNTHESIZER_PROMPT_ID = "deep_research_final_synthesizer_v1"

# --- Pydantic Models for LLM Structured Outputs ---
class ResearchTaskModel(BaseModel):
    sub_query: str = PydanticField(description="A specific question or topic to research for a part of the main goal.")
    suggested_tool_types: List[Literal["rag_search", "web_search", "arxiv_search"]] = PydanticField(
        description="One or more suggested tool types (rag_search, web_search, arxiv_search) to gather information for this sub-query."
    )
    priority: Optional[int] = PydanticField(default=1, description="Priority of this task (e.g., 1-high, 5-low).")
    status: Literal["pending", "in_progress", "completed", "failed_execution", "failed_no_tool"] = PydanticField(
        default="pending", description="Current status of this research task."
    )

class PlannerOutputModel(BaseModel):
    research_tasks: List[ResearchTaskModel] = PydanticField(
        description="A list of specific sub-queries or research tasks derived from the main goal."
    )
    estimated_depth: Literal["Quick Overview", "Standard Investigation", "In-depth Analysis"] = PydanticField(
        description="The estimated depth of research required for the overall goal."
    )
    overall_reasoning: str = PydanticField(description="Brief reasoning for the plan and depth estimation.")

class ToolSelectionOutputModel(BaseModel):
    selected_tool_id: str = PydanticField(
        description="The identifier of the single best tool to use for the sub-query from the provided candidates."
    )
    reasoning: str = PydanticField(description="Brief reasoning for selecting this specific tool.")

class ExtractionOutputModel(BaseModel):
    relevant_summary: str = PydanticField(
        description="A concise summary of the information directly relevant to the sub-query, extracted from the tool's output."
    )
    key_points: Optional[List[str]] = PydanticField(
        default_factory=list, description="A list of key facts, figures, or arguments found."
    )
    source_identifier: str = PydanticField(
        description="The primary source identifier (e.g., URL, ArXiv ID, document title/ID) from which this information was derived."
    )

class ControllerOutputModel(BaseModel):
    decision: Literal["continue_gathering", "refine_plan", "proceed_to_synthesis"] = PydanticField(
        description="The next action for the agent."
    )
    updated_tasks: Optional[List[ResearchTaskModel]] = PydanticField(
        default=None, description="If decision is 'refine_plan', provide the new or modified list of research tasks. Otherwise, this can be null or omitted."
    )
    reasoning: str = PydanticField(description="Brief reasoning for the decision and any plan updates.")

# --- Internal Data Structures ---
class ProcessedSnippet(TypedDict):
    source_type: str
    source_identifier: str
    sub_query: str
    extracted_info: str
    key_points: Optional[List[str]]

class DeepResearchAgent(BaseAgent):
    def __init__(self, genie: "Genie", agent_config: Optional[Dict[str, Any]] = None):
        super().__init__(genie, agent_config)
        
        self.planner_llm_provider_id: Optional[str] = self.agent_config.get("planner_llm_provider_id")
        self.tool_selector_llm_provider_id: Optional[str] = self.agent_config.get("tool_selector_llm_provider_id")
        self.extractor_llm_provider_id: Optional[str] = self.agent_config.get("extractor_llm_provider_id")
        self.controller_llm_provider_id: Optional[str] = self.agent_config.get("controller_llm_provider_id")
        self.synthesizer_llm_provider_id: Optional[str] = self.agent_config.get("synthesizer_llm_provider_id")

        self.max_total_gathering_cycles: int = int(self.agent_config.get("max_total_gathering_cycles", 15))
        self.max_plan_refinement_cycles: int = int(self.agent_config.get("max_plan_refinement_cycles", 2))
        self.max_sub_questions_per_plan: int = int(self.agent_config.get("max_sub_questions_per_plan", 7))
        self.llm_call_retries: int = int(self.agent_config.get("llm_call_retries", 1))

        self.tool_configs: Dict[str, Dict[str, Any]] = {
            "web_search": self.agent_config.get("web_search_tool_config", {"tool_id": "google_search_tool_v1", "num_results": 3}),
            "arxiv_search": self.agent_config.get("arxiv_search_tool_config", {"tool_id": "arxiv_search_tool", "max_results": 3}),
            "rag_search": self.agent_config.get("rag_search_tool_config", {"collection_name": "main_knowledge_base", "top_k": 3}),
        }
        self.tool_definition_formatter_id: str = self.agent_config.get("tool_definition_formatter_id", "compact_text_formatter_plugin_v1")

        # Store schema dictionaries directly
        self.planner_schema_dict = PlannerOutputModel.model_json_schema()
        self.tool_selector_schema_dict = ToolSelectionOutputModel.model_json_schema()
        self.extractor_schema_dict = ExtractionOutputModel.model_json_schema()
        self.controller_schema_dict = ControllerOutputModel.model_json_schema()

        logger.info(
            f"DeepResearchAgent initialized. Max gathering: {self.max_total_gathering_cycles}, "
            f"Max refinement: {self.max_plan_refinement_cycles}. "
            f"Tool configs: {self.tool_configs}"
        )

    async def _call_llm_with_retry(
        self,
        messages: List[ChatMessage],
        output_schema: Optional[Type[BaseModel]] = None,
        provider_id: Optional[str] = None,
        prompt_id_for_log: str = "LLM Call"
    ) -> Optional[Any]:
        for attempt in range(self.llm_call_retries + 1):
            try:
                llm_response = await self.genie.llm.chat(messages=messages, provider_id=provider_id)
                
                if output_schema:
                    parsed_output = await self.genie.llm.parse_output(llm_response, schema=output_schema)
                    if isinstance(parsed_output, output_schema):
                        return parsed_output
                    else:
                        logger.warning(f"LLM output for {prompt_id_for_log} did not parse into {output_schema.__name__}. Got: {type(parsed_output)}. Raw: {llm_response['message']['content'][:200] if llm_response['message']['content'] else 'None'}. Attempt {attempt+1}.")
                elif llm_response["message"]["content"]:
                    try:
                        content_str = llm_response["message"]["content"]
                        match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', content_str, re.DOTALL)
                        if match:
                            return json.loads(match.group(0))
                        logger.warning(f"No JSON block found in LLM response for {prompt_id_for_log} when no schema provided. Content: {content_str[:200]}")
                    except json.JSONDecodeError as e_json:
                        logger.warning(f"Failed to parse JSON from LLM for {prompt_id_for_log} when no schema provided. Content: {llm_response['message']['content'][:200]}. Error: {e_json}")
                else:
                    logger.warning(f"LLM returned no content for {prompt_id_for_log}.")

                if attempt < self.llm_call_retries:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                return None 

            except Exception as e:
                logger.error(f"Error in LLM call for {prompt_id_for_log} (attempt {attempt+1}): {e}", exc_info=True)
                if attempt < self.llm_call_retries:
                    await asyncio.sleep(1.0 * (attempt + 1))
                else:
                    return None
        return None

    async def _generate_initial_plan(self, goal: str) -> Tuple[Optional[List[ResearchTaskModel]], Optional[str]]:
        logger.info(f"Generating initial research plan for: {goal}")
        await self.genie.observability.trace_event("deep_research_agent.generate_initial_plan.start", {"goal": goal}, "DeepResearchAgent")

        # Pass the schema dictionary to the template data
        prompt_data = {"goal": goal, "planner_schema_dict_for_template": self.planner_schema_dict}
        planner_messages = await self.genie.prompts.render_chat_prompt(name=PLANNER_PROMPT_ID, data=prompt_data)
        if not planner_messages:
            logger.error(f"Could not render planner prompt: {PLANNER_PROMPT_ID}")
            await self.genie.observability.trace_event("deep_research_agent.generate_initial_plan.error", {"error": "PlannerPromptRenderingFailed"}, "DeepResearchAgent")
            return None, None

        planner_output: Optional[PlannerOutputModel] = await self._call_llm_with_retry(
            messages=planner_messages,
            output_schema=PlannerOutputModel,
            provider_id=self.planner_llm_provider_id,
            prompt_id_for_log=PLANNER_PROMPT_ID
        )

        if planner_output:
            logger.info(f"Initial plan generated with {len(planner_output.research_tasks)} tasks. Depth: {planner_output.estimated_depth}.")
            await self.genie.observability.trace_event("deep_research_agent.generate_initial_plan.success", {"plan_length": len(planner_output.research_tasks), "depth": planner_output.estimated_depth}, "DeepResearchAgent")
            return planner_output.research_tasks, planner_output.estimated_depth
        
        logger.error("Failed to generate a valid initial plan from LLM.")
        await self.genie.observability.trace_event("deep_research_agent.generate_initial_plan.error", {"error": "PlannerLLMOutputInvalid"}, "DeepResearchAgent")
        return None, None

    async def _select_tool_for_sub_query(self, sub_query: str, suggested_tool_types: List[str]) -> Optional[Tuple[str, str]]:
        logger.debug(f"Selecting tool for sub_query: '{sub_query}' with suggestions: {suggested_tool_types}")
        await self.genie.observability.trace_event("deep_research_agent.select_tool.start", {"sub_query": sub_query, "suggestions": suggested_tool_types}, "DeepResearchAgent")

        candidate_tools_from_lookup: List[RankedToolResult] = []
        if self.genie._tool_lookup_service:
            try:
                candidate_tools_from_lookup = await self.genie._tool_lookup_service.find_tools(
                    sub_query,
                    top_k=self.agent_config.get("tool_lookup_candidates_k", 5),
                    indexing_formatter_id_override=self.genie._config.default_tool_indexing_formatter_id
                )
            except Exception as e_lookup:
                logger.warning(f"Tool lookup failed for sub_query '{sub_query}': {e_lookup}. Proceeding without lookup.")
        
        candidate_tool_definitions: List[str] = []
        candidate_tool_ids_for_llm: List[str] = []

        if candidate_tools_from_lookup:
            for ranked_tool in candidate_tools_from_lookup:
                tool_id = ranked_tool.tool_identifier
                tool_type_for_this_id: Optional[str] = None
                for type_key, cfg_dict in self.tool_configs.items():
                    if cfg_dict.get("tool_id") == tool_id:
                        tool_type_for_this_id = type_key
                        break
                if tool_type_for_this_id and tool_type_for_this_id in suggested_tool_types:
                    formatted_def = await self.genie._tool_manager.get_formatted_tool_definition(tool_id, self.tool_definition_formatter_id)
                    if formatted_def:
                        candidate_tool_definitions.append(str(formatted_def))
                        candidate_tool_ids_for_llm.append(tool_id)
        
        if not candidate_tool_ids_for_llm:
            logger.debug("Tool lookup yielded no specific candidates matching suggestions, or lookup was skipped. Considering all configured tools matching suggestions.")
            for tool_type_key, cfg_dict in self.tool_configs.items():
                if tool_type_key in suggested_tool_types:
                    tool_id = cfg_dict.get("tool_id")
                    if tool_id:
                        formatted_def = await self.genie._tool_manager.get_formatted_tool_definition(tool_id, self.tool_definition_formatter_id)
                        if formatted_def:
                            candidate_tool_definitions.append(str(formatted_def))
                            candidate_tool_ids_for_llm.append(tool_id)
        
        if not candidate_tool_definitions:
            logger.warning(f"No candidate tools found or could be formatted for sub_query '{sub_query}' matching suggestions {suggested_tool_types}.")
            await self.genie.observability.trace_event("deep_research_agent.select_tool.error", {"error": "NoCandidateToolsFound"}, "DeepResearchAgent")
            return None

        tool_defs_string = "\n\n".join(candidate_tool_definitions)
        # Pass the schema dictionary to the template data
        prompt_data = {"sub_query": sub_query, "candidate_tool_definitions_string": tool_defs_string, "tool_selector_schema_dict_for_template": self.tool_selector_schema_dict}
        selector_messages = await self.genie.prompts.render_chat_prompt(name=TOOL_SELECTOR_PROMPT_ID, data=prompt_data)
        if not selector_messages:
            logger.error(f"Could not render tool selector prompt: {TOOL_SELECTOR_PROMPT_ID}")
            await self.genie.observability.trace_event("deep_research_agent.select_tool.error", {"error": "ToolSelectorPromptRenderingFailed"}, "DeepResearchAgent")
            return None

        selection_output: Optional[ToolSelectionOutputModel] = await self._call_llm_with_retry(
            messages=selector_messages,
            output_schema=ToolSelectionOutputModel,
            provider_id=self.tool_selector_llm_provider_id,
            prompt_id_for_log=TOOL_SELECTOR_PROMPT_ID
        )

        if selection_output and selection_output.selected_tool_id in candidate_tool_ids_for_llm:
            selected_tool_id = selection_output.selected_tool_id
            final_tool_type: Optional[str] = None
            for type_key, cfg_dict in self.tool_configs.items():
                if cfg_dict.get("tool_id") == selected_tool_id:
                    final_tool_type = type_key
                    break
            if final_tool_type:
                logger.info(f"Tool Selector LLM chose '{selected_tool_id}' (type: {final_tool_type}) for sub_query '{sub_query}'. Reasoning: {selection_output.reasoning}")
                await self.genie.observability.trace_event("deep_research_agent.select_tool.success", {"selected_tool_id": selected_tool_id, "tool_type": final_tool_type, "reasoning": selection_output.reasoning}, "DeepResearchAgent")
                return final_tool_type, selected_tool_id
            else:
                 logger.error(f"Selected tool_id '{selected_tool_id}' has no configured tool_type. This is an internal logic error.")
        
        logger.warning(f"Tool Selector LLM failed to choose a valid tool or provide valid output for sub_query '{sub_query}'.")
        await self.genie.observability.trace_event("deep_research_agent.select_tool.error", {"error": "ToolSelectionLLMOutputInvalid"}, "DeepResearchAgent")
        return None

    async def _execute_tool_and_process(self, tool_type: str, tool_id: str, sub_query: str) -> Optional[ProcessedSnippet]:
        logger.info(f"Executing tool '{tool_id}' (type: {tool_type}) for sub_query: {sub_query}")
        await self.genie.observability.trace_event("deep_research_agent.execute_tool_process.start", {"tool_id": tool_id, "tool_type": tool_type, "sub_query": sub_query}, "DeepResearchAgent")
        raw_output: Any = None
        source_identifier = sub_query 

        try:
            tool_params = {"query": sub_query} 
            if tool_type == "rag_search":
                rag_config = self.tool_configs["rag_search"]
                raw_output = await self.genie.rag.search(
                    query=sub_query,
                    collection_name=rag_config.get("collection_name", "default_rag_collection"),
                    top_k=rag_config.get("top_k", 3)
                )
                if isinstance(raw_output, list) and raw_output:
                    source_identifier = ", ".join(sorted(list(set(str(item.metadata.get('source_uri', item.id)) for item in raw_output if hasattr(item, 'metadata') and hasattr(item, 'id')))))
            elif tool_type == "web_search":
                web_config = self.tool_configs["web_search"]
                tool_params["num_results"] = web_config.get("num_results", 3)
                raw_output = await self.genie.execute_tool(tool_id, **tool_params)
                if isinstance(raw_output, dict) and raw_output.get("results") and isinstance(raw_output["results"], list) and raw_output["results"]:
                    source_identifier = raw_output["results"][0].get("link", sub_query)
            elif tool_type == "arxiv_search":
                arxiv_config = self.tool_configs["arxiv_search"]
                tool_params["max_results"] = arxiv_config.get("max_results", 3)
                raw_output = await self.genie.execute_tool(tool_id, **tool_params)
                if isinstance(raw_output, dict) and raw_output.get("results") and isinstance(raw_output["results"], list) and raw_output["results"]:
                    source_identifier = raw_output["results"][0].get("arxiv_id", sub_query)
            else:
                logger.error(f"Unknown tool type '{tool_type}' for execution and processing.")
                await self.genie.observability.trace_event("deep_research_agent.execute_tool_process.error", {"error": "UnknownToolType", "tool_type": tool_type}, "DeepResearchAgent")
                return None

            if not raw_output or (isinstance(raw_output, dict) and raw_output.get("error")):
                error_detail = raw_output.get("error") if isinstance(raw_output, dict) else "No output"
                logger.warning(f"Tool '{tool_id}' returned no output or an error: {error_detail}")
                await self.genie.observability.trace_event("deep_research_agent.execute_tool_process.tool_error", {"tool_id": tool_id, "error": error_detail}, "DeepResearchAgent")
                return ProcessedSnippet(source_type=tool_type, source_identifier=source_identifier, sub_query=sub_query, extracted_info=f"[Tool Error: {error_detail}]", key_points=None)

            extractor_prompt_id_map = {
                "rag_search": RAG_EXTRACTOR_PROMPT_ID,
                "web_search": WEB_EXTRACTOR_PROMPT_ID,
                "arxiv_search": ARXIV_EXTRACTOR_PROMPT_ID,
            }
            extractor_prompt_id = extractor_prompt_id_map.get(tool_type)
            if not extractor_prompt_id:
                logger.error(f"No extractor prompt ID configured for tool type: {tool_type}")
                return ProcessedSnippet(source_type=tool_type, source_identifier=source_identifier, sub_query=sub_query, extracted_info=str(raw_output)[:1000], key_points=None)

            # Pass the schema dictionary to the template data
            extractor_messages = await self.genie.prompts.render_chat_prompt(
                name=extractor_prompt_id,
                data={"sub_query": sub_query, "raw_tool_output_json_string": json.dumps(raw_output, default=str), "extractor_schema_dict_for_template": self.extractor_schema_dict}
            )
            if not extractor_messages:
                 logger.error(f"Could not render extractor prompt: {extractor_prompt_id}")
                 return ProcessedSnippet(source_type=tool_type, source_identifier=source_identifier, sub_query=sub_query, extracted_info=str(raw_output)[:1000], key_points=None)

            extraction_output: Optional[ExtractionOutputModel] = await self._call_llm_with_retry(
                messages=extractor_messages,
                output_schema=ExtractionOutputModel,
                provider_id=self.extractor_llm_provider_id,
                prompt_id_for_log=extractor_prompt_id
            )

            if extraction_output:
                logger.info(f"Information extracted for sub_query '{sub_query}' from source '{extraction_output.source_identifier}'.")
                await self.genie.observability.trace_event("deep_research_agent.execute_tool_process.extraction_success", {"source": extraction_output.source_identifier}, "DeepResearchAgent")
                return ProcessedSnippet(
                    source_type=tool_type,
                    source_identifier=extraction_output.source_identifier,
                    sub_query=sub_query,
                    extracted_info=extraction_output.relevant_summary,
                    key_points=extraction_output.key_points
                )
            else:
                logger.warning(f"Extractor LLM failed to process output from {tool_id} for sub_query: {sub_query}. Using raw output snippet.")
                await self.genie.observability.trace_event("deep_research_agent.execute_tool_process.extraction_failed", {"tool_id": tool_id}, "DeepResearchAgent")
                return ProcessedSnippet(source_type=tool_type, source_identifier=source_identifier, sub_query=sub_query, extracted_info=str(raw_output)[:1000], key_points=None)

        except Exception as e:
            logger.error(f"Error executing or processing tool {tool_id} for sub_query '{sub_query}': {e}", exc_info=True)
            await self.genie.observability.trace_event("deep_research_agent.execute_tool_process.error", {"error": str(e), "tool_id": tool_id}, "DeepResearchAgent")
            return ProcessedSnippet(source_type=tool_type, source_identifier=source_identifier, sub_query=sub_query, extracted_info=f"[Error during tool execution/processing: {str(e)}]", key_points=None)

    async def _decide_next_action(self, goal: str, current_plan: List[ResearchTaskModel], gathered_snippets: List[ProcessedSnippet], estimated_depth: str, cycle_counts: Dict[str, int]) -> Tuple[str, Optional[List[ResearchTaskModel]]]:
        logger.info("Controller LLM deciding next action...")
        await self.genie.observability.trace_event("deep_research_agent.decide_next_action.start", {"plan_size": len(current_plan), "snippets_count": len(gathered_snippets)}, "DeepResearchAgent")

        # Pass the schema dictionary to the template data
        prompt_data = {
            "goal": goal,
            "current_plan_json_string": json.dumps([task.model_dump(exclude_none=True) for task in current_plan]),
            "gathered_snippets_json_string": json.dumps(gathered_snippets),
            "initial_estimated_depth": estimated_depth,
            "current_gathering_cycles": cycle_counts["gathering"],
            "max_total_gathering_cycles": self.max_total_gathering_cycles,
            "current_refinement_cycles": cycle_counts["refinement"],
            "max_plan_refinement_cycles": self.max_plan_refinement_cycles,
            "controller_schema_dict_for_template": self.controller_schema_dict
        }
        controller_messages = await self.genie.prompts.render_chat_prompt(name=CONTROLLER_PROMPT_ID, data=prompt_data)
        if not controller_messages:
            logger.error(f"Could not render controller prompt: {CONTROLLER_PROMPT_ID}")
            return "proceed_to_synthesis", current_plan

        controller_output: Optional[ControllerOutputModel] = await self._call_llm_with_retry(
            messages=controller_messages,
            output_schema=ControllerOutputModel,
            provider_id=self.controller_llm_provider_id,
            prompt_id_for_log=CONTROLLER_PROMPT_ID
        )

        if controller_output:
            logger.info(f"Controller decision: {controller_output.decision}. Reasoning: {controller_output.reasoning}")
            await self.genie.observability.trace_event("deep_research_agent.decide_next_action.success", {"decision": controller_output.decision, "reasoning": controller_output.reasoning}, "DeepResearchAgent")
            return controller_output.decision, controller_output.updated_tasks
        
        logger.error("Controller LLM failed to provide a valid decision. Defaulting to synthesis.")
        await self.genie.observability.trace_event("deep_research_agent.decide_next_action.error", {"error": "ControllerLLMOutputInvalid"}, "DeepResearchAgent")
        return "proceed_to_synthesis", current_plan

    async def _synthesize_report(self, goal: str, all_snippets: List[ProcessedSnippet], is_final: bool, previous_synthesis: Optional[str] = None) -> str:
        prompt_id = FINAL_SYNTHESIZER_PROMPT_ID if is_final else ITERATIVE_SYNTHESIZER_PROMPT_ID
        logger.info(f"Synthesizing report (final: {is_final}) using prompt: {prompt_id}")
        await self.genie.observability.trace_event("deep_research_agent.synthesize_report.start", {"is_final": is_final, "snippet_count": len(all_snippets)}, "DeepResearchAgent")

        prompt_data = {
            "goal": goal,
            "information_snippets_json_string": json.dumps(all_snippets),
            "previous_synthesis_text": previous_synthesis or ""
        }
        synthesizer_messages = await self.genie.prompts.render_chat_prompt(name=prompt_id, data=prompt_data)
        if not synthesizer_messages:
            logger.error(f"Could not render synthesizer prompt: {prompt_id}")
            return "Error: Could not render synthesis prompt."

        llm_response = await self.genie.llm.chat(
            messages=synthesizer_messages,
            provider_id=self.synthesizer_llm_provider_id
        )
        report_content = llm_response["message"]["content"] or "Error: Synthesizer LLM returned empty content."
        logger.info(f"Synthesis complete. Report length: {len(report_content)}")
        await self.genie.observability.trace_event("deep_research_agent.synthesize_report.success", {"report_length": len(report_content)}, "DeepResearchAgent")
        return report_content

    async def run(self, goal: str, **kwargs: Any) -> AgentOutput:
        await self.genie.observability.trace_event("deep_research_agent.run.start", {"goal": goal}, "DeepResearchAgent", correlation_id=str(uuid.uuid4()))
        
        current_plan_models, estimated_depth = await self._generate_initial_plan(goal)
        if not current_plan_models or not estimated_depth:
            logger.error("Failed to generate initial research plan.")
            return AgentOutput(status="error", output="Failed to generate initial research plan.", plan=None, history=[])

        gathered_snippets: List[ProcessedSnippet] = []
        current_synthesized_report: Optional[str] = None
        cycle_counts = {"gathering": 0, "refinement": 0}
        
        history_log: List[Dict[str, Any]] = [{"type": "initial_plan", "plan": [task.model_dump(exclude_none=True) for task in current_plan_models], "estimated_depth": estimated_depth}]

        while True:
            if cycle_counts["gathering"] >= self.max_total_gathering_cycles:
                logger.warning("Max total information gathering cycles reached.")
                history_log.append({"type": "limit_reached", "limit": "max_total_gathering_cycles"})
                break
            if cycle_counts["refinement"] >= self.max_plan_refinement_cycles:
                logger.warning("Max plan refinement cycles reached.")
                history_log.append({"type": "limit_reached", "limit": "max_plan_refinement_cycles"})
                break

            next_task_index = -1
            for i, task_model in enumerate(current_plan_models):
                if task_model.status == "pending":
                    next_task_index = i
                    break
            
            if next_task_index != -1:
                current_task_model = current_plan_models[next_task_index]
                current_task_model.status = "in_progress"
                history_log.append({"type": "task_start", "task": current_task_model.model_dump(exclude_none=True)})
                await self.genie.observability.trace_event("deep_research_agent.task.start", current_task_model.model_dump(exclude_none=True), "DeepResearchAgent")

                tool_selection_result = await self._select_tool_for_sub_query(current_task_model.sub_query, current_task_model.suggested_tool_types)

                if tool_selection_result:
                    tool_type, tool_id = tool_selection_result
                    snippet = await self._execute_tool_and_process(tool_type, tool_id, current_task_model.sub_query)
                    if snippet:
                        gathered_snippets.append(snippet)
                        history_log.append({"type": "snippet_gathered", "snippet_source": snippet["source_identifier"], "sub_query": snippet["sub_query"]})
                    current_task_model.status = "completed" if snippet and "[Error" not in snippet["extracted_info"] else "failed_execution"
                else:
                    logger.warning(f"No tool selected for sub_query: {current_task_model.sub_query}. Marking task as failed.")
                    current_task_model.status = "failed_no_tool"
                
                await self.genie.observability.trace_event("deep_research_agent.task.end", {"task_status": current_task_model.status, "sub_query": current_task_model.sub_query}, "DeepResearchAgent")
                cycle_counts["gathering"] += 1
            else: 
                logger.info("All current tasks addressed or no pending tasks. Consulting Controller LLM.")
                decision, updated_plan_tasks_models = await self._decide_next_action(goal, current_plan_models, gathered_snippets, estimated_depth, cycle_counts)
                history_log.append({"type": "controller_decision", "decision": decision, "updated_plan_count": len(updated_plan_tasks_models) if updated_plan_tasks_models else 0})

                if decision == "proceed_to_synthesis":
                    break 
                elif decision == "refine_plan" and updated_plan_tasks_models:
                    current_plan_models = updated_plan_tasks_models
                    cycle_counts["refinement"] += 1
                    logger.info(f"Plan refined. New plan has {len(current_plan_models)} tasks. Refinement cycle {cycle_counts['refinement']}.")
                    history_log.append({"type": "plan_refined", "new_plan": [task.model_dump(exclude_none=True) for task in current_plan_models]})
                    for task_item_model in current_plan_models:
                        if task_item_model.status != "completed": task_item_model.status = "pending"
                elif decision == "continue_gathering":
                    if updated_plan_tasks_models:
                        current_plan_models = updated_plan_tasks_models
                        history_log.append({"type": "plan_adjusted", "new_plan": [task.model_dump(exclude_none=True) for task in current_plan_models]})
                        for task_item_model in current_plan_models:
                            if task_item_model.status != "completed": task_item_model.status = "pending"
                    if not any(task.status == "pending" for task in current_plan_models):
                        logger.info("Controller decided to continue gathering, but no pending tasks remain. Proceeding to synthesis.")
                        break
                else: 
                    logger.warning(f"Controller returned unexpected decision: {decision} or failed to update plan. Proceeding to synthesis.")
                    break
            
            if cycle_counts["gathering"] > 0 and cycle_counts["gathering"] % self.agent_config.get("iterative_synthesis_interval", 3) == 0:
                logger.info(f"Performing iterative synthesis after {cycle_counts['gathering']} gathering cycles...")
                current_synthesized_report = await self._synthesize_report(goal, gathered_snippets, is_final=False, previous_synthesis=current_synthesized_report)
                history_log.append({"type": "iterative_synthesis", "report_snapshot_length": len(current_synthesized_report or "")})

        logger.info("Proceeding to final synthesis of research report.")
        final_report = await self._synthesize_report(goal, gathered_snippets, is_final=True, previous_synthesis=current_synthesized_report)
        history_log.append({"type": "final_synthesis", "report_length": len(final_report)})

        final_status: Literal["success", "error", "max_iterations_reached"] = "success"
        if cycle_counts["gathering"] >= self.max_total_gathering_cycles or cycle_counts["refinement"] >= self.max_plan_refinement_cycles:
            final_status = "max_iterations_reached"
        
        await self.genie.observability.trace_event("deep_research_agent.run.end", {"status": final_status, "final_report_length": len(final_report)}, "DeepResearchAgent")
        return AgentOutput(status=final_status, output=final_report, history=history_log, plan=[task.model_dump(exclude_none=True) for task in current_plan_models])