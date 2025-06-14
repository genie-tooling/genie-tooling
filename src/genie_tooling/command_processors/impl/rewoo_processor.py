# src/genie_tooling/command_processors/impl/rewoo_processor.py
import asyncio
import json
import logging
import re
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypedDict,
    Union,
)

# Conditional Pydantic import
try:
    from pydantic import BaseModel as PydanticBaseModelImport
    from pydantic import Field as PydanticFieldImport
    from pydantic import ValidationError as PydanticValidationErrorImport
    from pydantic import create_model as create_model_import
    PYDANTIC_INSTALLED_FOR_REWOO_PROCESSOR = True
except ImportError:
    PydanticBaseModelImport = object # type: ignore
    PydanticFieldImport = lambda **kwargs: {} # type: ignore
    create_model_import = lambda *args, **kwargs: type("FallbackModel", (dict,), {}) # type: ignore
    PydanticValidationErrorImport = ValueError # type: ignore
    PYDANTIC_INSTALLED_FOR_REWOO_PROCESSOR = False


from genie_tooling.command_processors.abc import CommandProcessorPlugin
from genie_tooling.command_processors.types import CommandProcessorResponse
from genie_tooling.llm_providers.types import ChatMessage
from genie_tooling.utils.placeholder_resolution import resolve_placeholders

if TYPE_CHECKING:
    from genie_tooling.genie import Genie
    from genie_tooling.tools.abc import Tool  # For type hinting

logger = logging.getLogger(__name__)

# --- Updated Internal Data Structures ---
class SourceDetails(TypedDict, total=False):
    type: str
    identifier: str
    title: Optional[str]

class ExecutionEvidence(TypedDict):
    step: Dict[str, Any]
    result: Any
    error: Optional[str]
    detailed_summary_or_extraction: Optional[Union[str, Dict[str, Any]]]
    source_details: Optional[SourceDetails]

# --- End of Updated Data Structures ---


class ReWOOCommandProcessorPlugin(CommandProcessorPlugin):
    plugin_id: str = "rewoo_command_processor_v1"
    description: str = "Implements the ReWOO (Reason-Act) pattern: Plan -> Execute -> Solve."
    # --- MODIFICATION START: Use a raw string (r"") to prevent backslash escaping ---
    _DEFAULT_PLANNER_PROMPT = r"""
You are an expert AI assistant that plans a sequence of tool calls to solve a user's goal.
Do NOT execute the tools. Your only job is to create a plan.
The `tool_id` in each step of your plan MUST exactly match one of the `ToolID` values provided in the "Available Tools" section below.

---
**CRITICAL INSTRUCTION FOR `params` FIELD**:
The `params` field for each step MUST be a JSON-encoded STRING. The string itself must contain a valid JSON object.
For example: `"params": "{\"query\": \"Llama 3 scores\"}"`
Another example: `"params": "{\"num1\": 5, \"num2\": 10, \"operation\": \"add\"}"`
If a tool takes no parameters, use an empty JSON object string: `"params": "{}"`
---

If a step's output is needed by a later step, define an `output_variable_name` for that step (e.g., "search_results", "user_details", "page_content").
The `output_variable_name` must be a valid Python identifier (letters, numbers, underscores, not starting with a number).
Subsequent steps can then reference this output in their `params` using the syntax `{{outputs.variable_name.path.to.value}}` or `{{outputs.variable_name.path.to.value}}`.
When constructing the `path.to.value`, you must infer the structure of the tool's output.
- For most tools, the output is a direct dictionary. Example: `{{outputs.extracted_scores.llama3_8b_instruct_mmlu_score}}`.
- For search tools like `intelligent_search_aggregator_v1` or `community_google_search`, the results are usually in `{{outputs.search_output_variable.results}}` which is a list. To access the URL of the first result: `{{outputs.search_output_variable.results.0.url}}`. The snippet is often at `{{outputs.search_output_variable.results.0.snippet_or_summary}}`.
- For `web_page_content_extractor`, the main text is usually in `{{outputs.page_content_variable.content}}`.

Be mindful of potential `null` or missing values from `output_variable_name`s. If a critical piece of data might be missing, the plan should ideally acknowledge that subsequent steps depending on it might not receive valid input, or if possible, include a fallback or check. If a placeholder resolves to `null` and a tool requires a non-null value for that parameter, the step will fail.

---
IMPORTANT RULES FOR TOOL USAGE:
1.  **Web Research Workflow**: If a search tool (like `community_google_search` or `intelligent_search_aggregator_v1`) returns a list of URLs and snippets, and you need the *full content* of a specific URL, you MUST add a subsequent step using the `web_page_content_extractor` tool with the relevant URL (e.g., from `{{outputs.search_results.results.0.url}}`). Store its output (e.g., as `page_text`).
2.  **Data Extraction Workflow**: If you have a block of text (e.g., from `{{outputs.page_text.content}}`) and need to extract specific data points (like numbers, scores, names, dates), you MUST add a subsequent step using the `custom_text_parameter_extractor` tool. Its `text_content` parameter should be the placeholder for the text, and `parameters_to_extract` MUST be provided with the regex patterns (using the key `regex_pattern`). If a parameter is not found, its value will be null/None. Example for `parameters_to_extract`: `[{{\"name\": \"score_value\", \"regex_pattern\": "Score:\\s*(\\d+)"}}]`.
3.  **Calculation Workflow**: For mathematical calculations based on extracted data (e.g., from `custom_text_parameter_extractor`), use the `calculator_tool`. Its `num1` and `num2` parameters should use placeholders referencing the extracted numeric values. Ensure the referenced values are indeed numbers and not null.
4.  **Sentiment Analysis Workflow**: For analyzing sentiment from multiple text sources (e.g., multiple search snippets or extracted web page sections), use the `discussion_sentiment_summarizer` tool. Its `text_snippets` parameter MUST be a list of strings. If using output from a search tool, you need to extract the relevant text field (e.g., `snippet_or_summary`) from each search result item to form this list. For example: `{{\"text_snippets\": ["{{outputs.search_output.results.0.snippet_or_summary}}", "{{outputs.search_output.results.1.snippet_or_summary}}"]}}`.
5.  **RAG Workflow**: To query the internal knowledge base for historical data or specific facts, use the `oss_model_release_impact_analyzer` tool. You MUST provide the `model_family_keywords` and `release_type_tags` parameters.
6.  **Search Preference**: If an `intelligent_search_aggregator_v1` tool is available, prefer it for initial broad searches across multiple sources (web, ArXiv). For highly specific follow-up on a single URL, use `web_page_content_extractor`.
---

{{ schema_instruction }}

Available Tools:
---
{{ tool_definitions }}
---

User's Goal: "{{ goal }}"
{{ previous_attempt_feedback }}
Now, generate the plan. Your response MUST be a single, valid JSON object that conforms to the schema and nothing else. Do not include the schema definition itself.
"""
    # --- MODIFICATION END ---

    _DEFAULT_SOLVER_PROMPT_V2 = """
You are an expert AI assistant that synthesizes a final answer for a user based on their original goal and the evidence gathered from a series of tool calls.
Your answer should comprehensively address all aspects of the original goal. If the goal has multiple parts or implies multiple questions, address each one clearly.
When using information from the evidence, you MUST cite the source using its step number identifier (e.g., "[Evidence from Step 1]").
If some evidence steps report errors (e.g., "Execution Error: ..." or "Step failed with error: ..."), acknowledge these limitations in your answer. Focus on the successfully gathered information to address the parts of the goal you can. If critical information is missing due to errors, clearly state that in your final answer.
After your main answer, list all cited sources under a "### Sources Cited:" heading. For each source, include its step number, title, and identifier (URL or query).

Original Goal: "{goal}"

The following plan was executed:
---
{% for step_item in plan.plan %}
Step {{ loop.index }}:
  Thought: {{ step_item.thought }}
  Action: {{ step_item.tool_id }}[{{ step_item.params | tojson }}]
  {% if step_item.output_variable_name %}Output stored as: {{ step_item.output_variable_name }}{% endif %}
{% endfor %}
---

The following evidence was gathered from executing the plan:
---
{% for item in evidence %}
[Evidence from Step {{ loop.index }} for Action: {{ item.step.tool_id }}]
Source Type: {{ item.source_details.type if item.source_details else 'N/A' }}
Source Title: {{ item.source_details.title if item.source_details else 'N/A' }}
Source Identifier: {{ item.source_details.identifier if item.source_details else 'N/A' }}
{% if item.error %}
Execution Error: {{ item.error }}
{% else %}
Extracted Information/Summary:
{{ item.detailed_summary_or_extraction if item.detailed_summary_or_extraction is not none else "No specific summary/extraction available for this step." }}
{% endif %}
---
{% endfor %}
---

Based on the original goal and the gathered evidence, provide your comprehensive answer.
Remember to cite sources like [Evidence from Step X] and list them at the end.
Your final response should begin directly with the answer, without any preliminary thoughts or XML tags like <think>.

Answer:
"""
    _genie: "Genie"
    _planner_llm_id: Optional[str]
    _solver_llm_id: Optional[str]
    _tool_formatter_id: str
    _max_plan_retries: int
    _planner_prompt_template_engine_id: Optional[str] # Allow None
    _solver_prompt_template_engine_id: Optional[str] # Allow None

    async def setup(self, config: Optional[Dict[str, Any]]) -> None:
        await super().setup(config)
        cfg = config or {}
        self._genie = cfg.get("genie_facade")
        if not self._genie:
            raise ValueError(f"{self.plugin_id} requires a 'genie_facade' instance in its config.")
        self._planner_llm_id = cfg.get("planner_llm_provider_id")
        self._solver_llm_id = cfg.get("solver_llm_provider_id")
        self._tool_formatter_id = cfg.get("tool_formatter_id", "compact_text_formatter_plugin_v1")
        self._max_plan_retries = int(cfg.get("max_plan_retries", 1))
        self._planner_prompt_template_engine_id = cfg.get("planner_prompt_template_engine_id")
        self._solver_prompt_template_engine_id = cfg.get("solver_prompt_template_engine_id")


    def _create_dynamic_plan_model(self, tool_ids: List[str]) -> Type[PydanticBaseModelImport]:
        if not PYDANTIC_INSTALLED_FOR_REWOO_PROCESSOR:
            return dict # Fallback if pydantic is not installed

        if not tool_ids:
            tool_ids = ["placeholder_tool_id_if_none_available"] # Ensure Literal is not empty

        ToolIDEnumForDynamicModel = Literal[tuple(tool_ids)] # type: ignore

        DynamicReWOOStepModel = create_model_import( # type: ignore
            "DynamicReWOOStep",
            thought=(str, PydanticFieldImport(description="The reasoning for why this specific tool call is necessary.")),
            tool_id=(ToolIDEnumForDynamicModel, PydanticFieldImport(description="The identifier of the tool to execute.")), # type: ignore
            params=(str, PydanticFieldImport(default="{}", description="A JSON-encoded STRING containing parameters for the tool.")),
            output_variable_name=(Optional[str], PydanticFieldImport(None, description="If this step's output should be stored for later use, provide a variable name here (e.g., 'search_results'). Use only letters, numbers, and underscores.")),
            __base__=PydanticBaseModelImport
        )
        DynamicReWOOPlanModel = create_model_import( # type: ignore
            "DynamicReWOOPlan",
            plan=(List[DynamicReWOOStepModel], PydanticFieldImport(description="The sequence of tool calls to execute.")), # type: ignore
            overall_reasoning=(Optional[str], PydanticFieldImport(None, description="Overall reasoning for the plan.")),
            __base__=PydanticBaseModelImport
        )
        return DynamicReWOOPlanModel # type: ignore

    async def _generate_plan(self, goal: str, tool_definitions: str, candidate_tool_ids: List[str], correlation_id: Optional[str], previous_attempt_feedback: Optional[str] = None) -> Tuple[Optional[PydanticBaseModelImport], Optional[str]]:
        if not PYDANTIC_INSTALLED_FOR_REWOO_PROCESSOR:
            await self._genie.observability.trace_event("rewoo.plan.error", {"error": "Pydantic not installed, cannot generate structured plan."}, self.plugin_id, correlation_id)
            return None, "Pydantic not installed; cannot generate structured plan."

        DynamicPlanModelForSchema = self._create_dynamic_plan_model(candidate_tool_ids)

        is_gemini_provider = False
        planner_id = self._planner_llm_id or self._genie._config.default_llm_provider_id # type: ignore
        if planner_id and ("gemini" in planner_id):
            is_gemini_provider = True

        schema_instruction_str: str
        if is_gemini_provider:
            schema_instruction_str = """
Your output MUST be a JSON object with two keys: "overall_reasoning" (string or null) and "plan" (a list of objects).
Each object in the "plan" list MUST have the following keys:
- "thought": A string explaining the step.
- "tool_id": A string matching one of the available ToolIDs.
- "params": A JSON-encoded STRING containing a dictionary of parameters for the tool.
- "output_variable_name": An optional string to name the output of this step.
"""
            logger.debug("Using explicit textual schema instruction for Gemini provider.")
        else:
            plan_schema_dict = DynamicPlanModelForSchema.model_json_schema()
            plan_schema_json_str_for_prompt = json.dumps(plan_schema_dict, indent=2)
            schema_instruction_str = f"The final JSON object must conform to this Pydantic schema:\n{plan_schema_json_str_for_prompt}"

        feedback_str = ""
        if previous_attempt_feedback:
            feedback_str = f"\n---\nPREVIOUS ATTEMPT FAILED. PLEASE CORRECT THE FOLLOWING AND GENERATE A NEW, VALID PLAN.\nFeedback: {previous_attempt_feedback}\n---\n"

        prompt_data = {"goal": goal, "tool_definitions": tool_definitions, "schema_instruction": schema_instruction_str, "previous_attempt_feedback": feedback_str}

        planner_engine_id = self._planner_prompt_template_engine_id

        planner_prompt_str_any = await self._genie.prompts.render_prompt(
            template_content=self._DEFAULT_PLANNER_PROMPT, # Use the class attribute
            data=prompt_data,
            template_engine_id=planner_engine_id
        )
        planner_prompt_str = str(planner_prompt_str_any)


        if not planner_prompt_str or planner_prompt_str == "None": # Check for "None" string if fallback returns it
            await self._genie.observability.trace_event("rewoo.plan.error", {"error": "Planner prompt string rendering failed or returned None string."}, self.plugin_id, correlation_id)
            return None, None

        planner_messages: List[ChatMessage] = [{"role": "user", "content": planner_prompt_str}]
        raw_planner_output: Optional[str] = None
        await self._genie.observability.trace_event("rewoo.plan.prompt_ready", {"messages_content_snippet": planner_prompt_str[:500], "has_feedback": previous_attempt_feedback is not None, "engine_used": planner_engine_id or "GenieDefault"}, self.plugin_id, correlation_id)

        for attempt in range(self._max_plan_retries + 1):
            try:
                llm_response = await self._genie.llm.chat(messages=planner_messages, provider_id=self._planner_llm_id, output_schema=DynamicPlanModelForSchema)
                raw_planner_output = llm_response["message"]["content"] or ""
                await self._genie.observability.trace_event("rewoo.plan.raw_llm_output", {"attempt": attempt + 1, "output": raw_planner_output}, self.plugin_id, correlation_id)
                parsed_plan = await self._genie.llm.parse_output(llm_response, schema=DynamicPlanModelForSchema)

                if isinstance(parsed_plan, DynamicPlanModelForSchema):
                    for step_idx, step in enumerate(parsed_plan.plan): # type: ignore
                        if step.tool_id not in candidate_tool_ids:
                            raise ValueError(f"Plan validation failed: Step {step_idx + 1} used an invalid tool_id '{step.tool_id}'. Valid tools: {candidate_tool_ids}")
                        if step.output_variable_name and not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", step.output_variable_name):
                            raise ValueError(f"Plan validation failed: Step {step_idx + 1} has an invalid output_variable_name '{step.output_variable_name}'. Must be valid Python identifier.")
                    await self._genie.observability.trace_event("rewoo.plan.validation_success", {"attempt": attempt + 1}, self.plugin_id, correlation_id)
                    return parsed_plan, raw_planner_output
                else:
                    raise ValueError(f"Parsed output was not an instance of the expected dynamic Pydantic model. Type: {type(parsed_plan)}")
            except (PydanticValidationErrorImport, ValueError) as e_val:
                current_error_feedback = str(e_val)
                if isinstance(e_val, PydanticValidationErrorImport):
                    try: current_error_feedback = json.dumps(e_val.errors(), indent=2)
                    except Exception: pass
                await self._genie.observability.trace_event("rewoo.plan.retry", {"attempt": attempt + 1, "error": current_error_feedback, "raw_output_if_any": raw_planner_output}, self.plugin_id, correlation_id)
                if attempt >= self._max_plan_retries:
                    await self._genie.observability.trace_event("rewoo.plan.error", {"error": f"Plan generation failed after retries: {current_error_feedback}"}, self.plugin_id, correlation_id)
                    return None, raw_planner_output

                prompt_data["previous_attempt_feedback"] = f"\n---\nPREVIOUS ATTEMPT FAILED. PLEASE CORRECT THE FOLLOWING AND GENERATE A NEW, VALID PLAN.\nError: {current_error_feedback}\nRaw Output from last attempt (if any):\n{raw_planner_output}\n---\n"
                planner_prompt_str_any_retry = await self._genie.prompts.render_prompt(template_content=self._DEFAULT_PLANNER_PROMPT, data=prompt_data, template_engine_id=planner_engine_id)
                planner_prompt_str_retry = str(planner_prompt_str_any_retry)
                planner_messages = [{"role": "user", "content": planner_prompt_str_retry}]
                await asyncio.sleep(1)
            except Exception as e:
                current_error_feedback = str(e)
                await self._genie.observability.trace_event("rewoo.plan.retry", {"attempt": attempt + 1, "error": current_error_feedback, "raw_output_if_any": raw_planner_output, "exc_info": True}, self.plugin_id, correlation_id)
                if attempt >= self._max_plan_retries:
                    await self._genie.observability.trace_event("rewoo.plan.error", {"error": f"Plan generation failed after retries with unexpected error: {current_error_feedback}"}, self.plugin_id, correlation_id)
                    return None, raw_planner_output

                prompt_data["previous_attempt_feedback"] = f"\n---\nPREVIOUS ATTEMPT FAILED WITH AN UNEXPECTED ERROR. PLEASE TRY AGAIN, ENSURING THE PLAN IS VALID.\nError: {current_error_feedback}\nRaw Output from last attempt (if any):\n{raw_planner_output}\n---\n"
                planner_prompt_str_any_retry_exc = await self._genie.prompts.render_prompt(template_content=self._DEFAULT_PLANNER_PROMPT, data=prompt_data, template_engine_id=planner_engine_id)
                planner_prompt_str_retry_exc = str(planner_prompt_str_any_retry_exc)
                planner_messages = [{"role": "user", "content": planner_prompt_str_retry_exc}]
                await asyncio.sleep(1)
        return None, raw_planner_output

    async def _process_step_result_for_evidence(
        self,
        step_model_dict: Dict[str, Any],
        tool_result: Any,
        tool_error: Optional[str],
        original_user_goal: str,
        correlation_id: Optional[str]
    ) -> ExecutionEvidence:
        evidence_item = ExecutionEvidence(
            step=step_model_dict,
            result=tool_result,
            error=tool_error,
            detailed_summary_or_extraction=None,
            source_details=None
        )
        tool_id = step_model_dict.get("tool_id", "unknown_tool")

        if tool_error:
            evidence_item["detailed_summary_or_extraction"] = f"Step failed with error: {tool_error}"
            evidence_item["source_details"] = {"type": "error", "identifier": tool_id, "title": "Step Execution Error"}
            return evidence_item

        evidence_item["source_details"] = {"type": "tool_output", "identifier": tool_id, "title": f"Output of {tool_id}"}

        if tool_id == "web_page_content_extractor" and isinstance(tool_result, dict):
            content = tool_result.get("content", "")
            url = tool_result.get("url", "Unknown URL")
            title = tool_result.get("title", url)
            evidence_item["source_details"] = {"type": "web_page", "identifier": url, "title": title}
            if len(content) > 8000: # Increased threshold for summarization
                extraction_prompt = (
                    f"Original Goal: {original_user_goal}\n"
                    f"Reason for fetching this page (from plan): {step_model_dict.get('reasoning', 'N/A')}\n"
                    f"Extract the most relevant information (up to 1000 words) from the following web page content "
                    f"that helps achieve the goal and matches the reasoning. Preserve key facts, figures, and context. "
                    f"If the content is short, return it as is.\n\nWeb Page Content:\n{content[:15000]}..." # Increased context to LLM
                )
                try:
                    # Use the configured solver LLM for consistency, or default if not set
                    response = await self._genie.llm.generate(extraction_prompt, provider_id=self._solver_llm_id)
                    evidence_item["detailed_summary_or_extraction"] = response.get("text", content[:4000] + "... (fallback truncation)").strip()
                except Exception as e:
                    await self._genie.observability.trace_event("rewoo.evidence_processing.error", {"tool_id": tool_id, "error": str(e)}, self.plugin_id, correlation_id)
                    evidence_item["detailed_summary_or_extraction"] = content[:4000] + "... (extraction failed, truncated)"
            else:
                evidence_item["detailed_summary_or_extraction"] = content
        elif tool_id in ["community_google_search", "intelligent_search_aggregator_v1", "arxiv_search_tool"] and isinstance(tool_result, dict):
            results_list = tool_result.get("results", [])
            summary_items = []
            for i, res_item in enumerate(results_list[:3]): # Limit to top 3 for brevity in evidence
                if isinstance(res_item, dict):
                    title = res_item.get("title", "N/A")
                    snippet = res_item.get("snippet", res_item.get("summary", res_item.get("description", res_item.get("snippet_or_summary", ""))))
                    url = res_item.get("url", res_item.get("link", res_item.get("pdf_url", "")))
                    summary_items.append(f"Item {i+1}:\n  Title: {title}\n  URL/ID: {url}\n  Snippet: {snippet}\n---")
            evidence_item["detailed_summary_or_extraction"] = "\n".join(summary_items) if summary_items else "No results found or results in unexpected format."
            evidence_item["source_details"] = {"type": f"{tool_id}_results", "identifier": step_model_dict.get("params", {}).get("query", "N/A"), "title": f"Search for: {step_model_dict.get('params', {}).get('query', 'N/A')}"}
        elif tool_id == "custom_text_parameter_extractor" and isinstance(tool_result, dict):
            evidence_item["detailed_summary_or_extraction"] = tool_result # The direct output is the extraction
            evidence_item["source_details"] = {"type": "data_extraction", "identifier": tool_id, "title": f"Extracted parameters: {list(tool_result.keys())}"}
        elif tool_id == "calculator_tool" and isinstance(tool_result, dict):
            if tool_result.get("error_message"): evidence_item["detailed_summary_or_extraction"] = f"Calculation Error: {tool_result['error_message']}"
            else: evidence_item["detailed_summary_or_extraction"] = f"Calculation Result: {tool_result.get('result')}"
            evidence_item["source_details"] = {"type": "calculation", "identifier": tool_id, "title": f"Operation: {step_model_dict.get('params', {}).get('operation')}"}
        elif tool_id == "discussion_sentiment_summarizer" and isinstance(tool_result, dict):
            evidence_item["detailed_summary_or_extraction"] = tool_result # The direct output is the summary
            evidence_item["source_details"] = {"type": "sentiment_analysis", "identifier": tool_id, "title": "Sentiment Summary"}
        elif tool_id == "oss_model_release_impact_analyzer" and isinstance(tool_result, dict):
            evidence_item["detailed_summary_or_extraction"] = tool_result
            evidence_item["source_details"] = {"type": "rag_analysis", "identifier": tool_id, "title": "OSS Model Impact Analysis"}
        else: # Generic fallback for other tools
            if isinstance(tool_result, (dict, list)):
                try:
                    json_str = json.dumps(tool_result, default=str)
                    evidence_item["detailed_summary_or_extraction"] = json_str[:4000] + ("... (truncated)" if len(json_str) > 4000 else "")
                except Exception:
                    evidence_item["detailed_summary_or_extraction"] = str(tool_result)[:4000] + ("... (truncated)" if len(str(tool_result)) > 4000 else "")
            else:
                evidence_item["detailed_summary_or_extraction"] = str(tool_result)[:4000] + ("... (truncated)" if len(str(tool_result)) > 4000 else "")
        return evidence_item

    async def _synthesize_answer(self, goal: str, plan: PydanticBaseModelImport, evidence: List[ExecutionEvidence], correlation_id: Optional[str]) -> Tuple[str, str]:
        plan_data_for_prompt = plan.model_dump() if PYDANTIC_INSTALLED_FOR_REWOO_PROCESSOR and isinstance(plan, PydanticBaseModelImport) else dict(plan) # type: ignore

        prompt_data = {"goal": goal, "plan": plan_data_for_prompt, "evidence": evidence}

        solver_engine_id = self._solver_prompt_template_engine_id

        solver_prompt_str_any = await self._genie.prompts.render_prompt(
            template_content=self._DEFAULT_SOLVER_PROMPT_V2,
            data=prompt_data,
            template_engine_id=solver_engine_id
        )
        solver_prompt_str = str(solver_prompt_str_any)

        if not solver_prompt_str or solver_prompt_str == "None":
            await self._genie.observability.trace_event("rewoo.solve.error", {"error": "Solver prompt rendering failed or returned None string."}, self.plugin_id, correlation_id)
            return "Error: Could not render the final answer prompt.", ""

        await self._genie.observability.trace_event("rewoo.solve.prompt_ready", {"prompt_length": len(solver_prompt_str), "prompt_snippet": solver_prompt_str[:500], "engine_used": solver_engine_id or "GenieDefault"}, self.plugin_id, correlation_id)
        raw_solver_output: str = ""
        try:
            response = await self._genie.llm.chat(messages=[{"role": "user", "content": solver_prompt_str}], provider_id=self._solver_llm_id)
            raw_solver_output = response["message"]["content"] or ""
            await self._genie.observability.trace_event("rewoo.solve.raw_llm_output", {"output": raw_solver_output}, self.plugin_id, correlation_id)
            final_answer = raw_solver_output.strip() or "The solver LLM returned an empty response."
            await self._genie.observability.trace_event("rewoo.solve.success", {"answer_length": len(final_answer)}, self.plugin_id, correlation_id)
            return final_answer, raw_solver_output
        except Exception as e:
            await self._genie.observability.trace_event("rewoo.solve.error", {"error": str(e), "raw_output_if_any": raw_solver_output, "exc_info": True}, self.plugin_id, correlation_id)
            return f"Error: The final answer could not be synthesized. Reason: {e}", raw_solver_output

    async def process_command(
        self, command: str, conversation_history: Optional[List[ChatMessage]] = None, correlation_id: Optional[str] = None
    ) -> CommandProcessorResponse:
        all_tools = await self._genie._tool_manager.list_tools(enabled_only=True) # type: ignore
        if not all_tools:
            await self._genie.observability.trace_event("rewoo.process.error", {"error": "NoToolsAvailable"}, self.plugin_id, correlation_id)
            return {"error": "No tools available for planning.", "llm_thought_process": "No tools available."}

        candidate_tool_ids = [t.identifier for t in all_tools]
        tool_definitions_list = [str(await self._genie._tool_manager.get_formatted_tool_definition(t_id, self._tool_formatter_id)) for t_id in candidate_tool_ids] # type: ignore
        tool_definitions_str = "\n\n".join(filter(None, tool_definitions_list))
        if not tool_definitions_str:
            await self._genie.observability.trace_event("rewoo.process.error", {"error": "NoToolDefinitionsFormatted"}, self.plugin_id, correlation_id)
            return {"error": "Could not format any tool definitions for planning.", "llm_thought_process": "Tool definition formatting failed."}

        plan_model, raw_planner_output = await self._generate_plan(command, tool_definitions_str, candidate_tool_ids, correlation_id)
        raw_outputs_for_response: Dict[str, Any] = {}
        if raw_planner_output: raw_outputs_for_response["planner_llm_output"] = raw_planner_output

        if not plan_model:
            return {"error": "Failed to generate a valid execution plan.", "raw_response": raw_outputs_for_response}

        evidence: List[ExecutionEvidence] = []
        scratchpad: Dict[str, Any] = {"outputs": {}} # Ensure 'outputs' key exists

        for step_model in plan_model.plan: # type: ignore
            step_dict_for_evidence = step_model.model_dump()
            await self._genie.observability.trace_event("rewoo.step.execute.start", {"tool_id": step_model.tool_id, "params": step_model.params, "reasoning": step_model.thought}, self.plugin_id, correlation_id)

            resolved_params: Dict[str, Any] = {}
            step_execution_error: Optional[str] = None
            parameter_resolution_failed = False

            try:
                if not isinstance(step_model.params, str):
                    raise ValueError(f"The 'params' field for step '{step_model.tool_id}' must be a JSON-encoded string, but got type {type(step_model.params)}.")
                params_dict = json.loads(step_model.params)
                if not isinstance(params_dict, dict):
                    raise ValueError(f"The decoded 'params' string for step '{step_model.tool_id}' did not yield a dictionary.")
                resolved_params = resolve_placeholders(params_dict, scratchpad)
                await self._genie.observability.trace_event("rewoo.step.params_resolved", {"tool_id": step_model.tool_id, "original_params": step_model.params, "resolved_params": resolved_params}, self.plugin_id, correlation_id)
            except (ValueError, json.JSONDecodeError) as e_resolve:
                step_execution_error = f"Parameter resolution/parsing failed for tool '{step_model.tool_id}': {e_resolve}"
                parameter_resolution_failed = True
                await self._genie.observability.trace_event("rewoo.step.param_resolution.error", {"tool_id": step_model.tool_id, "params": step_model.params, "error": str(e_resolve)}, self.plugin_id, correlation_id)

            tool_result: Any = None
            if not parameter_resolution_failed:
                target_tool: Optional["Tool"] = await self._genie._tool_manager.get_tool(step_model.tool_id) # type: ignore
                if target_tool:
                    tool_metadata = await target_tool.get_metadata()
                    input_schema = tool_metadata.get("input_schema", {})
                    required_params_from_schema = set(input_schema.get("required", []))
                    properties_schema = input_schema.get("properties", {})

                    for param_name, resolved_value in resolved_params.items():
                        if resolved_value is None and param_name in required_params_from_schema:
                            param_prop_schema = properties_schema.get(param_name, {})
                            param_type = param_prop_schema.get("type")
                            is_optional_by_type = False
                            if isinstance(param_type, list) and "null" in param_type:
                                is_optional_by_type = True
                            elif param_type == "null":
                                is_optional_by_type = True

                            if not is_optional_by_type:
                                step_execution_error = f"Parameter '{param_name}' for tool '{step_model.tool_id}' is required but resolved to None from a placeholder. Previous step might have failed or returned no value."
                                await self._genie.observability.trace_event("rewoo.step.required_param_is_none", {"tool_id": step_model.tool_id, "param_name": param_name, "error": step_execution_error}, self.plugin_id, correlation_id)
                                break
                    if step_execution_error: # If a required param was None, skip execution
                        pass
                else:
                    step_execution_error = f"Tool '{step_model.tool_id}' not found in ToolManager before execution."
                    await self._genie.observability.trace_event("rewoo.step.tool_not_found_pre_exec", {"tool_id": step_model.tool_id, "error": step_execution_error}, self.plugin_id, correlation_id)

            if not step_execution_error and not parameter_resolution_failed:
                try:
                    tool_context = {"original_user_goal": command, "current_step_reasoning": step_model.thought}
                    tool_result = await self._genie.execute_tool(step_model.tool_id, context=tool_context, **resolved_params)

                    # Check if the result is a StructuredError dictionary.
                    if isinstance(tool_result, dict) and "type" in tool_result and "message" in tool_result:
                        step_execution_error = f"Tool execution failed: {tool_result.get('type')} - {tool_result.get('message')}"
                        tool_result = None # Nullify the result to prevent it from being stored

                    if not step_execution_error:
                        # Only store to scratchpad on success
                        await self._genie.observability.trace_event("rewoo.step.execute.success", {"tool_id": step_model.tool_id}, self.plugin_id, correlation_id)
                        if step_model.output_variable_name:
                            scratchpad["outputs"][step_model.output_variable_name] = tool_result
                            await self._genie.observability.trace_event("rewoo.step.output_stored", {"variable_name": step_model.output_variable_name, "tool_id": step_model.tool_id, "output_snippet": str(tool_result)[:200]}, self.plugin_id, correlation_id)
                    else:
                        # Log the error that was detected from the tool's result
                        await self._genie.observability.trace_event("rewoo.step.execute.tool_reported_error", {"tool_id": step_model.tool_id, "error": step_execution_error}, self.plugin_id, correlation_id)

                except Exception as e_exec:
                    step_execution_error = f"Error executing tool '{step_model.tool_id}' or during its validation: {e_exec}"
                    tool_result = None # Nullify result on exception
                    await self._genie.observability.trace_event("rewoo.step.execute.error", {"tool_id": step_model.tool_id, "error": step_execution_error, "exc_info": True}, self.plugin_id, correlation_id)

            current_evidence_item = await self._process_step_result_for_evidence(step_model_dict=step_dict_for_evidence, tool_result=tool_result, tool_error=step_execution_error, original_user_goal=command, correlation_id=correlation_id)
            evidence.append(current_evidence_item)

            if step_execution_error:
                await self._genie.observability.trace_event("rewoo.plan.execution_halted", {"failed_step_tool_id": step_model.tool_id, "error": step_execution_error}, self.plugin_id, correlation_id)
                break

        final_answer, raw_solver_llm_output = await self._synthesize_answer(command, plan_model, evidence, correlation_id)
        if raw_solver_llm_output: raw_outputs_for_response["solver_llm_output"] = raw_solver_llm_output

        final_solver_prompt_for_trace_any = await self._genie.prompts.render_prompt(
            template_content=self._DEFAULT_SOLVER_PROMPT_V2,
            data={"goal": command, "plan": plan_model.model_dump() if PYDANTIC_INSTALLED_FOR_REWOO_PROCESSOR and isinstance(plan_model, PydanticBaseModelImport) else dict(plan_model), "evidence": evidence},
            template_engine_id=self._solver_prompt_template_engine_id
        )
        final_solver_prompt_for_trace = str(final_solver_prompt_for_trace_any)
        await self._genie.observability.trace_event("rewoo.solver.prompt_sent", {"prompt_length": len(final_solver_prompt_for_trace) if final_solver_prompt_for_trace else 0}, self.plugin_id, correlation_id)


        thought_process_str: str
        try:
            plan_dump_for_thought = plan_model.model_dump() if PYDANTIC_INSTALLED_FOR_REWOO_PROCESSOR and isinstance(plan_model, PydanticBaseModelImport) else dict(plan_model) # type: ignore
            serializable_evidence = []
            for ev_item in evidence:
                item_copy = ev_item.copy()
                if "result" in item_copy and not isinstance(item_copy["result"], (str, int, float, bool, list, dict, type(None))):
                    item_copy["result"] = f"<Unserializable: {type(item_copy['result']).__name__}>"
                if "detailed_summary_or_extraction" in item_copy and                    not isinstance(item_copy["detailed_summary_or_extraction"], (str, int, float, bool, list, dict, type(None))):
                    item_copy["detailed_summary_or_extraction"] = f"<Unserializable: {type(item_copy['detailed_summary_or_extraction']).__name__}>"
                serializable_evidence.append(item_copy)
            thought_process_str = json.dumps({"plan": plan_dump_for_thought, "evidence": serializable_evidence}, indent=2, default=str)
        except Exception as e_json_dump:
            logger.error(f"{self.plugin_id}: Failed to serialize plan/evidence for llm_thought_process: {e_json_dump}", exc_info=True)
            thought_process_str = f"Error serializing thought process: {e_json_dump}. Plan (type: {type(plan_model)}), Evidence items: {len(evidence)}"
            raw_outputs_for_response.setdefault("plan_error_details", str(plan_model))
            raw_outputs_for_response.setdefault("evidence_summary", f"{len(evidence)} items")

        final_response: CommandProcessorResponse = {
            "final_answer": final_answer,
            "llm_thought_process": thought_process_str,
            "raw_response": {**raw_outputs_for_response}
        }
        if any(ev.get("error") for ev in evidence):
            final_response["error"] = "One or more steps in the plan failed. See thought process for details."
        return final_response