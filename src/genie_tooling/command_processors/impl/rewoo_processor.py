# src/genie_tooling/command_processors/impl/rewoo_processor.py
import asyncio
import json
import logging
import re
import uuid
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

from genie_tooling.input_validators import InputValidationException
from genie_tooling.utils.placeholder_resolution import resolve_placeholders

# Conditional Pydantic import for type safety and optional dependency
try:
    from pydantic import BaseModel as PydanticBaseModelImport
    from pydantic import Field as PydanticFieldImport
    from pydantic import ValidationError as PydanticValidationErrorImport
    from pydantic import create_model as create_model_import

    PYDANTIC_INSTALLED_FOR_REWOO_PROCESSOR = True
except ImportError:

    def _pydantic_field_fallback(**kwargs: Any) -> Dict[str, Any]:
        return {}

    def _create_model_fallback(*args: Any, **kwargs: Any) -> Type[Dict[Any, Any]]:
        return type("FallbackModel", (dict,), {})

    PydanticBaseModelImport = object  # type: ignore
    PydanticFieldImport = _pydantic_field_fallback  # type: ignore
    create_model_import = _create_model_fallback # type: ignore
    PydanticValidationErrorImport = ValueError  # type: ignore
    PYDANTIC_INSTALLED_FOR_REWOO_PROCESSOR = False


from genie_tooling.command_processors.abc import CommandProcessorPlugin
from genie_tooling.command_processors.types import CommandProcessorResponse
from genie_tooling.llm_providers.types import ChatMessage

if TYPE_CHECKING:
    from genie_tooling.genie import Genie

logger = logging.getLogger(__name__)


# --- Internal Data Structures ---
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


class AgentRunResult(TypedDict):
    """Internal result type for the _execute_plan method."""

    status: Literal["success", "error"]
    final_output: Any
    evidence: List[ExecutionEvidence]


# --- End of Internal Data Structures ---


class ReWOOCommandProcessorPlugin(CommandProcessorPlugin):
    """
    Implements the ReWOO (Reason-Act) pattern for complex command processing.

    This processor acts as a self-contained agent. It takes a high-level goal and:
    1.  **Planner**: Creates a multi-step plan of tool calls.
    2.  **Executor**: Executes the plan sequentially, gathering evidence.
    3.  **Solver**: Synthesizes the evidence into a final, comprehensive answer.
    """
    plugin_id: str = "rewoo_command_processor_v1"
    description: str = "Implements the ReWOO (Reason-Act) pattern: Plan -> Execute -> Solve."
    _DEFAULT_PLANNER_PROMPT = r"""
You are an expert AI assistant that plans a sequence of tool calls to solve a user's goal.
Do NOT execute the tools. Your only job is to create a plan.
The `tool_id` in each step of your plan MUST exactly match one of the `ToolID` values provided in the "Available Tools" section below.

---
**CRITICAL INSTRUCTION FOR `params` FIELD**:
The `params` field for each step MUST be a valid JSON **object**. Do NOT wrap it in quotes.
The system will handle resolving placeholders like {{ '{{' }}outputs.variable.path{{ '}}' }}.

- **CORRECT Example**: `"params": {"query": "Llama 3 scores"}`
- **CORRECT Example with placeholder**: `"params": {"url": "{{ '{{' }}outputs.search_results.results.0.url{{ '}}' }}"}`
- **INCORRECT Example**: `"params": "{\"query\": \"Llama 3 scores\"}"` (This is a string, not an object)

This is the most critical part of the plan. Ensure the `params` value is a JSON object.
---

If a step's output is needed by a later step, define an `output_variable_name` for that step (e.g., "search_results", "page_content").
Subsequent steps can then reference this output in their `params` using the syntax `{{ '{{' }}outputs.variable_name.path.to.value{{ '}}' }}`.
When constructing the `path.to.value`, you must infer the structure of the tool's output.
- For most tools, the output is a direct dictionary. Example: `{{ '{{' }}outputs.extracted_scores.llama3_8b_instruct_mmlu_score{{ '}}' }}`.
- For search tools (like `intelligent_search_aggregator_v1`), results are a list. To get the first URL: `{{ '{{' }}outputs.search_variable.results.0.url{{ '}}' }}`.
- For the `content_retriever_tool`, the main text is in `{{ '{{' }}outputs.content_variable.content{{ '}}' }}`.

Be mindful of potential `null` or missing values. If a step might fail or return nothing, subsequent steps relying on its output will also fail if they require a value.

---
IMPORTANT RULES FOR TOOL USAGE:
1.  **Search First**: Always start with a search tool like `intelligent_search_aggregator_v1` to gather initial information and URLs.
2.  **Deep Dive with ContentRetriever**: If a search result gives you a URL and you need its full content, **ALWAYS** use the `content_retriever_tool_v1` in a subsequent step. This tool automatically handles both web pages and PDFs. Store its output in a variable (e.g., `page_content`).
3.  **Extract Specifics**: After getting text from `content_retriever_tool_v1`, if you need to pull out specific data points (like numbers, names, dates), use the `custom_text_parameter_extractor` tool in the next step.
4.  **Calculate**: Use the `calculator_tool` for any math operations on numbers you have gathered or extracted.
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

    _DEFAULT_SOLVER_PROMPT_V2 = r"""
You are an expert AI assistant that synthesizes a final answer for a user based on their original goal and the evidence gathered from a series of tool calls.
Your answer should comprehensively address all aspects of the original goal. If the goal has multiple parts or implies multiple questions, address each one clearly.
When using information from the evidence, you MUST cite the source using its step number identifier in markdown format (e.g., [1], [2], [3]).
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
Remember to cite sources like [1] and list them at the end.
Your final response should begin directly with the answer, without any preliminary thoughts or XML tags like <think>.

Answer:
"""
    _genie: Optional["Genie"] = None
    _planner_llm_id: Optional[str]
    _solver_llm_id: Optional[str]
    _tool_formatter_id: str
    _max_plan_retries: int
    _replan_on_step_failure: bool
    _max_replan_attempts: int
    _planner_prompt_template_engine_id: Optional[str]
    _solver_prompt_template_engine_id: Optional[str]
    _min_high_quality_sources: int = 3

    async def setup(self, config: Optional[Dict[str, Any]]) -> None:
        """
        Initializes the ReWOO processor with its configuration.
        """
        await super().setup(config)
        cfg = config or {}
        # FIX: Be lenient during setup. The facade might not be available yet.
        self._genie = cfg.get("genie_facade")
        if not self._genie:
            logger.info(f"[{self.plugin_id}] Genie facade not provided during setup. It must be injected before 'process_command' is called.")

        self._planner_llm_id = cfg.get("planner_llm_provider_id")
        self._solver_llm_id = cfg.get("solver_llm_provider_id")
        self._tool_formatter_id = cfg.get("tool_formatter_id", "compact_text_formatter_plugin_v1")
        self._max_plan_retries = int(cfg.get("max_plan_retries", 1))
        self._replan_on_step_failure = bool(cfg.get("replan_on_step_failure", True))
        self._max_replan_attempts = int(cfg.get("max_replan_attempts", 2))
        self._planner_prompt_template_engine_id = cfg.get("planner_prompt_template_engine_id")
        self._solver_prompt_template_engine_id = cfg.get("solver_prompt_template_engine_id")
        self._min_high_quality_sources = int(cfg.get("min_high_quality_sources", 3))

    def _create_dynamic_plan_model(self, tool_ids: List[str]) -> Type[PydanticBaseModelImport]:
        if not PYDANTIC_INSTALLED_FOR_REWOO_PROCESSOR:
            return dict

        if not tool_ids:
            tool_ids = ["placeholder_tool_id_if_none_available"]

        ToolIDEnumForDynamicModel = Literal[tuple(tool_ids)]  # type: ignore

        DynamicReWOOStep = create_model_import(
            "DynamicReWOOStep",
            step_number=(int, PydanticFieldImport(description="Sequential number of the step, starting from 1.")),
            thought=(str, PydanticFieldImport(description="The reasoning for why this specific tool call is necessary.")),
            tool_id=(
                ToolIDEnumForDynamicModel,
                PydanticFieldImport(description="The identifier of the tool to execute."),
            ),
            params=(
                Any,
                PydanticFieldImport(
                    default_factory=dict,
                    description='A valid JSON object or JSON string of an object for the tool. Example: {"query": "Llama 3 scores"}.',
                ),
            ),
            output_variable_name=(
                Optional[str],
                PydanticFieldImport(
                    None,
                    description="If this step's output should be stored for later use, provide a variable name here (e.g., 'search_results'). Use only letters, numbers, and underscores.",
                    pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$",
                ),
            ),
            __base__=PydanticBaseModelImport,
        )
        DynamicReWOOPlan = create_model_import(
            "DynamicReWOOPlan",
            plan=(
                List[DynamicReWOOStep],
                PydanticFieldImport(description="The sequence of tool calls to execute."),
            ),
            overall_reasoning=(Optional[str], PydanticFieldImport(None, description="Overall reasoning for the plan.")),
            __base__=PydanticBaseModelImport,
        )
        return DynamicReWOOPlan

    async def _generate_plan(
        self,
        goal: str,
        tool_definitions: str,
        candidate_tool_ids: List[str],
        correlation_id: Optional[str],
        previous_attempt_feedback: Optional[str] = None,
    ) -> Tuple[Optional[PydanticBaseModelImport], Optional[str]]:
        if not PYDANTIC_INSTALLED_FOR_REWOO_PROCESSOR:
            await self._genie.observability.trace_event(
                "rewoo.plan.error",
                {"error": "Pydantic not installed, cannot generate structured plan."},
                self.plugin_id,
                correlation_id,
            )
            return None, "Pydantic not installed; cannot generate structured plan."

        DynamicPlanModelForSchema = self._create_dynamic_plan_model(candidate_tool_ids)

        is_gemini_provider = False
        planner_id = self._planner_llm_id or self._genie._config.default_llm_provider_id
        if planner_id and ("gemini" in planner_id):
            is_gemini_provider = True

        schema_instruction_str: str
        if is_gemini_provider:
            schema_instruction_str = """
Your output MUST be a JSON object with two keys: "overall_reasoning" (string or null) and "plan" (a list of objects).
Each object in the "plan" list MUST have the following keys:
- "step_number": An integer for the step sequence.
- "thought": A string explaining the step.
- "tool_id": A string matching one of the available ToolIDs.
- "params": A JSON **object** containing parameters for the tool.
- "output_variable_name": An optional string to name the output of this step.
"""
            logger.debug("Using explicit textual schema instruction for Gemini provider.")
        else:
            plan_schema_dict = DynamicPlanModelForSchema.model_json_schema()
            plan_schema_json_str_for_prompt = json.dumps(plan_schema_dict, indent=2)
            schema_instruction_str = (
                f"The final JSON object must conform to this Pydantic schema:\n{plan_schema_json_str_for_prompt}"
            )

        feedback_str = ""
        if previous_attempt_feedback:
            feedback_str = f"\n---\nPREVIOUS ATTEMPT FAILED. PLEASE CORRECT THE FOLLOWING AND GENERATE A NEW, VALID PLAN.\nFeedback: {previous_attempt_feedback}\n---\n"

        prompt_data = {
            "goal": goal,
            "tool_definitions": tool_definitions,
            "schema_instruction": schema_instruction_str,
            "previous_attempt_feedback": feedback_str,
        }

        planner_engine_id = self._planner_prompt_template_engine_id

        planner_prompt_str_any = await self._genie.prompts.render_prompt(
            template_content=self._DEFAULT_PLANNER_PROMPT,
            data=prompt_data,
            template_engine_id=planner_engine_id,
        )
        planner_prompt_str = str(planner_prompt_str_any)

        if not planner_prompt_str or planner_prompt_str == "None":
            await self._genie.observability.trace_event(
                "rewoo.plan.error",
                {"error": "Planner prompt string rendering failed or returned None string."},
                self.plugin_id,
                correlation_id,
            )
            return None, None

        planner_messages: List[ChatMessage] = [{"role": "user", "content": planner_prompt_str}]
        raw_planner_output: Optional[str] = None
        await self._genie.observability.trace_event(
            "rewoo.plan.prompt_ready",
            {
                "messages_content_snippet": planner_prompt_str[:500],
                "has_feedback": previous_attempt_feedback is not None,
                "engine_used": planner_engine_id or "GenieDefault",
            },
            self.plugin_id,
            correlation_id,
        )

        for attempt in range(self._max_plan_retries + 1):
            try:
                llm_response = await self._genie.llm.chat(
                    messages=planner_messages, provider_id=self._planner_llm_id, output_schema=DynamicPlanModelForSchema
                )
                raw_planner_output = llm_response["message"]["content"] or ""
                await self._genie.observability.trace_event(
                    "rewoo.plan.raw_llm_output",
                    {"attempt": attempt + 1, "output": raw_planner_output},
                    self.plugin_id,
                    correlation_id,
                )
                parsed_plan = await self._genie.llm.parse_output(llm_response, schema=DynamicPlanModelForSchema)

                if isinstance(parsed_plan, DynamicPlanModelForSchema):
                    for step_idx, step in enumerate(parsed_plan.plan):
                        if step.tool_id not in candidate_tool_ids:
                            raise ValueError(
                                f"Plan validation failed: Step {step_idx + 1} used an invalid tool_id '{step.tool_id}'. Valid tools: {candidate_tool_ids}"
                            )
                        if step.output_variable_name and not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", step.output_variable_name):
                            raise ValueError(
                                f"Plan validation failed: Step {step_idx + 1} has an invalid output_variable_name '{step.output_variable_name}'. Must be valid Python identifier."
                            )
                    await self._genie.observability.trace_event(
                        "rewoo.plan.validation_success", {"attempt": attempt + 1}, self.plugin_id, correlation_id
                    )
                    return parsed_plan, raw_planner_output
                else:
                    raise ValueError(
                        f"Parsed output was not an instance of the expected dynamic Pydantic model. Type: {type(parsed_plan)}"
                    )
            except (PydanticValidationErrorImport, ValueError) as e_val:
                current_error_feedback = str(e_val)
                if isinstance(e_val, PydanticValidationErrorImport):
                    try:
                        current_error_feedback = json.dumps(e_val.errors(), indent=2)
                    except Exception as e_json:
                        await self._genie.observability.trace_event(
                            "rewoo.plan.error",
                            {"error": f"Error {e_json}"},
                            self.plugin_id,
                            correlation_id,
                        )
                        pass
                await self._genie.observability.trace_event(
                    "rewoo.plan.retry",
                    {"attempt": attempt + 1, "error": current_error_feedback, "raw_output_if_any": raw_planner_output},
                    self.plugin_id,
                    correlation_id,
                )
                if attempt >= self._max_plan_retries:
                    await self._genie.observability.trace_event(
                        "rewoo.plan.error",
                        {"error": f"Plan generation failed after retries: {current_error_feedback}"},
                        self.plugin_id,
                        correlation_id,
                    )
                    return None, raw_planner_output

                prompt_data[
                    "previous_attempt_feedback"
                ] = f"\n---\nPREVIOUS ATTEMPT FAILED. PLEASE CORRECT THE FOLLOWING AND GENERATE A NEW, VALID PLAN.\nError: {current_error_feedback}\nRaw Output from last attempt (if any):\n{raw_planner_output}\n---\n"
                planner_prompt_str_any_retry = await self._genie.prompts.render_prompt(
                    template_content=self._DEFAULT_PLANNER_PROMPT, data=prompt_data, template_engine_id=planner_engine_id
                )
                planner_prompt_str_retry = str(planner_prompt_str_any_retry)
                planner_messages = [{"role": "user", "content": planner_prompt_str_retry}]
                await asyncio.sleep(1)
            except Exception as e:
                current_error_feedback = str(e)
                await self._genie.observability.trace_event(
                    "rewoo.plan.retry",
                    {
                        "attempt": attempt + 1,
                        "error": current_error_feedback,
                        "raw_output_if_any": raw_planner_output,
                        "exc_info": True,
                    },
                    self.plugin_id,
                    correlation_id,
                )
                if attempt >= self._max_plan_retries:
                    await self._genie.observability.trace_event(
                        "rewoo.plan.error",
                        {"error": f"Plan generation failed after retries with unexpected error: {current_error_feedback}"},
                        self.plugin_id,
                        correlation_id,
                    )
                    return None, raw_planner_output

                prompt_data[
                    "previous_attempt_feedback"
                ] = f"\n---\nPREVIOUS ATTEMPT FAILED WITH AN UNEXPECTED ERROR. PLEASE TRY AGAIN, ENSURING THE PLAN IS VALID.\nError: {current_error_feedback}\nRaw Output from last attempt (if any):\n{raw_planner_output}\n---\n"
                planner_prompt_str_any_retry_exc = await self._genie.prompts.render_prompt(
                    template_content=self._DEFAULT_PLANNER_PROMPT, data=prompt_data, template_engine_id=planner_engine_id
                )
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
        correlation_id: Optional[str],
    ) -> ExecutionEvidence:
        evidence_item = ExecutionEvidence(
            step=step_model_dict,
            result=tool_result,
            error=tool_error,
            detailed_summary_or_extraction=None,
            source_details=None,
        )
        tool_id = step_model_dict.get("tool_id", "unknown_tool")

        if tool_error:
            evidence_item["detailed_summary_or_extraction"] = f"Step failed with error: {tool_error}"
            evidence_item["source_details"] = {"type": "error", "identifier": tool_id, "title": "Step Execution Error"}
            return evidence_item

        params_dict = step_model_dict.get("params_resolved", {})
        query_from_params = params_dict.get("query", "N/A")

        content_to_summarize: Optional[str] = None
        source_details_set = False

        if isinstance(tool_result, dict):
            content_keys = ["content", "text_content"]
            for key in content_keys:
                if isinstance(tool_result.get(key), str):
                    content_to_summarize = tool_result[key]
                    break

        if isinstance(content_to_summarize, str) and len(content_to_summarize) > 8000:
            url = tool_result.get("url", "Unknown URL")
            title = tool_result.get("title", url)
            source_type = "pdf" if "pdf" in (tool_result.get("source_type") or tool_id) else "web_page"
            evidence_item["source_details"] = {"type": source_type, "identifier": url, "title": title}
            source_details_set = True

            extraction_prompt = (
                f"Original Goal: {original_user_goal}\n"
                f"Reason for fetching this page (from plan): {step_model_dict.get('reasoning', 'N/A')}\n"
                f"Extract the most relevant information (up to 10000 words) from the following web page content "
                f"that helps achieve the goal and matches the reasoning. Preserve key facts, figures, and context. "
                f"If the content is short, return it as is.\n\nWeb Page Content:\n{content_to_summarize[:25000]}..."
            )
            try:
                response = await self._genie.llm.generate(extraction_prompt, provider_id=self._solver_llm_id)
                evidence_item["detailed_summary_or_extraction"] = response.get("text", content_to_summarize[:4000] + "... (fallback truncation)").strip()
            except Exception:
                evidence_item["detailed_summary_or_extraction"] = content_to_summarize[:4000] + "... (extraction failed, truncated)"

        if evidence_item["detailed_summary_or_extraction"] is None and tool_id in ["community_google_search", "intelligent_search_aggregator_v1", "arxiv_search_tool"] and isinstance(tool_result, dict):
            results_list = tool_result.get("results", [])
            summary_items = []
            for i, res_item in enumerate(results_list[:3]):
                if isinstance(res_item, dict):
                    title = res_item.get("title", "N/A")
                    snippet = res_item.get("snippet", res_item.get("summary", res_item.get("description", res_item.get("snippet_or_summary", ""))))
                    url = res_item.get("url", res_item.get("link", res_item.get("pdf_url", "")))
                    summary_items.append(f"Item {i+1}:\n  Title: {title}\n  URL/ID: {url}\n  Snippet: {snippet}\n---")
            evidence_item["detailed_summary_or_extraction"] = "\n".join(summary_items) if summary_items else "No results found or results in unexpected format."
            evidence_item["source_details"] = {"type": f"{tool_id}_results", "identifier": query_from_params, "title": f"Search for: {query_from_params}"}
            source_details_set = True

        if evidence_item["detailed_summary_or_extraction"] is None:
            if not source_details_set:
                evidence_item["source_details"] = {"type": "tool_output", "identifier": tool_id, "title": f"Output of {tool_id}"}

            if isinstance(tool_result, (dict, list)):
                try:
                    json_str = json.dumps(tool_result, default=str)
                    evidence_item["detailed_summary_or_extraction"] = json_str[:4000] + ("... (truncated)" if len(json_str) > 4000 else "")
                except Exception:
                    evidence_item["detailed_summary_or_extraction"] = str(tool_result)[:4000] + ("... (truncated)" if len(str(tool_result)) > 4000 else "")
            else:
                evidence_item["detailed_summary_or_extraction"] = str(tool_result)[:4000] + ("... (truncated)" if len(str(tool_result)) > 4000 else "")

        return evidence_item

    async def _is_high_quality_evidence(self, evidence: ExecutionEvidence, goal: str) -> bool:
        if evidence.get("error"):
            return False

        source_type = (evidence.get("source_details") or {}).get("type")
        if source_type == "tool_output":
            return True

        summary = evidence.get("detailed_summary_or_extraction")
        if summary is None:
            return False

        if isinstance(summary, dict):
            return any(value is not None for value in summary.values())

        if isinstance(summary, str):
            summary_lower = summary.lower().strip()
            if len(summary_lower) < 50:
                logger.debug(f"Evidence discarded: content too short ({len(summary_lower)} chars).")
                return False

            low_quality_phrases = ["not relevant", "no information found", "could not extract", "extraction failed", "error"]
            if any(phrase in summary_lower for phrase in low_quality_phrases):
                logger.debug("Evidence discarded: contains low quality phrase.")
                return False

            if source_type in ["web_page", "pdf"]:
                relevance_prompt = f"Original goal: '{goal}'\n\nIs the following text snippet relevant to the goal? Answer ONLY with 'yes' or 'no'.\n\nSnippet: {summary[:1000]}..."
                try:
                    relevance_response = await self._genie.llm.generate(relevance_prompt, temperature=0.0, max_tokens=5)
                    answer_text = relevance_response.get("text")
                    if not answer_text or not isinstance(answer_text, str):
                        answer = "no"
                    else:
                        answer = answer_text.strip().lower()
                    is_relevant = answer.startswith("yes")
                    if not is_relevant:
                        logger.debug(f"Evidence discarded: LLM deemed content irrelevant. Response: '{answer}'.")
                    return is_relevant
                except Exception as e:
                    logger.warning(f"LLM relevance check failed: {e}. Assuming not relevant as a precaution.")
                    return False

            return True

        return False

    async def _synthesize_answer(
        self, goal: str, plan: PydanticBaseModelImport, evidence: List[ExecutionEvidence], correlation_id: Optional[str]
    ) -> Tuple[str, str]:
        plan_data_for_prompt = (
            plan.model_dump()
            if PYDANTIC_INSTALLED_FOR_REWOO_PROCESSOR and isinstance(plan, PydanticBaseModelImport)
            else dict(plan)
        )

        successful_evidence = [item for item in evidence if not item.get("error")]
        prompt_data: Dict[str, Any]

        if not successful_evidence and evidence:
            error_summary = "\n".join([f"- Step {i+1} ({item['step'].get('tool_id', 'N/A')}) failed: {item['error']}" for i, item in enumerate(evidence) if item.get("error")])
            prompt_data = {
                "goal": goal,
                "plan": plan_data_for_prompt,
                "evidence": [{"step": {"tool_id": "System"}, "detailed_summary_or_extraction": f"All plan steps failed. Summary of errors:\n{error_summary}", "source_details": {"type": "error_summary", "identifier": "plan_execution", "title": "All steps failed"}}]
            }
        else:
            prompt_data = {"goal": goal, "plan": plan_data_for_prompt, "evidence": successful_evidence}

        solver_engine_id = self._solver_prompt_template_engine_id

        solver_prompt_str_any = await self._genie.prompts.render_prompt(
            template_content=self._DEFAULT_SOLVER_PROMPT_V2, data=prompt_data, template_engine_id=solver_engine_id
        )
        solver_prompt_str = str(solver_prompt_str_any)

        if not solver_prompt_str or solver_prompt_str == "None":
            return "Error: Could not render the final answer prompt.", ""

        raw_solver_output: str = ""
        try:
            response = await self._genie.llm.chat(
                messages=[{"role": "user", "content": solver_prompt_str}], provider_id=self._solver_llm_id
            )
            raw_solver_output = response["message"]["content"] or ""
            final_answer = raw_solver_output.strip() or "The solver LLM returned an empty response."
            return final_answer, raw_solver_output
        except Exception as e:
            return f"Error: The final answer could not be synthesized. Reason: {e}", raw_solver_output

    async def _execute_plan(
        self, plan_model: PydanticBaseModelImport, goal: str, correlation_id: Optional[str]
    ) -> AgentRunResult:
        evidence: List[ExecutionEvidence] = []
        scratchpad: Dict[str, Any] = {"outputs": {}}
        current_plan = list(plan_model.plan)

        for i, step in enumerate(current_plan):
            step_dict_for_evidence = step.model_dump()
            step_execution_error: Optional[str] = None
            tool_result: Any = None
            resolved_params: Dict[str, Any] = {}

            await self._genie.observability.trace_event(
                "rewoo.step.start",
                {
                    "step_number": i + 1,
                    "tool_id": step.tool_id,
                    "params_template": step.params,
                },
                self.plugin_id,
                correlation_id,
            )

            try:
                params_with_placeholders = step.params or {}

                if isinstance(params_with_placeholders, str):
                    try:
                        params_with_placeholders = json.loads(params_with_placeholders)
                    except json.JSONDecodeError as e_json:
                        raise TypeError(f"The 'params' field was a string but could not be parsed as JSON: {e_json!s}") from e_json
                if not isinstance(params_with_placeholders, dict):
                    raise TypeError(f"The 'params' field for step {i+1} must be a dictionary or a JSON string representing a dictionary, got {type(step.params)}.")

                resolved_params = resolve_placeholders(params_with_placeholders, scratchpad)
                step_dict_for_evidence["params_resolved"] = resolved_params

                tool_context = {"original_user_goal": goal, "current_step_reasoning": step.thought}
                tool_result = await self._genie.execute_tool(step.tool_id, context=tool_context, **resolved_params)

                if isinstance(tool_result, str) and tool_result.lower().strip().startswith("error"):
                    step_execution_error = tool_result
                elif isinstance(tool_result, dict):
                    if tool_result.get("type") and "Error" in tool_result["type"]:
                        step_execution_error = f"Invocation strategy error: {tool_result.get('message', 'No message')}"
                    elif tool_result.get("error") is not None:
                        step_execution_error = f"Tool reported error: {tool_result['error']}"

            except (ValueError, TypeError, KeyError, InputValidationException) as e_prep:
                step_execution_error = f"Error during step {step.step_number} preparation or validation: {e_prep!s}"
                tool_result = None
            except Exception as e_exec:
                step_execution_error = f"Error during step {step.step_number} execution: {e_exec!s}"
                tool_result = None

            evidence_item = await self._process_step_result_for_evidence(
                step_model_dict=step_dict_for_evidence,
                tool_result=tool_result,
                tool_error=step_execution_error,
                original_user_goal=goal,
                correlation_id=correlation_id,
            )

            is_quality = await self._is_high_quality_evidence(evidence_item, goal)
            if step.output_variable_name:
                if not step_execution_error and is_quality:
                    scratchpad["outputs"][step.output_variable_name] = tool_result
                else:
                    scratchpad["outputs"][step.output_variable_name] = None
                    if not step_execution_error:
                        evidence_item["error"] = "Content discarded as irrelevant or low-quality."
                        logger.warning(
                            f"Step {i+1} ({step.tool_id}) produced low-quality evidence. Continuing with plan."
                        )

            evidence.append(evidence_item)
            if step_execution_error:
                await self._genie.observability.trace_event("rewoo.step.failed", {"step": i + 1, "error": step_execution_error}, self.plugin_id, correlation_id)
                return {"status": "error", "final_output": f"Execution failed at step {i+1}: {step_execution_error}", "evidence": evidence}

        return {"status": "success", "final_output": None, "evidence": evidence}

    async def process_command(
        self, command: str, conversation_history: Optional[List[ChatMessage]] = None, correlation_id: Optional[str] = None
    ) -> CommandProcessorResponse:
        # FIX: Check for the facade at the beginning of the execution logic.
        if not self._genie:
            return {"error": "ReWOO processor not properly initialized with Genie facade."}

        all_tools = await self._genie._tool_manager.list_tools(enabled_only=True)
        if not all_tools:
            return {"error": "No tools available for planning."}

        correlation_id = correlation_id or str(uuid.uuid4())
        candidate_tool_ids = [t.identifier for t in all_tools]
        tool_definitions_str = "\n\n".join(
            filter(None, [
                str(await self._genie._tool_manager.get_formatted_tool_definition(t_id, self._tool_formatter_id))
                for t_id in candidate_tool_ids
            ])
        )

        plan_model: Optional[PydanticBaseModelImport] = None
        raw_planner_output: Optional[str] = None
        execution_result: AgentRunResult = {"status": "error", "final_output": "No execution occurred.", "evidence": []}

        for attempt in range(self._max_replan_attempts + 1):
            feedback_for_planner: Optional[str] = None
            if attempt > 0:
                error_summary = ". ".join(filter(None, [ev.get("error") for ev in execution_result.get("evidence", [])]))
                feedback_for_planner = (
                    f"Previous attempt (attempt {attempt}) did not yield enough high-quality information "
                    f"({self._min_high_quality_sources} required). Summary of issues: {error_summary}. "
                    "Please create a new plan with different search terms or a different approach to gather more relevant sources."
                )

            plan_model, raw_planner_output = await self._generate_plan(
                command, tool_definitions_str, candidate_tool_ids, correlation_id, feedback_for_planner
            )
            if not plan_model:
                return {"error": "Failed to generate a valid execution plan.", "raw_response": {"planner_llm_output": raw_planner_output}}

            execution_result = await self._execute_plan(plan_model, command, correlation_id)
            if execution_result["status"] == "error" and not self._replan_on_step_failure:
                break

            high_quality_evidence = [ev for ev in execution_result.get("evidence", []) if await self._is_high_quality_evidence(ev, command)]

            if len(high_quality_evidence) >= self._min_high_quality_sources:
                await self._genie.observability.trace_event("rewoo.research_loop.success", {"attempt": attempt + 1, "quality_sources": len(high_quality_evidence)}, self.plugin_id, correlation_id)
                break
            elif attempt < self._max_replan_attempts:
                await self._genie.observability.trace_event("rewoo.research_loop.retry", {"attempt": attempt + 1, "quality_sources": len(high_quality_evidence), "required": self._min_high_quality_sources}, self.plugin_id, correlation_id)
                await asyncio.sleep(1)
            else:
                await self._genie.observability.trace_event("rewoo.research_loop.max_attempts", {"attempt": attempt + 1, "quality_sources": len(high_quality_evidence)}, self.plugin_id, correlation_id)
                break

        if not plan_model:
             return {"error": "Internal error: Plan became None after research loop."}
        final_answer, raw_solver_output = await self._synthesize_answer(command, plan_model, execution_result["evidence"], correlation_id)

        try:
            thought_process_data = {
                "plan": plan_model.model_dump(),
                "evidence": execution_result.get("evidence", [])
            }
            thought_process_str = json.dumps(thought_process_data, indent=2, default=str)
        except Exception:
            thought_process_str = "Could not serialize thought process."

        raw_outputs_for_response = {
            "planner_llm_output": raw_planner_output,
            "solver_llm_output": raw_solver_output
        }

        final_response: CommandProcessorResponse = {
            "final_answer": final_answer,
            "llm_thought_process": thought_process_str,
            "raw_response": raw_outputs_for_response,
        }
        if execution_result.get("status") == "error":
            final_response["error"] = execution_result.get("final_output", "One or more steps failed during execution.")

        return final_response