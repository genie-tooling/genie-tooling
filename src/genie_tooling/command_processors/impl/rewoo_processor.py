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
)

from pydantic import BaseModel, Field, create_model

from genie_tooling.command_processors.abc import CommandProcessorPlugin
from genie_tooling.command_processors.types import CommandProcessorResponse
from genie_tooling.llm_providers.types import ChatMessage

if TYPE_CHECKING:
    from genie_tooling.genie import Genie

logger = logging.getLogger(__name__)

# --- Pydantic Models for Structured Planning ---
class ReWOOStep(BaseModel):
    thought: str = Field(description="The reasoning for why this specific tool call is necessary.")
    tool_id: str = Field(description="The identifier of the tool to execute.")
    params: Dict[str, Any] = Field(default_factory=dict, description="The parameters to pass to the tool.")

class ReWOOPlan(BaseModel):
    plan: List[ReWOOStep] = Field(description="The sequence of tool calls to execute to gather evidence.")

# --- Internal Data Structure for Execution ---
class ExecutionEvidence(TypedDict):
    step: Dict[str, Any]
    result: Any
    error: Optional[str]

# --- The Plugin Implementation ---
class ReWOOCommandProcessorPlugin(CommandProcessorPlugin):
    plugin_id: str = "rewoo_command_processor_v1"
    description: str = "Implements the ReWOO (Reason-Act) pattern: Plan -> Execute -> Solve."

    _DEFAULT_PLANNER_PROMPT = """
You are an expert AI assistant that plans a sequence of tool calls to solve a user's goal.
Do NOT execute the tools. Your only job is to create a plan.
The `tool_id` in each step of your plan MUST exactly match one of the `ToolID` values provided in the "Available Tools" section below.

The final JSON object must conform to this Pydantic schema:
{{ ReWOOPlan.model_json_schema() | tojson(indent=2) }}

Available Tools:
---
{{ tool_definitions }}
---

User's Goal: "{{ goal }}"

Now, generate the plan as a single JSON object using the exact ToolID for any chosen tool.
"""

    _DEFAULT_SOLVER_PROMPT = """
You are an expert AI assistant that synthesizes a final answer for a user based on their original goal and the evidence gathered from a series of tool calls.

Original Goal: "{{ goal }}"

The following plan was executed:
---
{% for step in plan.plan %}
Step {{ loop.index }}:
  Thought: {{ step.thought }}
  Action: {{ step.tool_id }}[{{ step.params | tojson }}]
{% endfor %}
---

The following evidence was gathered from executing the plan:
---
{% for item in evidence %}
[Evidence from Step {{ loop.index }} for Action: {{ item.step.tool_id }}]
{% if item.error %}
Execution Error: {{ item.error }}
{% else %}
Execution Result: {{ item.result | tojson(indent=2) }}
{% endif %}
{% endfor %}
---

Based on the original goal and the gathered evidence, provide a comprehensive, final answer to the user.
Do not mention the tools or the planning process in your final answer.
Your final response should begin directly with the answer, without any preliminary thoughts or XML tags like <think>.
Answer:
"""
    _genie: "Genie"
    _planner_llm_id: Optional[str]
    _solver_llm_id: Optional[str]
    _tool_formatter_id: str
    _max_plan_retries: int

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

    def _create_dynamic_plan_model(self, tool_ids: List[str]) -> Type[BaseModel]:
        if not tool_ids:
            tool_ids = [""] 
        
        ToolIDEnum = Literal[tuple(tool_ids)] # type: ignore
        DynamicReWOOStep = create_model(
            'DynamicReWOOStep',
            thought=(str, Field(description="The reasoning for why this specific tool call is necessary.")),
            tool_id=(ToolIDEnum, Field(description="The identifier of the tool to execute.")),
            params=(Dict[str, Any], Field(default_factory=dict, description="The parameters to pass to the tool.")),
        )
        DynamicReWOOPlan = create_model(
            'DynamicReWOOPlan',
            plan=(List[DynamicReWOOStep], Field(description="The sequence of tool calls to execute.")), # type: ignore
        )
        return DynamicReWOOPlan

    async def _generate_plan(self, goal: str, tool_definitions: str, candidate_tool_ids: List[str], correlation_id: Optional[str]) -> Tuple[Optional[ReWOOPlan], Optional[str]]:
        DynamicPlanModel = self._create_dynamic_plan_model(candidate_tool_ids)
        prompt_data = {"goal": goal, "tool_definitions": tool_definitions, "ReWOOPlan": DynamicPlanModel}
        planner_prompt_str = await self._genie.prompts.render_prompt(template_content=self._DEFAULT_PLANNER_PROMPT, data=prompt_data)
        if not planner_prompt_str:
            await self._genie.observability.trace_event("rewoo.plan.error", {"error": "Planner prompt string rendering failed."}, "ReWOOCommandProcessor", correlation_id)
            return None, None
        
        planner_messages: List[ChatMessage] = [{"role": "user", "content": planner_prompt_str}]
        raw_planner_output: Optional[str] = None
        await self._genie.observability.trace_event("rewoo.plan.prompt_ready", {"messages": planner_messages}, "ReWOOCommandProcessor", correlation_id)

        for attempt in range(self._max_plan_retries + 1):
            try:
                llm_response = await self._genie.llm.chat(messages=planner_messages, provider_id=self._planner_llm_id, output_schema=DynamicPlanModel)
                raw_planner_output = llm_response["message"]["content"] or ""
                await self._genie.observability.trace_event("rewoo.plan.raw_llm_output", {"attempt": attempt + 1, "output": raw_planner_output}, "ReWOOCommandProcessor", correlation_id)
                
                parsed_plan = await self._genie.llm.parse_output(llm_response, schema=DynamicPlanModel)
                if isinstance(parsed_plan, BaseModel):
                    reconstructed_plan = ReWOOPlan(**parsed_plan.model_dump())
                    await self._genie.observability.trace_event("rewoo.plan.success", {"plan_steps": len(reconstructed_plan.plan)}, "ReWOOCommandProcessor", correlation_id)
                    return reconstructed_plan, raw_planner_output
                else:
                    raise ValueError(f"Parsed output was not a Pydantic model. Type: {type(parsed_plan)}")
            except Exception as e:
                await self._genie.observability.trace_event("rewoo.plan.retry", {"attempt": attempt + 1, "error": str(e), "raw_output_if_any": raw_planner_output}, "ReWOOCommandProcessor", correlation_id)
                if attempt >= self._max_plan_retries:
                    await self._genie.observability.trace_event("rewoo.plan.error", {"error": f"Plan generation failed after retries: {e}"}, "ReWOOCommandProcessor", correlation_id)
                    return None, raw_planner_output
                await asyncio.sleep(1)
        return None, raw_planner_output

    async def _synthesize_answer(self, goal: str, plan: ReWOOPlan, evidence: List[ExecutionEvidence], correlation_id: Optional[str]) -> Tuple[str, str]:
        prompt_data = {"goal": goal, "plan": plan, "evidence": evidence}
        solver_prompt_str = await self._genie.prompts.render_prompt(template_content=self._DEFAULT_SOLVER_PROMPT, data=prompt_data)
        if not solver_prompt_str:
            await self._genie.observability.trace_event("rewoo.solve.error", {"error": "Solver prompt rendering failed."}, "ReWOOCommandProcessor", correlation_id)
            return "Error: Could not render the final answer prompt.", ""
        raw_solver_output: str = ""
        try:
            response = await self._genie.llm.chat(messages=[{"role": "user", "content": solver_prompt_str}], provider_id=self._solver_llm_id)
            raw_solver_output = response["message"]["content"] or ""
            await self._genie.observability.trace_event("rewoo.solve.raw_llm_output", {"output": raw_solver_output}, "ReWOOCommandProcessor", correlation_id)
            
            # --- FIX: Strip <think> tags from the final answer ---
            think_match = re.search(r"<think>(.*?)</think>", raw_solver_output, re.DOTALL | re.IGNORECASE)
            if think_match:
                solver_thought = think_match.group(1).strip()
                await self._genie.observability.trace_event("rewoo.solve.thought_extracted", {"thought": solver_thought}, "ReWOOCommandProcessor", correlation_id)
                final_answer = raw_solver_output[think_match.end():].strip()
            else:
                final_answer = raw_solver_output.strip()

            final_answer = final_answer or "The solver LLM returned an empty response."
            await self._genie.observability.trace_event("rewoo.solve.success", {"answer_length": len(final_answer)}, "ReWOOCommandProcessor", correlation_id)
            return final_answer, raw_solver_output
        except Exception as e:
            await self._genie.observability.trace_event("rewoo.solve.error", {"error": str(e), "raw_output_if_any": raw_solver_output}, "ReWOOCommandProcessor", correlation_id)
            return f"Error: The final answer could not be synthesized. Reason: {e}", raw_solver_output

    async def process_command(
        self, command: str, conversation_history: Optional[List[ChatMessage]] = None, correlation_id: Optional[str] = None
    ) -> CommandProcessorResponse:
        all_tools = await self._genie._tool_manager.list_tools(enabled_only=True)
        candidate_tool_ids = [t.identifier for t in all_tools]
        tool_definitions_list = [str(await self._genie._tool_manager.get_formatted_tool_definition(t_id, self._tool_formatter_id)) for t_id in candidate_tool_ids]
        tool_definitions_str = "\n\n".join(filter(None, tool_definitions_list))

        plan, raw_planner_output = await self._generate_plan(command, tool_definitions_str, candidate_tool_ids, correlation_id)
        
        if not plan:
            return {"error": "Failed to generate a valid execution plan.", "raw_response": {"planner_llm_output": raw_planner_output}}
        
        evidence: List[ExecutionEvidence] = []
        for step in plan.plan:
            await self._genie.observability.trace_event("rewoo.step.execute.start", {"tool_id": step.tool_id, "params": step.params}, "ReWOOCommandProcessor", correlation_id)
            try:
                result = await self._genie.execute_tool(step.tool_id, **step.params)
                evidence.append({"step": step.model_dump(), "result": result, "error": None})
                await self._genie.observability.trace_event("rewoo.step.execute.success", {"tool_id": step.tool_id}, "ReWOOCommandProcessor", correlation_id)
            except Exception as e:
                error_str = f"Error executing tool '{step.tool_id}': {e}"
                evidence.append({"step": step.model_dump(), "result": None, "error": error_str})
                await self._genie.observability.trace_event("rewoo.step.execute.error", {"tool_id": step.tool_id, "error": error_str}, "ReWOOCommandProcessor", correlation_id)
        
        final_answer, raw_solver_llm_output = await self._synthesize_answer(command, plan, evidence, correlation_id)
        
        raw_outputs_for_response = {}
        if raw_planner_output: raw_outputs_for_response["planner_llm_output"] = raw_planner_output
        if raw_solver_llm_output: raw_outputs_for_response["solver_llm_output"] = raw_solver_llm_output
        
        return {
            "final_answer": final_answer,
            "llm_thought_process": json.dumps({"plan": plan.model_dump(), "evidence": evidence}, indent=2, default=str),
            "raw_response": {"plan": plan.model_dump(), "evidence": evidence, **raw_outputs_for_response}
        }
