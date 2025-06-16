# src/genie_tooling/agents/plan_and_execute_agent.py
"""Plan-and-Execute Agent Implementation."""
import asyncio
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from genie_tooling.utils.placeholder_resolution import resolve_placeholders

from .base_agent import BaseAgent
from .types import (
    AgentOutput,
    PlanModelPydantic,
    PlannedStep,
)

if TYPE_CHECKING:
    from genie_tooling.genie import Genie
    from genie_tooling.prompts.llm_output_parsers.types import ParsedOutput

logger = logging.getLogger(__name__)

DEFAULT_PLANNER_SYSTEM_PROMPT_ID = "plan_and_execute_planner_prompt_v1"


class PlanAndExecuteAgent(BaseAgent):
    """
    Implements the Plan-and-Execute agentic loop.
    1. Planner: LLM generates a sequence of steps (tool calls) to achieve the goal.
                Steps can name their outputs for use in subsequent steps.
    2. Executor: Executes these steps sequentially, resolving placeholders.
    """

    def __init__(self, genie: "Genie", agent_config: Optional[Dict[str, Any]] = None):
        super().__init__(genie, agent_config)
        self.planner_system_prompt_id = self.agent_config.get("planner_system_prompt_id", DEFAULT_PLANNER_SYSTEM_PROMPT_ID)
        self.planner_llm_provider_id = self.agent_config.get("planner_llm_provider_id")
        self.tool_formatter_id = self.agent_config.get("tool_formatter_id", "compact_text_formatter_plugin_v1")
        self.max_plan_retries = self.agent_config.get("max_plan_retries", 1)
        self.max_step_retries = self.agent_config.get("max_step_retries", 0)
        self.replan_on_step_failure = self.agent_config.get("replan_on_step_failure", False)

        logger.info(
            f"PlanAndExecuteAgent initialized. Planner Prompt ID: {self.planner_system_prompt_id}, "
            f"Planner LLM: {self.planner_llm_provider_id or 'Genie Default'}"
        )

    async def _generate_plan(self, goal: str, correlation_id: str, scratchpad_summary: Optional[str] = None) -> Optional[List[PlannedStep]]:
        await self.genie.observability.trace_event("log.info", {"message": f"Generating plan for goal: {goal}"}, "PlanAndExecuteAgent", correlation_id)
        await self.genie.observability.trace_event("plan_execute_agent.generate_plan.start", {"goal": goal, "has_scratchpad_summary": scratchpad_summary is not None}, "PlanAndExecuteAgent", correlation_id)

        all_tools = await self.genie._tool_manager.list_tools() # type: ignore
        tool_definitions_list = []
        for tool_instance in all_tools:
            formatted_def = await self.genie._tool_manager.get_formatted_tool_definition( # type: ignore
                tool_instance.identifier, self.tool_formatter_id
            )
            if formatted_def:
                tool_definitions_list.append(str(formatted_def))
        tool_definitions_string = "\n\n".join(tool_definitions_list) if tool_definitions_list else "No tools available."

        prompt_data: Dict[str, Any] = {"goal": goal, "tool_definitions": tool_definitions_string}
        if scratchpad_summary: # For re-planning
            prompt_data["scratchpad_summary"] = scratchpad_summary

        planner_prompt_messages = await self.genie.prompts.render_chat_prompt(
            name=self.planner_system_prompt_id, data=prompt_data
        )
        if not planner_prompt_messages:
            await self.genie.observability.trace_event("log.error", {"message": f"Could not render planner prompt (ID: {self.planner_system_prompt_id})."}, "PlanAndExecuteAgent", correlation_id)
            await self.genie.observability.trace_event("plan_execute_agent.generate_plan.error", {"error": "PlannerPromptRenderingFailed"}, "PlanAndExecuteAgent", correlation_id)
            return None

        for attempt in range(self.max_plan_retries + 1):
            try:
                llm_response = await self.genie.llm.chat(
                    messages=planner_prompt_messages, provider_id=self.planner_llm_provider_id
                )
                parsed_plan_model: Optional["ParsedOutput"] = await self.genie.llm.parse_output(
                    llm_response,
                    parser_id="pydantic_output_parser_v1",
                    schema=PlanModelPydantic
                )

                if isinstance(parsed_plan_model, PlanModelPydantic):
                    plan_steps: List[PlannedStep] = []
                    for step_model in parsed_plan_model.plan:

                        params_dict = step_model.params if isinstance(step_model.params, dict) else {}
                        plan_steps.append(PlannedStep(
                            step_number=step_model.step_number,
                            tool_id=step_model.tool_id,
                            params=params_dict,
                            reasoning=step_model.reasoning,
                            output_variable_name=step_model.output_variable_name
                        ))
                    await self.genie.observability.trace_event("log.info", {"message": f"Plan generated with {len(plan_steps)} steps. Overall reasoning: {parsed_plan_model.overall_reasoning}"}, "PlanAndExecuteAgent", correlation_id)
                    await self.genie.observability.trace_event("plan_execute_agent.generate_plan.success", {"plan_length": len(plan_steps), "reasoning": parsed_plan_model.overall_reasoning}, "PlanAndExecuteAgent", correlation_id)
                    return plan_steps
                else:
                    await self.genie.observability.trace_event("log.warning", {"message": f"LLM output for plan was not a valid PlanModelPydantic. Attempt {attempt+1}. Output: {llm_response['message']['content'][:200]}..."}, "PlanAndExecuteAgent", correlation_id)
                    if attempt < self.max_plan_retries:
                        await asyncio.sleep(1)
                        continue
            except Exception as e_plan:
                await self.genie.observability.trace_event("log.error", {"message": f"Error generating plan (attempt {attempt+1}): {e_plan}", "exc_info": True}, "PlanAndExecuteAgent", correlation_id)
                if attempt < self.max_plan_retries:
                    await asyncio.sleep(1)
                    continue
        await self.genie.observability.trace_event("plan_execute_agent.generate_plan.error", {"error": "PlanGenerationFailedAfterRetries"}, "PlanAndExecuteAgent", correlation_id)
        return None

    async def _execute_plan(self, plan: List[PlannedStep], goal: str, correlation_id: str) -> AgentOutput:
        await self.genie.observability.trace_event("log.info", {"message": f"Executing plan with {len(plan)} steps."}, "PlanAndExecuteAgent", correlation_id)
        await self.genie.observability.trace_event("plan_execute_agent.execute_plan.start", {"plan_length": len(plan)}, "PlanAndExecuteAgent", correlation_id)

        step_results_history: List[Dict[str, Any]] = []
        scratchpad: Dict[str, Any] = {"outputs": {}}
        current_plan = list(plan)

        for i, step_typed_dict in enumerate(current_plan):
            step_reasoning = step_typed_dict.get("reasoning", "No reasoning provided.")
            step_output_var_name = step_typed_dict.get("output_variable_name")

            await self.genie.observability.trace_event("log.info", {"message": f"Executing step {i+1}/{len(current_plan)}: Tool '{step_typed_dict['tool_id']}' with params {step_typed_dict['params']}. Reasoning: {step_reasoning}"}, "PlanAndExecuteAgent", correlation_id)
            await self.genie.observability.trace_event("plan_execute_agent.step.start", {"step_number": i+1, "tool_id": step_typed_dict["tool_id"], "params": step_typed_dict["params"]}, "PlanAndExecuteAgent", correlation_id)

            step_output: Any = None
            step_error: Optional[str] = None
            resolved_params: Dict[str, Any] = {}

            try:
                params_with_placeholders = step_typed_dict.get("params", {})
                if isinstance(params_with_placeholders, dict):
                    resolved_params = resolve_placeholders(params_with_placeholders, scratchpad)
                else:
                    raise TypeError(f"Params for step {i+1} are not a dictionary.")
            except (ValueError, TypeError) as e_resolve:
                step_error = f"Parameter resolution failed for tool '{step_typed_dict['tool_id']}': {e_resolve}"
                await self.genie.observability.trace_event("log.warning", {"message": f"PlanAndExecuteAgent: {step_error}"}, "PlanAndExecuteAgent", correlation_id)
                await self.genie.observability.trace_event("plan_execute_agent.step.param_resolution.error", {"step_number": i+1, "tool_id": step_typed_dict["tool_id"], "error": str(e_resolve)}, "PlanAndExecuteAgent", correlation_id)

            if not step_error:
                for attempt in range(self.max_step_retries + 1):
                    try:
                        approval_req = {
                            "request_id": f"plan_step_{i+1}_{correlation_id}",
                            "prompt": f"Approve execution of plan step {i+1}: Tool '{step_typed_dict['tool_id']}' with params {resolved_params} for goal '{goal}'?",
                            "data_to_approve": {"tool_id": step_typed_dict["tool_id"], "params": resolved_params, "step_reasoning": step_reasoning}
                        }
                        approval_resp = await self.genie.human_in_loop.request_approval(approval_req) # type: ignore

                        if approval_resp["status"] != "approved":
                            step_error = f"Step {i+1} execution denied by HITL: {approval_resp.get('reason')}"
                            await self.genie.observability.trace_event("log.warning", {"message": f"PlanAndExecuteAgent: {step_error}"}, "PlanAndExecuteAgent", correlation_id)
                            await self.genie.observability.trace_event("plan_execute_agent.step.hitl_denied", {"step_number": i+1, "reason": approval_resp.get("reason")}, "PlanAndExecuteAgent", correlation_id)
                            break

                        step_output = await self.genie.execute_tool(step_typed_dict["tool_id"], **resolved_params)
                        step_error = None

                        if isinstance(step_output, dict) and step_output.get("error"):
                            step_error = f"Tool '{step_typed_dict['tool_id']}' execution reported error: {step_output['error']}"
                            await self.genie.observability.trace_event("log.warning", {"message": f"PlanAndExecuteAgent: {step_error}"}, "PlanAndExecuteAgent", correlation_id)

                        if not step_error and step_output_var_name:
                            scratchpad["outputs"][step_output_var_name] = step_output
                            await self.genie.observability.trace_event("log.debug", {"message": f"Stored output of step {i+1} as '{step_output_var_name}' in scratchpad."}, "PlanAndExecuteAgent", correlation_id)
                        break
                    except Exception as e_exec:
                        step_error = f"Error executing tool '{step_typed_dict['tool_id']}' for step {i+1}: {e_exec!s}"
                        await self.genie.observability.trace_event("log.error", {"message": f"PlanAndExecuteAgent: {step_error}", "exc_info": True}, "PlanAndExecuteAgent", correlation_id)
                        if attempt < self.max_step_retries:
                            await self.genie.observability.trace_event("log.info", {"message": f"Retrying step {i+1} (attempt {attempt+2}/{self.max_step_retries+1})"}, "PlanAndExecuteAgent", correlation_id)
                            await asyncio.sleep(1)
                        else:
                            await self.genie.observability.trace_event("log.error", {"message": f"Step {i+1} failed after all retries."}, "PlanAndExecuteAgent", correlation_id)

            step_results_history.append({
                "step_number": step_typed_dict["step_number"],
                "tool_id": step_typed_dict["tool_id"],
                "params_planned": step_typed_dict["params"],
                "params_resolved": resolved_params if not step_error else step_typed_dict["params"],
                "output": step_output if not step_error else None,
                "error": step_error,
                "output_variable_name": step_output_var_name
            })
            await self.genie.observability.trace_event("plan_execute_agent.step.end", {"step_number": i+1, "tool_id": step_typed_dict["tool_id"], "success": not step_error, "error": step_error}, "PlanAndExecuteAgent", correlation_id)

            if step_error:
                if self.replan_on_step_failure:
                    await self.genie.observability.trace_event("log.info", {"message": f"Step {i+1} failed. Attempting to re-plan..."}, "PlanAndExecuteAgent", correlation_id)
                    await self.genie.observability.trace_event("plan_execute_agent.replan.start", {"failed_step": i+1, "error": step_error, "current_results_history": step_results_history}, "PlanAndExecuteAgent", correlation_id)
                    scratchpad_summary_for_replan = json.dumps(scratchpad, default=str, indent=2)
                    new_plan = await self._generate_plan(goal, correlation_id, scratchpad_summary=scratchpad_summary_for_replan)
                    if new_plan:
                        await self.genie.observability.trace_event("log.info", {"message": "Re-planning successful. Executing new plan."}, "PlanAndExecuteAgent", correlation_id)
                        return await self._execute_plan(new_plan, goal, correlation_id)
                    else:
                        await self.genie.observability.trace_event("log.error", {"message": "Re-planning failed. Aborting execution."}, "PlanAndExecuteAgent", correlation_id)
                        await self.genie.observability.trace_event("plan_execute_agent.replan.error", {"error": "ReplanGenerationFailed"}, "PlanAndExecuteAgent", correlation_id)
                        return AgentOutput(status="error", output=f"Step {i+1} failed and re-planning also failed.", history=step_results_history, plan=current_plan)
                else:
                    await self.genie.observability.trace_event("log.error", {"message": f"Step {i+1} failed. Plan execution aborted as re-planning is disabled."}, "PlanAndExecuteAgent", correlation_id)
                    return AgentOutput(status="error", output=f"Execution failed at step {i+1}: {step_error}", history=step_results_history, plan=current_plan)

        final_result = step_results_history[-1].get("output") if step_results_history else "No steps executed or last step had no output."
        await self.genie.observability.trace_event("log.info", {"message": f"Plan execution completed. Final result (from last step): {str(final_result)[:200]}..."}, "PlanAndExecuteAgent", correlation_id)
        await self.genie.observability.trace_event("plan_execute_agent.execute_plan.success", {"final_result_type": type(final_result).__name__}, "PlanAndExecuteAgent", correlation_id)
        return AgentOutput(status="success", output=final_result, history=step_results_history, plan=current_plan)


    async def run(self, goal: str, **kwargs: Any) -> AgentOutput:
        correlation_id = str(uuid.uuid4())
        await self.genie.observability.trace_event("log.info", {"message": f"PlanAndExecuteAgent starting run for goal: {goal}"}, "PlanAndExecuteAgent", correlation_id)
        await self.genie.observability.trace_event("plan_execute_agent.run.start", {"goal": goal}, "PlanAndExecuteAgent", correlation_id)

        initial_plan = await self._generate_plan(goal, correlation_id)

        if not initial_plan:
            await self.genie.observability.trace_event("log.error", {"message": "Failed to generate an initial plan."}, "PlanAndExecuteAgent", correlation_id)
            await self.genie.observability.trace_event("plan_execute_agent.run.error", {"error": "InitialPlanGenerationFailed"}, "PlanAndExecuteAgent", correlation_id)
            return AgentOutput(status="error", output="Failed to generate a plan for the goal.", plan=None, history=None)

        execution_result = await self._execute_plan(initial_plan, goal, correlation_id)

        await self.genie.observability.trace_event(
            "plan_execute_agent.run.end",
            {"status": execution_result["status"], "output_type": type(execution_result["output"]).__name__},
            "PlanAndExecuteAgent",
            correlation_id
        )
        return execution_result
