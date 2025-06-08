# src/genie_tooling/agents/plan_and_execute_agent.py
"""Plan-and-Execute Agent Implementation."""
import asyncio
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .base_agent import BaseAgent

# Import Pydantic models from types.py
from .types import AgentOutput, PlanModelPydantic, PlannedStep

if TYPE_CHECKING:
    from genie_tooling.genie import Genie
    from genie_tooling.prompts.llm_output_parsers.types import ParsedOutput

# Use standard logger for module-level info, agent operational logs go through tracing
logger = logging.getLogger(__name__)

DEFAULT_PLANNER_SYSTEM_PROMPT_ID = "plan_and_execute_planner_prompt_v1"
DEFAULT_EXECUTOR_RECONCILE_PROMPT_ID = "plan_and_execute_reconcile_prompt_v1" # If needed

class PlanAndExecuteAgent(BaseAgent):
    """
    Implements the Plan-and-Execute agentic loop.
    1. Planner: LLM generates a sequence of steps (tool calls) to achieve the goal.
    2. Executor: Executes these steps sequentially.
    """

    def __init__(self, genie: "Genie", agent_config: Optional[Dict[str, Any]] = None):
        super().__init__(genie, agent_config)
        self.planner_system_prompt_id = self.agent_config.get("planner_system_prompt_id", DEFAULT_PLANNER_SYSTEM_PROMPT_ID)
        self.planner_llm_provider_id = self.agent_config.get("planner_llm_provider_id")
        self.tool_formatter_id = self.agent_config.get("tool_formatter_id", "compact_text_formatter_plugin_v1")
        self.max_plan_retries = self.agent_config.get("max_plan_retries", 1)
        self.max_step_retries = self.agent_config.get("max_step_retries", 0) # Default: no retry on step failure
        self.replan_on_step_failure = self.agent_config.get("replan_on_step_failure", False)

        logger.info(
            f"PlanAndExecuteAgent initialized. Planner Prompt ID: {self.planner_system_prompt_id}, "
            f"Planner LLM: {self.planner_llm_provider_id or 'Genie Default'}"
        )

    async def _generate_plan(self, goal: str, correlation_id: str, previous_steps_results: Optional[List[Dict[str,Any]]] = None) -> Optional[List[PlannedStep]]:
        await self.genie.observability.trace_event("log.info", {"message": f"Generating plan for goal: {goal}"}, "PlanAndExecuteAgent", correlation_id)
        await self.genie.observability.trace_event("plan_execute_agent.generate_plan.start", {"goal": goal}, "PlanAndExecuteAgent", correlation_id)

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
        if previous_steps_results:
            prompt_data["previous_steps_results_json"] = json.dumps(previous_steps_results)

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
                        plan_steps.append(PlannedStep(
                            step_number=step_model.step_number,
                            tool_id=step_model.tool_id,
                            params=step_model.params,
                            reasoning=step_model.reasoning
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

        step_results: List[Dict[str, Any]] = []
        current_plan = list(plan)

        for i, step in enumerate(current_plan):
            await self.genie.observability.trace_event("log.info", {"message": f"Executing step {i+1}/{len(current_plan)}: Tool '{step['tool_id']}' with params {step['params']}. Reasoning: {step.get('reasoning')}"}, "PlanAndExecuteAgent", correlation_id)
            await self.genie.observability.trace_event("plan_execute_agent.step.start", {"step_number": i+1, "tool_id": step["tool_id"], "params": step["params"]}, "PlanAndExecuteAgent", correlation_id)

            step_output: Any = None
            step_error: Optional[str] = None

            for attempt in range(self.max_step_retries + 1):
                try:
                    approval_req = {
                        "request_id": f"plan_step_{i+1}_{correlation_id}",
                        "prompt": f"Approve execution of plan step {i+1}: Tool '{step['tool_id']}' with params {step['params']} for goal '{goal}'?",
                        "data_to_approve": {"tool_id": step["tool_id"], "params": step["params"], "step_reasoning": step.get("reasoning")}
                    }
                    approval_resp = await self.genie.human_in_loop.request_approval(approval_req) # type: ignore

                    if approval_resp["status"] != "approved":
                        step_error = f"Step {i+1} execution denied by HITL: {approval_resp.get('reason')}"
                        await self.genie.observability.trace_event("log.warning", {"message": f"PlanAndExecuteAgent: {step_error}"}, "PlanAndExecuteAgent", correlation_id)
                        await self.genie.observability.trace_event("plan_execute_agent.step.hitl_denied", {"step_number": i+1, "reason": approval_resp.get('reason')}, "PlanAndExecuteAgent", correlation_id)
                        break

                    step_output = await self.genie.execute_tool(step["tool_id"], **step["params"])
                    step_error = None

                    if isinstance(step_output, dict) and step_output.get("error"):
                        step_error = f"Tool '{step['tool_id']}' execution reported error: {step_output['error']}"
                        await self.genie.observability.trace_event("log.warning", {"message": f"PlanAndExecuteAgent: {step_error}"}, "PlanAndExecuteAgent", correlation_id)

                    break
                except Exception as e_exec:
                    step_error = f"Error executing tool '{step['tool_id']}' for step {i+1}: {str(e_exec)}"
                    await self.genie.observability.trace_event("log.error", {"message": f"PlanAndExecuteAgent: {step_error}", "exc_info": True}, "PlanAndExecuteAgent", correlation_id)
                    if attempt < self.max_step_retries:
                        await self.genie.observability.trace_event("log.info", {"message": f"Retrying step {i+1} (attempt {attempt+2}/{self.max_step_retries+1})"}, "PlanAndExecuteAgent", correlation_id)
                        await asyncio.sleep(1)
                    else:
                        await self.genie.observability.trace_event("log.error", {"message": f"Step {i+1} failed after all retries."}, "PlanAndExecuteAgent", correlation_id)

            step_results.append({
                "step_number": step["step_number"],
                "tool_id": step["tool_id"],
                "params": step["params"],
                "output": step_output if not step_error else None,
                "error": step_error
            })
            await self.genie.observability.trace_event("plan_execute_agent.step.end", {"step_number": i+1, "tool_id": step["tool_id"], "success": not step_error, "error": step_error}, "PlanAndExecuteAgent", correlation_id)

            if step_error:
                if self.replan_on_step_failure:
                    await self.genie.observability.trace_event("log.info", {"message": f"Step {i+1} failed. Attempting to re-plan..."}, "PlanAndExecuteAgent", correlation_id)
                    await self.genie.observability.trace_event("plan_execute_agent.replan.start", {"failed_step": i+1, "error": step_error, "current_results": step_results}, "PlanAndExecuteAgent", correlation_id)
                    new_plan = await self._generate_plan(goal, correlation_id, previous_steps_results=step_results)
                    if new_plan:
                        await self.genie.observability.trace_event("log.info", {"message": "Re-planning successful. Executing new plan."}, "PlanAndExecuteAgent", correlation_id)
                        return await self._execute_plan(new_plan, goal, correlation_id)
                    else:
                        await self.genie.observability.trace_event("log.error", {"message": "Re-planning failed. Aborting execution."}, "PlanAndExecuteAgent", correlation_id)
                        await self.genie.observability.trace_event("plan_execute_agent.replan.error", {"error": "ReplanGenerationFailed"}, "PlanAndExecuteAgent", correlation_id)
                        return AgentOutput(status="error", output=f"Step {i+1} failed and re-planning also failed.", history=step_results, plan=current_plan)
                else:
                    await self.genie.observability.trace_event("log.error", {"message": f"Step {i+1} failed. Plan execution aborted as re-planning is disabled."}, "PlanAndExecuteAgent", correlation_id)
                    return AgentOutput(status="error", output=f"Execution failed at step {i+1}: {step_error}", history=step_results, plan=current_plan)

        final_result = step_results[-1].get("output") if step_results else "No steps executed or last step had no output."
        await self.genie.observability.trace_event("log.info", {"message": f"Plan execution completed. Final result (from last step): {str(final_result)[:200]}..."}, "PlanAndExecuteAgent", correlation_id)
        await self.genie.observability.trace_event("plan_execute_agent.execute_plan.success", {"final_result_type": type(final_result).__name__}, "PlanAndExecuteAgent", correlation_id)
        return AgentOutput(status="success", output=final_result, history=step_results, plan=current_plan)


    async def run(self, goal: str, **kwargs: Any) -> AgentOutput:
        correlation_id = str(uuid.uuid4())
        await self.genie.observability.trace_event("log.info", {"message": f"PlanAndExecuteAgent starting run for goal: {goal}"}, "PlanAndExecuteAgent", correlation_id)
        await self.genie.observability.trace_event("plan_execute_agent.run.start", {"goal": goal}, "PlanAndExecuteAgent", correlation_id)

        initial_plan = await self._generate_plan(goal, correlation_id)

        if not initial_plan:
            await self.genie.observability.trace_event("log.error", {"message": "Failed to generate an initial plan."}, "PlanAndExecuteAgent", correlation_id)
            await self.genie.observability.trace_event("plan_execute_agent.run.error", {"error": "InitialPlanGenerationFailed"}, "PlanAndExecuteAgent", correlation_id)
            return AgentOutput(status="error", output="Failed to generate a plan for the goal.", plan=None)

        execution_result = await self._execute_plan(initial_plan, goal, correlation_id)

        await self.genie.observability.trace_event(
            "plan_execute_agent.run.end",
            {"status": execution_result["status"], "output_type": type(execution_result["output"]).__name__},
            "PlanAndExecuteAgent",
            correlation_id
        )
        return execution_result
