# src/genie_tooling/agents/plan_and_execute_agent.py
"""Plan-and-Execute Agent Implementation."""
import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .base_agent import BaseAgent

# Import Pydantic models from types.py
from .types import AgentOutput, PlanModelPydantic, PlannedStep

if TYPE_CHECKING:
    from genie_tooling.genie import Genie
    from genie_tooling.prompts.llm_output_parsers.types import ParsedOutput

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

    async def _generate_plan(self, goal: str, previous_steps_results: Optional[List[Dict[str,Any]]] = None) -> Optional[List[PlannedStep]]:
        logger.info(f"PlanAndExecuteAgent: Generating plan for goal: {goal}")
        await self.genie.observability.trace_event("plan_execute_agent.generate_plan.start", {"goal": goal}, "PlanAndExecuteAgent")

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
            logger.error(f"PlanAndExecuteAgent: Could not render planner prompt (ID: {self.planner_system_prompt_id}).")
            await self.genie.observability.trace_event("plan_execute_agent.generate_plan.error", {"error": "PlannerPromptRenderingFailed"}, "PlanAndExecuteAgent")
            return None

        for attempt in range(self.max_plan_retries + 1):
            try:
                llm_response = await self.genie.llm.chat(
                    messages=planner_prompt_messages, provider_id=self.planner_llm_provider_id
                )

                # Parse the LLM response into the PlanModelPydantic (imported from types.py)
                parsed_plan_model: Optional["ParsedOutput"] = await self.genie.llm.parse_output(
                    llm_response,
                    parser_id="pydantic_output_parser_v1",
                    schema=PlanModelPydantic # Use imported Pydantic model
                )

                if isinstance(parsed_plan_model, PlanModelPydantic): # Check against imported Pydantic model
                    plan_steps: List[PlannedStep] = []
                    for step_model in parsed_plan_model.plan:
                        plan_steps.append(PlannedStep(
                            step_number=step_model.step_number,
                            tool_id=step_model.tool_id,
                            params=step_model.params,
                            reasoning=step_model.reasoning
                        ))
                    logger.info(f"PlanAndExecuteAgent: Plan generated with {len(plan_steps)} steps. Overall reasoning: {parsed_plan_model.overall_reasoning}")
                    await self.genie.observability.trace_event("plan_execute_agent.generate_plan.success", {"plan_length": len(plan_steps), "reasoning": parsed_plan_model.overall_reasoning}, "PlanAndExecuteAgent")
                    return plan_steps
                else:
                    logger.warning(f"PlanAndExecuteAgent: LLM output for plan was not a valid PlanModelPydantic. Attempt {attempt+1}. Output: {llm_response['message']['content'][:200]}...")
                    if attempt < self.max_plan_retries:
                        await asyncio.sleep(1)
                        continue

            except Exception as e_plan:
                logger.error(f"PlanAndExecuteAgent: Error generating plan (attempt {attempt+1}): {e_plan}", exc_info=True)
                if attempt < self.max_plan_retries:
                    await asyncio.sleep(1)
                    continue

        await self.genie.observability.trace_event("plan_execute_agent.generate_plan.error", {"error": "PlanGenerationFailedAfterRetries"}, "PlanAndExecuteAgent")
        return None

    async def _execute_plan(self, plan: List[PlannedStep], goal: str) -> AgentOutput:
        logger.info(f"PlanAndExecuteAgent: Executing plan with {len(plan)} steps.")
        await self.genie.observability.trace_event("plan_execute_agent.execute_plan.start", {"plan_length": len(plan)}, "PlanAndExecuteAgent")

        step_results: List[Dict[str, Any]] = []
        current_plan = list(plan) # Make a mutable copy

        for i, step in enumerate(current_plan):
            logger.info(f"Executing step {i+1}/{len(current_plan)}: Tool '{step['tool_id']}' with params {step['params']}. Reasoning: {step.get('reasoning')}")
            await self.genie.observability.trace_event("plan_execute_agent.step.start", {"step_number": i+1, "tool_id": step["tool_id"], "params": step["params"]}, "PlanAndExecuteAgent")

            step_output: Any = None
            step_error: Optional[str] = None

            for attempt in range(self.max_step_retries + 1):
                try:
                    # HITL check for each step
                    approval_req = {
                        "request_id": f"plan_step_{i+1}_{self.genie._config._genie_instance._config.features.hitl_approver if hasattr(self.genie._config._genie_instance, '_config') else 'default_hitl'}", # type: ignore
                        "prompt": f"Approve execution of plan step {i+1}: Tool '{step['tool_id']}' with params {step['params']} for goal '{goal}'?",
                        "data_to_approve": {"tool_id": step["tool_id"], "params": step["params"], "step_reasoning": step.get("reasoning")}
                    }
                    approval_resp = await self.genie.human_in_loop.request_approval(approval_req) # type: ignore

                    if approval_resp["status"] != "approved":
                        step_error = f"Step {i+1} execution denied by HITL: {approval_resp.get('reason')}"
                        logger.warning(f"PlanAndExecuteAgent: {step_error}")
                        await self.genie.observability.trace_event("plan_execute_agent.step.hitl_denied", {"step_number": i+1, "reason": approval_resp.get("reason")}, "PlanAndExecuteAgent")
                        break # Break from retry loop for this step

                    step_output = await self.genie.execute_tool(step["tool_id"], **step["params"])
                    step_error = None # Clear error if successful

                    # Check if tool execution itself returned an error structure
                    if isinstance(step_output, dict) and step_output.get("error"):
                        step_error = f"Tool '{step['tool_id']}' execution reported error: {step_output['error']}"
                        logger.warning(f"PlanAndExecuteAgent: {step_error}")
                        # Treat tool-reported error as a step failure for retry/replan logic

                    break # Successful execution or tool-reported error (which is still a "completion" of the attempt)
                except Exception as e_exec:
                    step_error = f"Error executing tool '{step['tool_id']}' for step {i+1}: {str(e_exec)}"
                    logger.error(f"PlanAndExecuteAgent: {step_error}", exc_info=True)
                    if attempt < self.max_step_retries:
                        logger.info(f"Retrying step {i+1} (attempt {attempt+2}/{self.max_step_retries+1})")
                        await asyncio.sleep(1)
                    else:
                        logger.error(f"Step {i+1} failed after all retries.")

            step_results.append({
                "step_number": step["step_number"],
                "tool_id": step["tool_id"],
                "params": step["params"],
                "output": step_output if not step_error else None,
                "error": step_error
            })
            await self.genie.observability.trace_event("plan_execute_agent.step.end", {"step_number": i+1, "tool_id": step["tool_id"], "success": not step_error, "error": step_error}, "PlanAndExecuteAgent")

            if step_error:
                if self.replan_on_step_failure:
                    logger.info(f"Step {i+1} failed. Attempting to re-plan...")
                    await self.genie.observability.trace_event("plan_execute_agent.replan.start", {"failed_step": i+1, "error": step_error, "current_results": step_results}, "PlanAndExecuteAgent")
                    new_plan = await self._generate_plan(goal, previous_steps_results=step_results)
                    if new_plan:
                        logger.info("Re-planning successful. Executing new plan.")
                        return await self._execute_plan(new_plan, goal) # Recursive call with new plan
                    else:
                        logger.error("Re-planning failed. Aborting execution.")
                        await self.genie.observability.trace_event("plan_execute_agent.replan.error", {"error": "ReplanGenerationFailed"}, "PlanAndExecuteAgent")
                        return AgentOutput(status="error", output=f"Step {i+1} failed and re-planning also failed.", history=step_results, plan=current_plan)
                else:
                    logger.error(f"Step {i+1} failed. Plan execution aborted as re-planning is disabled.")
                    return AgentOutput(status="error", output=f"Execution failed at step {i+1}: {step_error}", history=step_results, plan=current_plan)

        final_result = step_results[-1].get("output") if step_results else "No steps executed or last step had no output."
        logger.info(f"PlanAndExecuteAgent: Plan execution completed. Final result (from last step): {str(final_result)[:200]}...")
        await self.genie.observability.trace_event("plan_execute_agent.execute_plan.success", {"final_result_type": type(final_result).__name__}, "PlanAndExecuteAgent")
        return AgentOutput(status="success", output=final_result, history=step_results, plan=current_plan)


    async def run(self, goal: str, **kwargs: Any) -> AgentOutput:
        logger.info(f"PlanAndExecuteAgent starting run for goal: {goal}")
        await self.genie.observability.trace_event("plan_execute_agent.run.start", {"goal": goal}, "PlanAndExecuteAgent")

        initial_plan = await self._generate_plan(goal)

        if not initial_plan:
            logger.error("PlanAndExecuteAgent: Failed to generate an initial plan.")
            await self.genie.observability.trace_event("plan_execute_agent.run.error", {"error": "InitialPlanGenerationFailed"}, "PlanAndExecuteAgent")
            return AgentOutput(status="error", output="Failed to generate a plan for the goal.", plan=None)

        execution_result = await self._execute_plan(initial_plan, goal)

        await self.genie.observability.trace_event(
            "plan_execute_agent.run.end",
            {"status": execution_result["status"], "output_type": type(execution_result["output"]).__name__}, # CORRECTED
            "PlanAndExecuteAgent"
        )
        return execution_result
