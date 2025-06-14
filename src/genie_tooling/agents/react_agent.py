"""ReAct (Reason-Act) Agent Implementation."""
import asyncio
import json
import logging
import re
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .base_agent import BaseAgent
from .types import (
    AgentOutput,
    ReActObservation,
)

if TYPE_CHECKING:
    from genie_tooling.genie import Genie
    from genie_tooling.llm_providers.types import ChatMessage

logger = logging.getLogger(__name__)

DEFAULT_REACT_MAX_ITERATIONS = 7
DEFAULT_REACT_SYSTEM_PROMPT_ID = "react_agent_system_prompt_v1"

class ReActAgent(BaseAgent):
    """
    Implements the ReAct (Reason-Act) agentic loop.
    """

    def __init__(self, genie: "Genie", agent_config: Optional[Dict[str, Any]] = None):
        super().__init__(genie, agent_config)
        self.max_iterations = self.agent_config.get("max_iterations", DEFAULT_REACT_MAX_ITERATIONS)
        self.system_prompt_id = self.agent_config.get("system_prompt_id", DEFAULT_REACT_SYSTEM_PROMPT_ID)
        self.llm_provider_id = self.agent_config.get("llm_provider_id")
        self.tool_formatter_id = self.agent_config.get("tool_formatter_id", "compact_text_formatter_plugin_v1")
        self.stop_sequences = self.agent_config.get("stop_sequences", ["Observation:"])
        self.llm_retry_attempts = self.agent_config.get("llm_retry_attempts", 1)
        self.llm_retry_delay = self.agent_config.get("llm_retry_delay_seconds", 2.0)

        logger.info(
            f"ReActAgent initialized. Max iterations: {self.max_iterations}, "
            f"System Prompt ID: {self.system_prompt_id}, LLM Provider: {self.llm_provider_id or 'Genie Default'}"
        )

    def _parse_llm_reason_act_output(self, llm_output: str, correlation_id: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        thought_match = re.search(r"Thought:\s*(.*?)(?:\nAction:|\nAnswer:|$)", llm_output, re.DOTALL | re.IGNORECASE)
        action_match = re.search(r"Action:\s*(\w+)\[(.*?)\]", llm_output, re.DOTALL | re.IGNORECASE)
        answer_match = re.search(r"Answer:\s*(.*)", llm_output, re.DOTALL | re.IGNORECASE)
        thought = thought_match.group(1).strip() if thought_match else None
        action_tool_name: Optional[str] = None
        action_params_json: Optional[str] = None
        if action_match:
            action_tool_name = action_match.group(1).strip()
            action_params_json = action_match.group(2).strip()
            if not (action_params_json.startswith("{") and action_params_json.endswith("}")):
                if not action_params_json:
                    action_params_json = "{}"
                else:
                    asyncio.create_task(self.genie.observability.trace_event("log.warning", {"message": f"Action params '{action_params_json}' not a valid JSON object string. Attempting to wrap."}, "ReActAgent", correlation_id))
                    if not action_params_json.startswith("{"): action_params_json = "{" + action_params_json
                    if not action_params_json.endswith("}"): action_params_json = action_params_json + "}"
        final_answer = answer_match.group(1).strip() if answer_match else None
        if not thought and not (action_tool_name or final_answer):
            asyncio.create_task(self.genie.observability.trace_event("log.warning", {"message": f"Could not parse Thought, Action, or Answer from LLM output: {llm_output[:200]}..."}, "ReActAgent", correlation_id))
            thought = llm_output.strip()
        return thought, (f"{action_tool_name}[{action_params_json}]" if action_tool_name else None), final_answer

    async def run(self, goal: str, **kwargs: Any) -> AgentOutput:
        correlation_id = str(uuid.uuid4())
        await self.genie.observability.trace_event("log.info", {"message": f"ReActAgent starting run for goal: {goal}"}, "ReActAgent", correlation_id)
        await self.genie.observability.trace_event("react_agent.run.start", {"goal": goal}, "ReActAgent", correlation_id)
        scratchpad: List[ReActObservation] = []
        all_tools = await self.genie._tool_manager.list_tools() # type: ignore
        tool_definitions_list = []
        candidate_tool_ids = [t.identifier for t in all_tools]
        for tool_instance in all_tools:
            formatted_def = await self.genie._tool_manager.get_formatted_tool_definition(tool_instance.identifier, self.tool_formatter_id) # type: ignore
            if formatted_def: tool_definitions_list.append(str(formatted_def))
        tool_definitions_string = "\n\n".join(tool_definitions_list) if tool_definitions_list else "No tools available."

        for i in range(self.max_iterations):
            await self.genie.observability.trace_event("log.info", {"message": f"ReAct Iteration {i+1}/{self.max_iterations}"}, "ReActAgent", correlation_id)
            await self.genie.observability.trace_event("react_agent.iteration.start", {"iteration": i+1, "goal": goal, "scratchpad_size": len(scratchpad)}, "ReActAgent", correlation_id)
            prompt_data: Dict[str, Any] = {"goal": goal, "scratchpad": "\n".join([f"Thought: {s['thought']}\nAction: {s['action']}\nObservation: {s['observation']}" for s in scratchpad]), "tool_definitions": tool_definitions_string}
            rendered_prompt_str = await self.genie.prompts.render_prompt(name=self.system_prompt_id, data=prompt_data)
            if not rendered_prompt_str:
                await self.genie.observability.trace_event("log.error", {"message": f"Could not render ReAct prompt content (ID: {self.system_prompt_id}). Terminating."}, "ReActAgent", correlation_id)
                await self.genie.observability.trace_event("react_agent.run.error", {"error": "PromptRenderingFailed"}, "ReActAgent", correlation_id)
                return AgentOutput(status="error", output="Failed to render ReAct prompt.", history=scratchpad)
            reasoning_prompt_messages: List["ChatMessage"] = [{"role": "user", "content": rendered_prompt_str}]
            llm_response_text: Optional[str] = None; llm_error_str: Optional[str] = None
            for attempt in range(self.llm_retry_attempts + 1):
                try:
                    llm_chat_response = await self.genie.llm.chat(messages=reasoning_prompt_messages, provider_id=self.llm_provider_id, stop=self.stop_sequences)
                    llm_response_text = llm_chat_response["message"]["content"]; llm_error_str = None; break
                except Exception as e_llm:
                    llm_error_str = str(e_llm)
                    await self.genie.observability.trace_event("log.warning", {"message": f"LLM call failed (attempt {attempt+1}/{self.llm_retry_attempts+1}): {llm_error_str}"}, "ReActAgent", correlation_id)
                    if attempt < self.llm_retry_attempts: await asyncio.sleep(self.llm_retry_delay * (attempt + 1))
                    else:
                        await self.genie.observability.trace_event("log.error", {"message": f"LLM call failed after all retries. Error: {llm_error_str}"}, "ReActAgent", correlation_id)
                        await self.genie.observability.trace_event("react_agent.llm.error", {"error": llm_error_str, "iteration": i+1}, "ReActAgent", correlation_id)
                        return AgentOutput(status="error", output=f"LLM failed after retries: {llm_error_str}", history=scratchpad)
            if not llm_response_text: return AgentOutput(status="error", output="LLM returned no response content.", history=scratchpad)
            thought, action_str, final_answer = self._parse_llm_reason_act_output(llm_response_text, correlation_id)
            await self.genie.observability.trace_event("log.debug", {"message": f"ReAct LLM Output Parsed: Thought='{thought}', Action='{action_str}', Answer='{final_answer}'"}, "ReActAgent", correlation_id)
            await self.genie.observability.trace_event("react_agent.llm.parsed", {"thought": thought, "action_str": action_str, "final_answer": final_answer, "iteration": i+1}, "ReActAgent", correlation_id)
            if final_answer:
                await self.genie.observability.trace_event("log.info", {"message": f"ReActAgent: Final answer received: {final_answer}"}, "ReActAgent", correlation_id)
                scratchpad.append(ReActObservation(thought=thought or "N/A", action="Answer", observation=final_answer))
                await self.genie.observability.trace_event("react_agent.run.success", {"answer": final_answer, "iterations": i+1}, "ReActAgent", correlation_id)
                return AgentOutput(status="success", output=final_answer, history=scratchpad)
            if not action_str:
                await self.genie.observability.trace_event("log.warning", {"message": "LLM did not specify a valid action or final answer. Terminating."}, "ReActAgent", correlation_id)
                scratchpad.append(ReActObservation(thought=thought or "No thought.", action="Error", observation="LLM did not provide a valid action or answer."))
                await self.genie.observability.trace_event("react_agent.run.error", {"error": "NoActionOrAnswerFromLLM", "iteration": i+1}, "ReActAgent", correlation_id)
                return AgentOutput(status="error", output="LLM did not provide a valid action or answer.", history=scratchpad)
            action_tool_match = re.match(r"(\w+)\[(.*)\]", action_str)
            if not action_tool_match:
                await self.genie.observability.trace_event("log.warning", {"message": f"Could not parse tool name and params from action string: {action_str}"}, "ReActAgent", correlation_id)
                observation_content = f"Error: Invalid action format '{action_str}'. Expected ToolName[JSON_params]."
                scratchpad.append(ReActObservation(thought=thought or "N/A", action=action_str, observation=observation_content)); continue
            tool_name_from_llm = action_tool_match.group(1).strip(); tool_params_json_str = action_tool_match.group(2).strip()
            try:
                tool_params = json.loads(tool_params_json_str or "{}")
                if not isinstance(tool_params, dict): raise json.JSONDecodeError("Params not a dict", tool_params_json_str,0)
            except json.JSONDecodeError as e_json:
                await self.genie.observability.trace_event("log.warning", {"message": f"Failed to parse JSON params for tool '{tool_name_from_llm}': {e_json}. Params string: '{tool_params_json_str}'"}, "ReActAgent", correlation_id)
                observation_content = f"Error: Invalid JSON parameters for tool '{tool_name_from_llm}': {e_json}. Parameters received: '{tool_params_json_str}'"
                scratchpad.append(ReActObservation(thought=thought or "N/A", action=action_str, observation=observation_content)); continue
            if tool_name_from_llm not in candidate_tool_ids:
                await self.genie.observability.trace_event("log.warning", {"message": f"LLM chose tool '{tool_name_from_llm}' which is not in the candidate list. Informing LLM."}, "ReActAgent", correlation_id)
                observation_content = f"Error: Tool '{tool_name_from_llm}' is not available or not in the provided list. Available tools: {', '.join(candidate_tool_ids[:3])}..."
                scratchpad.append(ReActObservation(thought=thought or "N/A", action=action_str, observation=observation_content)); continue
            await self.genie.observability.trace_event("log.info", {"message": f"Executing tool '{tool_name_from_llm}' with params: {tool_params}"}, "ReActAgent", correlation_id)
            await self.genie.observability.trace_event("react_agent.tool.execute.start", {"tool_id": tool_name_from_llm, "params": tool_params, "iteration": i+1}, "ReActAgent", correlation_id)
            try:
                tool_execution_result = await self.genie.execute_tool(tool_name_from_llm, **tool_params)
                observation_content = json.dumps(tool_execution_result) if isinstance(tool_execution_result, dict) else str(tool_execution_result)
                await self.genie.observability.trace_event("react_agent.tool.execute.success", {"tool_id": tool_name_from_llm, "result_type": type(tool_execution_result).__name__, "iteration": i+1}, "ReActAgent", correlation_id)
            except Exception as e_tool_exec:
                await self.genie.observability.trace_event("log.error", {"message": f"Error executing tool '{tool_name_from_llm}': {e_tool_exec}", "exc_info": True}, "ReActAgent", correlation_id)
                observation_content = f"Error executing tool '{tool_name_from_llm}': {e_tool_exec!s}"
                await self.genie.observability.trace_event("react_agent.tool.execute.error", {"tool_id": tool_name_from_llm, "error": str(e_tool_exec), "iteration": i+1}, "ReActAgent", correlation_id)
            scratchpad.append(ReActObservation(thought=thought or "N/A", action=action_str, observation=observation_content[:1000]))
        await self.genie.observability.trace_event("log.warning", {"message": f"Exceeded max iterations ({self.max_iterations}) for goal: {goal}"}, "ReActAgent", correlation_id)
        await self.genie.observability.trace_event("react_agent.run.max_iterations_reached", {"goal": goal}, "ReActAgent", correlation_id)
        return AgentOutput(status="max_iterations_reached", output=f"Max iterations ({self.max_iterations}) reached.", history=scratchpad)
