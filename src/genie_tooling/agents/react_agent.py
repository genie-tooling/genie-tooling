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
background_tasks = set()


# Built-in default system prompt used when no user-supplied template is found
# in `genie.prompts` under the configured ID. Mirrors ReWOO's approach
# (_DEFAULT_PLANNER_PROMPT in rewoo_processor.py) so the bundled agent works
# out of the box with no extra setup. Rendered via the configured template
# engine (default jinja2_chat_template_v1 — Jinja2 syntax).
#
# We wrap the body in `{% autoescape false %}` because the active Jinja2
# engine ships with HTML autoescaping on — without that wrap, quotes and
# angle-brackets in tool_definitions/scratchpad become HTML entities
# (`"` -> `&#34;`), which the LLM cannot parse as JSON.
DEFAULT_REACT_SYSTEM_PROMPT = """{% autoescape false %}You are a problem-solving agent that reasons step by step using the ReAct pattern.

You have access to the following tools:
{{ tool_definitions }}

Your task: {{ goal }}

You must respond using EXACTLY this format on each turn:

Thought: <your reasoning about what to do next>
Action: <ToolName>[<valid JSON parameters>]

OR, when you have the final answer:

Thought: <your final reasoning>
Answer: <the final answer to the user's task>

Rules:
- The Action's ToolName must EXACTLY match one of the tools listed above.
- The Action's parameters must be a single valid JSON object (use {} for no params).
- After each Action you will receive an "Observation:" with the tool result; reason about it on the next Thought.
- When you have enough information to answer, emit "Answer:" instead of another Action.
- Be concise. Do not invent tools that aren't listed.

History so far:
{{ scratchpad }}

What is your next Thought and Action (or Answer)?{% endautoescape %}
"""

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
        # M7: when True AND the active provider supports native tool-use
        # (Anthropic, OpenAI, Gemini, modern Ollama), the agent skips the
        # regex Thought/Action parsing and lets the provider's structured
        # tool_use round-trip drive the loop. The default stays False for
        # backward compatibility with local models that don't have native
        # tool-use (older llama.cpp, plain text completions). When the
        # provider doesn't support it, the chat call simply ignores the
        # `tools=` kwarg and behaves like text-only — but you get nothing
        # useful out of the loop, so set this False for those.
        self.use_native_tool_use = bool(self.agent_config.get("use_native_tool_use", False))
        # When True, every tool call inside the ReAct loop is gated behind a
        # HITL approval request. Denied calls become observations in the
        # scratchpad — the agent sees the denial and can choose a different
        # action or end with an answer. Default off; corporate deployments
        # that need to gate destructive tools (database writes, deploys,
        # outbound communications) set this True and configure a real
        # approver (webhook, policy, CLI) via MiddlewareConfig.
        self.hitl_per_action = bool(self.agent_config.get("hitl_per_action", False))

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
                    task = asyncio.create_task(self.genie.observability.trace_event("log.warning", {"message": f"Action params '{action_params_json}' not a valid JSON object string. Attempting to wrap."}, "ReActAgent", correlation_id))
                    background_tasks.add(task)
                    task.add_done_callback(background_tasks.discard)
                    if not action_params_json.startswith("{"):
                        action_params_json = "{" + action_params_json
                    if not action_params_json.endswith("}"):
                        action_params_json = action_params_json + "}"
        final_answer = answer_match.group(1).strip() if answer_match else None
        if not thought and not (action_tool_name or final_answer):
            task = asyncio.create_task(self.genie.observability.trace_event("log.warning", {"message": f"Could not parse Thought, Action, or Answer from LLM output: {llm_output[:200]}..."}, "ReActAgent", correlation_id))
            background_tasks.add(task)
            task.add_done_callback(background_tasks.discard)
            thought = llm_output.strip()
        return thought, (f"{action_tool_name}[{action_params_json}]" if action_tool_name else None), final_answer

    async def run(self, goal: str, **kwargs: Any) -> AgentOutput:
        if self.use_native_tool_use:
            return await self._run_native_tool_use(goal, **kwargs)
        return await self._run_regex_loop(goal, **kwargs)

    async def _run_regex_loop(self, goal: str, **kwargs: Any) -> AgentOutput:
        """Original ReAct loop that parses free-text `Thought:`/`Action:`
        tokens from the LLM with regex. Kept as the default path so models
        without native tool-use (older Ollama, llama.cpp) still work."""
        correlation_id = str(uuid.uuid4())
        await self.genie.observability.trace_event("log.info", {"message": f"ReActAgent starting run for goal: {goal}"}, "ReActAgent", correlation_id)
        await self.genie.observability.trace_event("react_agent.run.start", {"goal": goal}, "ReActAgent", correlation_id)
        scratchpad: List[ReActObservation] = []
        all_tools = await self.genie.tools.list() # type: ignore
        tool_definitions_list = []
        candidate_tool_ids = [t.identifier for t in all_tools]
        for tool_instance in all_tools:
            formatted_def = await self.genie.tools.get_definition(tool_instance.identifier, self.tool_formatter_id) # type: ignore
            if formatted_def:
                tool_definitions_list.append(str(formatted_def))
        tool_definitions_string = "\n\n".join(tool_definitions_list) if tool_definitions_list else "No tools available."

        for i in range(self.max_iterations):
            await self.genie.observability.trace_event("log.info", {"message": f"ReAct Iteration {i+1}/{self.max_iterations}"}, "ReActAgent", correlation_id)
            await self.genie.observability.trace_event("react_agent.iteration.start", {"iteration": i+1, "goal": goal, "scratchpad_size": len(scratchpad)}, "ReActAgent", correlation_id)
            prompt_data: Dict[str, Any] = {"goal": goal, "scratchpad": "\n".join([f"Thought: {s['thought']}\nAction: {s['action']}\nObservation: {s['observation']}" for s in scratchpad]), "tool_definitions": tool_definitions_string}
            rendered_prompt_str = await self.genie.prompts.render_prompt(name=self.system_prompt_id, data=prompt_data)
            if not rendered_prompt_str:
                # Fall back to the built-in default template if the registry
                # doesn't have one under this ID. Must pass the Jinja2 engine
                # explicitly — the PromptManager defaults to None, which makes
                # render_prompt use Python str.format() instead of Jinja2 and
                # leaves `{{ var }}` as literal `{ var }` in the output. The
                # ReWOO processor takes the same precaution.
                await self.genie.observability.trace_event("log.info", {"message": f"No registry template for '{self.system_prompt_id}'; rendering built-in default ReAct prompt."}, "ReActAgent", correlation_id)
                rendered_prompt_str = await self.genie.prompts.render_prompt(
                    template_content=DEFAULT_REACT_SYSTEM_PROMPT,
                    data=prompt_data,
                    template_engine_id="jinja2_chat_template_v1",
                )
            if not rendered_prompt_str:
                await self.genie.observability.trace_event("log.error", {"message": f"Could not render ReAct prompt content (ID: {self.system_prompt_id}) even with the built-in default. Terminating."}, "ReActAgent", correlation_id)
                await self.genie.observability.trace_event("react_agent.run.error", {"error": "PromptRenderingFailed"}, "ReActAgent", correlation_id)
                return AgentOutput(status="error", output="Failed to render ReAct prompt.", history=scratchpad)
            reasoning_prompt_messages: List["ChatMessage"] = [{"role": "user", "content": rendered_prompt_str}]
            llm_response_text: Optional[str] = None
            llm_error_str: Optional[str] = None
            for attempt in range(self.llm_retry_attempts + 1):
                try:
                    llm_chat_response = await self.genie.llm.chat(messages=reasoning_prompt_messages, provider_id=self.llm_provider_id, stop=self.stop_sequences)
                    llm_response_text = llm_chat_response["message"]["content"]
                    llm_error_str = None
                    break
                except Exception as e_llm:
                    llm_error_str = str(e_llm)
                    await self.genie.observability.trace_event("log.warning", {"message": f"LLM call failed (attempt {attempt+1}/{self.llm_retry_attempts+1}): {llm_error_str}"}, "ReActAgent", correlation_id)
                    if attempt < self.llm_retry_attempts:
                        await asyncio.sleep(self.llm_retry_delay * (attempt + 1))
                    else:
                        await self.genie.observability.trace_event("log.error", {"message": f"LLM call failed after all retries. Error: {llm_error_str}"}, "ReActAgent", correlation_id)
                        await self.genie.observability.trace_event("react_agent.llm.error", {"error": llm_error_str, "iteration": i+1}, "ReActAgent", correlation_id)
                        return AgentOutput(status="error", output=f"LLM failed after retries: {llm_error_str}", history=scratchpad)
            if not llm_response_text:
                return AgentOutput(status="error", output="LLM returned no response content.", history=scratchpad)
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
                scratchpad.append(ReActObservation(thought=thought or "N/A", action=action_str, observation=observation_content))
                continue
            tool_name_from_llm = action_tool_match.group(1).strip()
            tool_params_json_str = action_tool_match.group(2).strip()
            try:
                tool_params = json.loads(tool_params_json_str or "{}")
                if not isinstance(tool_params, dict):
                    raise json.JSONDecodeError("Params not a dict", tool_params_json_str,0)
            except json.JSONDecodeError as e_json:
                await self.genie.observability.trace_event("log.warning", {"message": f"Failed to parse JSON params for tool '{tool_name_from_llm}': {e_json}. Params string: '{tool_params_json_str}'"}, "ReActAgent", correlation_id)
                observation_content = f"Error: Invalid JSON parameters for tool '{tool_name_from_llm}': {e_json}. Parameters received: '{tool_params_json_str}'"
                scratchpad.append(ReActObservation(thought=thought or "N/A", action=action_str, observation=observation_content))
                continue
            if tool_name_from_llm not in candidate_tool_ids:
                await self.genie.observability.trace_event("log.warning", {"message": f"LLM chose tool '{tool_name_from_llm}' which is not in the candidate list. Informing LLM."}, "ReActAgent", correlation_id)
                observation_content = f"Error: Tool '{tool_name_from_llm}' is not available or not in the provided list. Available tools: {', '.join(candidate_tool_ids[:3])}..."
                scratchpad.append(ReActObservation(thought=thought or "N/A", action=action_str, observation=observation_content))
                continue
            # Per-action HITL gate (B3). Default off; turned on by
            # agent_config['hitl_per_action']=True for corporate deployments
            # that gate side-effectful tools behind approval.
            if self.hitl_per_action:
                approval_req = {
                    "request_id": f"react_iter_{i+1}_{correlation_id}",
                    "prompt": (
                        f"ReActAgent (iteration {i+1}/{self.max_iterations}) "
                        f"requests approval to execute tool "
                        f"'{tool_name_from_llm}' with params {tool_params}."
                    ),
                    "data_to_approve": {
                        "tool_id": tool_name_from_llm,
                        "params": tool_params,
                        "iteration": i + 1,
                        "thought": thought,
                    },
                }
                approval_resp = await self.genie.human_in_loop.request_approval(approval_req)  # type: ignore
                if approval_resp.get("status") != "approved":
                    deny_reason = approval_resp.get("reason") or "no reason provided"
                    observation_content = (
                        f"Action denied by HITL approver "
                        f"({approval_resp.get('approver_id', 'unknown')}): "
                        f"{deny_reason}"
                    )
                    await self.genie.observability.trace_event(
                        "react_agent.tool.hitl_denied",
                        {
                            "tool_id": tool_name_from_llm,
                            "params": tool_params,
                            "iteration": i + 1,
                            "approver_id": approval_resp.get("approver_id"),
                            "reason": deny_reason,
                        },
                        "ReActAgent",
                        correlation_id,
                    )
                    scratchpad.append(
                        ReActObservation(
                            thought=thought or "N/A",
                            action=action_str,
                            observation=observation_content[:1000],
                        )
                    )
                    continue

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

    async def _run_native_tool_use(self, goal: str, **kwargs: Any) -> AgentOutput:
        """Native tool-use loop (M7).

        Drives the conversation via provider-native ``tool_calls`` instead
        of regex-parsing free text. The provider returns structured tool
        calls; we execute each one, send the result back as a tool
        message, and loop until the model emits an answer with no
        tool_calls. Works against Anthropic, OpenAI, Gemini, and modern
        Ollama models that support function calling.

        The HITL gate (``hitl_per_action``) is honored on this path too.
        Observability events parallel the regex loop's events with
        ``react_agent.tool.*`` names so audit consumers don't need to
        special-case the path.
        """
        correlation_id = str(uuid.uuid4())
        await self.genie.observability.trace_event(
            "react_agent.run.start",
            {"goal": goal, "mode": "native_tool_use"},
            "ReActAgent",
            correlation_id,
        )

        all_tools = await self.genie.tools.list()  # type: ignore[attr-defined]
        candidate_tool_ids = {t.identifier for t in all_tools}
        if not all_tools:
            return AgentOutput(
                status="error",
                output="No tools registered; native tool-use path requires at least one tool.",
                history=[],
            )
        tool_specs = await self._build_openai_function_specs(all_tools)

        messages: List["ChatMessage"] = [
            {
                "role": "system",
                "content": (
                    "You are an agent that achieves goals using tool calls. "
                    "When you have enough information, respond with a plain "
                    "text final answer and no tool calls."
                ),
            },
            {"role": "user", "content": goal},
        ]
        scratchpad: List[ReActObservation] = []

        for i in range(self.max_iterations):
            await self.genie.observability.trace_event(
                "react_agent.iteration.start",
                {"iteration": i + 1, "scratchpad_size": len(scratchpad), "mode": "native_tool_use"},
                "ReActAgent",
                correlation_id,
            )

            try:
                response = await self.genie.llm.chat(
                    messages=messages,
                    provider_id=self.llm_provider_id,
                    tools=tool_specs,
                    tool_choice="auto",
                )
            except Exception as e:
                return AgentOutput(
                    status="error",
                    output=f"LLM call failed: {e}",
                    history=scratchpad,
                )

            assistant_msg = response.get("message") or {}
            tool_calls = assistant_msg.get("tool_calls") or []
            content = assistant_msg.get("content")

            # No tool calls → model produced the final answer.
            if not tool_calls:
                final_answer = content or ""
                scratchpad.append(
                    ReActObservation(thought="", action="Answer", observation=final_answer)
                )
                await self.genie.observability.trace_event(
                    "react_agent.run.success",
                    {"answer": final_answer, "iterations": i + 1, "mode": "native_tool_use"},
                    "ReActAgent",
                    correlation_id,
                )
                return AgentOutput(
                    status="success", output=final_answer, history=scratchpad
                )

            # Track the assistant message in the conversation so the
            # provider has the full context on the next turn.
            messages.append(assistant_msg)

            # Execute each requested tool call in sequence.
            for tc in tool_calls:
                tool_id = tc["function"]["name"]
                raw_args = tc["function"].get("arguments") or "{}"
                try:
                    params = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                    if not isinstance(params, dict):
                        params = {"_raw": params}
                except (json.JSONDecodeError, TypeError):
                    params = {"_raw_arguments": raw_args}

                if tool_id not in candidate_tool_ids:
                    tool_result_str = (
                        f"Error: tool {tool_id!r} not registered. "
                        f"Available: {sorted(candidate_tool_ids)}"
                    )
                else:
                    # HITL gate (B3) honored on the native path too.
                    if self.hitl_per_action:
                        approval_req = {
                            "request_id": f"react_native_{i+1}_{tc.get('id', '')}_{correlation_id}",
                            "prompt": (
                                f"ReActAgent (native, iter {i+1}/{self.max_iterations}) "
                                f"requests approval to execute tool "
                                f"{tool_id!r} with params {params}."
                            ),
                            "data_to_approve": {
                                "tool_id": tool_id,
                                "params": params,
                                "iteration": i + 1,
                            },
                        }
                        approval_resp = await self.genie.human_in_loop.request_approval(  # type: ignore[attr-defined]
                            approval_req
                        )
                        if approval_resp.get("status") != "approved":
                            deny_reason = approval_resp.get("reason") or "no reason"
                            tool_result_str = (
                                f"Action denied by HITL approver "
                                f"({approval_resp.get('approver_id', 'unknown')}): {deny_reason}"
                            )
                            await self.genie.observability.trace_event(
                                "react_agent.tool.hitl_denied",
                                {
                                    "tool_id": tool_id,
                                    "params": params,
                                    "iteration": i + 1,
                                    "approver_id": approval_resp.get("approver_id"),
                                    "reason": deny_reason,
                                    "mode": "native_tool_use",
                                },
                                "ReActAgent",
                                correlation_id,
                            )
                        else:
                            tool_result_str = await self._execute_and_format(
                                tool_id, params, i + 1, correlation_id
                            )
                    else:
                        tool_result_str = await self._execute_and_format(
                            tool_id, params, i + 1, correlation_id
                        )

                scratchpad.append(
                    ReActObservation(
                        thought=content or "",
                        action=f"{tool_id}[{raw_args}]",
                        observation=tool_result_str[:1000],
                    )
                )
                # Send the tool result back to the provider via a tool
                # message keyed to this specific tool_call id.
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": tool_result_str,
                    }
                )

        await self.genie.observability.trace_event(
            "react_agent.run.max_iterations_reached",
            {"goal": goal, "mode": "native_tool_use"},
            "ReActAgent",
            correlation_id,
        )
        return AgentOutput(
            status="max_iterations_reached",
            output=f"Max iterations ({self.max_iterations}) reached.",
            history=scratchpad,
        )

    async def _execute_and_format(
        self, tool_id: str, params: Dict[str, Any], iteration: int, correlation_id: str
    ) -> str:
        """Execute a tool and return its result as a string the LLM can
        observe. Errors become observation strings rather than exceptions
        so the loop continues."""
        await self.genie.observability.trace_event(
            "react_agent.tool.execute.start",
            {"tool_id": tool_id, "params": params, "iteration": iteration, "mode": "native_tool_use"},
            "ReActAgent",
            correlation_id,
        )
        try:
            result = await self.genie.execute_tool(
                tool_id,
                context={"caller_chain": ["ReActAgent.native_tool_use"]},
                **params,
            )
            await self.genie.observability.trace_event(
                "react_agent.tool.execute.success",
                {"tool_id": tool_id, "iteration": iteration, "mode": "native_tool_use"},
                "ReActAgent",
                correlation_id,
            )
            return json.dumps(result) if isinstance(result, (dict, list)) else str(result)
        except Exception as e:
            await self.genie.observability.trace_event(
                "react_agent.tool.execute.error",
                {"tool_id": tool_id, "error": str(e), "iteration": iteration, "mode": "native_tool_use"},
                "ReActAgent",
                correlation_id,
            )
            return f"Error executing tool {tool_id!r}: {e!s}"

    async def _build_openai_function_specs(self, tools) -> List[Dict[str, Any]]:
        """Build OpenAI-function-spec tool definitions from the registered
        tool set. This is the lingua franca the providers accept (Anthropic
        translates internally to its tool_use shape, Gemini to its
        function_declarations shape)."""
        specs: List[Dict[str, Any]] = []
        for tool in tools:
            try:
                meta = await tool.get_metadata()
            except Exception:
                continue
            specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": meta.get("identifier", tool.identifier),
                        "description": meta.get("description_llm")
                        or meta.get("description_human")
                        or meta.get("name", ""),
                        "parameters": meta.get(
                            "input_schema", {"type": "object", "properties": {}}
                        ),
                    },
                }
            )
        return specs
