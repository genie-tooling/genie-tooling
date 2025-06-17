# src/genie_tooling/agents/deep_research_agent.py
"""
DeepResearchAgent: An advanced, stateful agent designed for in-depth research tasks.

This agent follows a multi-phase process:
1.  **Plan:** Decomposes a high-level goal into a set of smaller, researchable sub-questions.
2.  **Gather:** For each sub-question, it creates and executes a tactical, multi-step plan
    (e.g., search -> retrieve content -> extract data) to gather evidence.
3.  **Evaluate & Adapt:** It assesses the quality of the gathered evidence. If the evidence is
    insufficient or low-quality (e.g., due to network errors or irrelevant content),
    it can adapt its plan by generating new sub-questions to fill the knowledge gaps.
4.  **Synthesize:** Once sufficient high-quality evidence is collected, it performs a two-stage
    synthesis process (outline generation -> full report generation) to produce a comprehensive,
    well-structured, and cited final answer.
"""
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from genie_tooling.utils.placeholder_resolution import resolve_placeholders

from .base_agent import BaseAgent
from .types import AgentOutput, ExecutionEvidence, InitialResearchPlan, TacticalPlan

if TYPE_CHECKING:
    from genie_tooling.genie import Genie

logger = logging.getLogger(__name__)


class DeepResearchAgent(BaseAgent):
    """
    An agent that performs deep research by planning, gathering evidence,
    and synthesizing a comprehensive report.
    """

    def __init__(self, genie: "Genie", agent_config: Optional[Dict[str, Any]] = None):
        super().__init__(genie, agent_config)
        # Tooling Configuration
        self.web_search_tool_id = self.agent_config.get(
            "web_search_tool_id", "intelligent_search_aggregator_v1"
        )
        self.academic_search_tool_id = self.agent_config.get(
            "academic_search_tool_id", "arxiv_search_tool"
        )
        self.content_extraction_tool_id = self.agent_config.get(
            "content_extraction_tool_id", "content_retriever_tool_v1"
        )
        self.data_extraction_tool_id = self.agent_config.get(
            "data_extraction_tool_id", "custom_text_parameter_extractor"
        )

        # LLM & Prompt Configuration
        self.planner_llm_id = self.agent_config.get("planner_llm_id")
        self.solver_llm_id = self.agent_config.get("solver_llm_id")
        self.tool_formatter_id = self.agent_config.get(
            "tool_formatter_id", "compact_text_formatter_plugin_v1"
        )

        # Loop Control Configuration
        self.max_replanning_loops = int(self.agent_config.get("max_replanning_loops", 2))
        self.min_high_quality_sources = int(
            self.agent_config.get("min_high_quality_sources", 3)
        )
        self.max_tactical_steps = int(self.agent_config.get("max_tactical_steps", 4))

        logger.info(
            f"{self.__class__.__name__} initialized with config: {self.agent_config}"
        )

    async def run(self, goal: str, **kwargs: Any) -> AgentOutput:
        """Orchestrates the entire research process."""
        correlation_id = str(uuid.uuid4())
        await self.genie.observability.trace_event(
            "deep_research_agent.run.start",
            {"goal": goal},
            self.__class__.__name__,
            correlation_id,
        )

        # Phase 1: Initial Planning
        sub_questions = await self._generate_initial_plan(goal, correlation_id)
        if not sub_questions:
            return AgentOutput(
                status="error",
                output="Failed to generate an initial research plan.",
                history=[],
                plan=[],
            )

        evidence_locker: List[ExecutionEvidence] = []
        completed_sub_questions: Set[str] = set()
        replan_attempts = 0

        # Main research loop
        while True:
            # Check for sufficiency at the start of each major decision cycle
            if await self._is_sufficient_evidence(evidence_locker):
                logger.info("Sufficient evidence gathered. Proceeding to synthesis.")
                await self.genie.observability.trace_event(
                    "deep_research_agent.evidence.sufficient",
                    {"count": len(evidence_locker)},
                    self.__class__.__name__,
                    correlation_id,
                )
                break

            # Find the next question to work on from the current plan
            next_question = next(
                (q for q in sub_questions if q not in completed_sub_questions), None
            )

            if next_question:
                # If there's a question to process, execute its tactical plan
                tactical_evidence = await self._execute_tactical_plan_for_question(
                    next_question, evidence_locker, correlation_id
                )
                evidence_locker.extend(tactical_evidence)
                completed_sub_questions.add(next_question)
            elif replan_attempts < self.max_replanning_loops:
                # No more questions, and we can still replan
                replan_attempts += 1
                logger.info(
                    f"Evidence insufficient, no more questions. "
                    f"Replanning (Attempt {replan_attempts}/{self.max_replanning_loops})."
                )
                await self.genie.observability.trace_event(
                    "deep_research_agent.replan.start",
                    {"loop": replan_attempts},
                    self.__class__.__name__,
                    correlation_id,
                )
                new_questions = await self._generate_new_sub_questions(
                    goal, evidence_locker, correlation_id
                )
                if not new_questions:
                    logger.warning(
                        "Replanning did not generate new questions. Exiting research loop."
                    )
                    break  # Exit if replanning yields nothing
                sub_questions.extend(new_questions)
            else:
                # No more questions and no more replans allowed, exit the loop
                logger.warning(
                    "Exhausted all questions and replanning attempts. "
                    "Proceeding with current evidence."
                )
                break

        # Phase 4: Synthesis
        final_report = await self._synthesize_final_report(
            goal, evidence_locker, correlation_id
        )

        return AgentOutput(
            status="success", output=final_report, history=evidence_locker, plan=sub_questions
        )

    async def _generate_initial_plan(
        self, goal: str, correlation_id: str
    ) -> List[str]:
        """Generates the initial list of sub-questions to research."""
        prompt = (
            "Based on the user's goal, generate a comprehensive list of sub-questions to research. "
            "Your output MUST be a JSON object with a single key 'sub_questions', which is a list of strings.\n\n"
            f'User Goal: "{goal}"'
        )
        try:
            response = await self.genie.llm.chat(
                [{"role": "user", "content": prompt}],
                provider_id=self.planner_llm_id,
                output_schema=InitialResearchPlan,
            )
            parsed_plan = await self.genie.llm.parse_output(
                response, schema=InitialResearchPlan
            )
            if isinstance(parsed_plan, InitialResearchPlan):
                return parsed_plan.sub_questions
        except Exception as e:
            await self.genie.observability.trace_event(
                "deep_research_agent.plan.initial.error",
                {"error": str(e), "exc_info": True},
                self.__class__.__name__,
                correlation_id,
            )
        return []

    async def _generate_tactical_plan(
        self, sub_question: str, evidence_summary: str, correlation_id: str
    ) -> Optional[TacticalPlan]:
        """Generates a small, executable plan for a single sub-question."""
        all_tools = await self.genie._tool_manager.list_tools(enabled_only=True)  # type: ignore
        tool_definitions = "\n".join(
            filter(
                None,
                [
                    str(
                        await self.genie._tool_manager.get_formatted_tool_definition(  # type: ignore
                            t.identifier, self.tool_formatter_id
                        )
                    )
                    for t in all_tools
                ],
            )
        )

        prompt = (
            f"You are a planner. Your task is to create a small, tactical plan (1-{self.max_tactical_steps} steps) to answer the following sub-question.\n"
            "Use the provided tools. Remember to use search tools first, then use content retrieval on the URLs you find.\n"
            "If a step's output is needed by a later step, you MUST define an `output_variable_name` (e.g., 'search_results', 'page_content').\n"
            "Subsequent steps MUST then reference this output in their `params` using the EXACT syntax `{{{{outputs.variable_name.path.to.value}}}}`.\n"
            'Example for search result URL: `"url": "{{{{outputs.search_variable.results.0.url}}}}"`\n'
            'Example for extracted content: `"text_content": "{{{{outputs.content_variable.content}}}}"`\n'
            f'Sub-question: "{sub_question}"\n\n'
            f"Previously gathered evidence summary (for context):\n{evidence_summary}\n\n"
            f"Available Tools:\n{tool_definitions}\n\n"
            "Your output MUST be a JSON object with a single key 'plan', containing a list of step objects."
        )

        try:
            response = await self.genie.llm.chat(
                [{"role": "user", "content": prompt}],
                provider_id=self.planner_llm_id,
                output_schema=TacticalPlan,
            )
            parsed_plan = await self.genie.llm.parse_output(
                response, schema=TacticalPlan
            )
            if isinstance(parsed_plan, TacticalPlan):
                return parsed_plan
        except Exception as e:
            await self.genie.observability.trace_event(
                "deep_research_agent.plan.tactical.error",
                {"error": str(e), "exc_info": True},
                self.__class__.__name__,
                correlation_id,
            )
        return None

    async def _execute_tactical_plan_for_question(
        self,
        question: str,
        evidence_locker: List[ExecutionEvidence],
        correlation_id: str,
    ) -> List[ExecutionEvidence]:
        """Executes a tactical plan for one sub-question and returns the evidence gathered."""
        evidence_summary = "\n".join(
            [
                f"- Step {ev['step_number']}: {ev['action'].get('tool_id')} -> {ev.get('error') or 'Success'}"
                for ev in evidence_locker
            ]
        )
        tactical_plan = await self._generate_tactical_plan(
            question, evidence_summary, correlation_id
        )
        if not tactical_plan:
            return []

        new_evidence: List[ExecutionEvidence] = []
        scratchpad: Dict[str, Any] = {"outputs": {}}

        for i, step in enumerate(tactical_plan.plan):
            step_error: Optional[str] = None
            tool_result: Any = None
            resolved_params: Dict[str, Any] = {}

            await self.genie.observability.trace_event(
                "rewoo.step.start",
                {
                    "step_number": i + 1,
                    "tool_id": step.tool_id,
                    "params_template": step.params,
                },
                self.__class__.__name__,
                correlation_id,
            )

            try:
                params_with_placeholders = step.params or {}
                if not isinstance(params_with_placeholders, dict):
                    raise TypeError(
                        f"The 'params' field for step {i+1} must be a dictionary, got {type(step.params)}."
                    )

                resolved_params = resolve_placeholders(
                    params_with_placeholders, scratchpad
                )

                tool_context = {"sub_question": question}
                tool_result = await self.genie.execute_tool(
                    step.tool_id, context=tool_context, **resolved_params
                )

                if isinstance(tool_result, dict) and tool_result.get("error"):
                    step_error = str(tool_result["error"])
            except (ValueError, TypeError, KeyError, IndexError) as e_prep:
                step_error = (
                    f"Step {i+1} preparation failed (likely invalid placeholder): {e_prep!s}"
                )
                logger.warning(f"{self.__class__.__name__}: {step_error}", exc_info=True)
            except Exception as e_exec:
                step_error = f"Agent-level execution error: {e_exec}"
                logger.error(
                    f"Error during tactical step execution: {e_exec}", exc_info=True
                )

            evidence_item = ExecutionEvidence(
                step_number=len(evidence_locker) + len(new_evidence) + 1,
                sub_question=question,
                action=step.model_dump(),
                outcome=tool_result,
                error=step_error,
                quality="unevaluated",
                source_url=resolved_params.get("url"),
            )

            is_quality = await self._evaluate_evidence(evidence_item, question)
            evidence_item["quality"] = "high" if is_quality else "low"
            new_evidence.append(evidence_item)

            if not step_error and is_quality and step.output_variable_name:
                scratchpad["outputs"][step.output_variable_name] = tool_result
            elif step.output_variable_name:
                scratchpad["outputs"][step.output_variable_name] = None
                if not step_error:
                    evidence_item[
                        "error"
                    ] = "Content discarded as irrelevant or low-quality."
                    logger.warning(
                        f"Step {i+1} ({step.tool_id}) produced low-quality evidence. Continuing with plan."
                    )

            if step_error:
                await self.genie.observability.trace_event(
                    "deep_research_agent.step.failed",
                    {"step": i + 1, "error": step_error},
                    self.__class__.__name__,
                    correlation_id,
                )
                break

        return new_evidence

    async def _evaluate_evidence(self, evidence: ExecutionEvidence, goal: str) -> bool:
        """Heuristically determines if a piece of evidence is high quality."""
        if evidence.get("error"):
            return False

        outcome = evidence.get("outcome")
        if outcome is None:
            return False

        tool_id = (evidence.get("action") or {}).get("tool_id", "")
        if "search" in tool_id:
            if isinstance(outcome, dict) and outcome.get("results"):
                return True
            else:
                logger.debug(
                    f"Evidence from tool '{tool_id}' discarded: search returned no results."
                )
                return False

        content_to_evaluate: Optional[str] = None
        if isinstance(outcome, dict):
            content_to_evaluate = (
                outcome.get("content")
                or outcome.get("text_content")
                or json.dumps(outcome)
            )
        else:
            content_to_evaluate = str(outcome)

        if not content_to_evaluate or len(content_to_evaluate) < 50:
            return False

        source_url = evidence.get("source_url")
        if source_url:
            relevance_prompt = f"Original goal: '{goal}'\n\nIs the following text snippet relevant to the goal? Answer ONLY with 'yes' or 'no'.\n\nSnippet: {content_to_evaluate[:1500]}..."
            try:
                response = await self.genie.llm.generate(
                    relevance_prompt, temperature=0.0, max_tokens=10
                )
                answer = (response.get("text") or "no").strip().lower()
                is_relevant = answer.startswith("yes")
                if not is_relevant:
                    logger.debug(
                        f"Evidence discarded: LLM deemed content irrelevant. Response: '{answer}'."
                    )
                return is_relevant
            except Exception as e:
                logger.warning(
                    f"LLM relevance check failed: {e}. Assuming not relevant as a precaution."
                )
                return False

        return True

    async def _is_sufficient_evidence(
        self, evidence_locker: List[ExecutionEvidence]
    ) -> bool:
        """Checks if enough high-quality evidence has been gathered."""
        high_quality_count = sum(
            1 for ev in evidence_locker if ev.get("quality") == "high"
        )
        return high_quality_count >= self.min_high_quality_sources

    async def _generate_new_sub_questions(
        self, goal: str, evidence_locker: List[ExecutionEvidence], correlation_id: str
    ) -> List[str]:
        """Asks the LLM to generate new questions to fill knowledge gaps."""
        failed_questions = {
            ev["sub_question"]
            for ev in evidence_locker
            if ev.get("quality") == "low"
        }
        prompt = (
            f"We are researching the goal: '{goal}'.\n"
            f"We have already attempted to research the following sub-questions but failed to find good information: {list(failed_questions)}\n"
            "Please suggest 1-2 new, re-phrased, or alternative sub-questions to find the missing information. "
            "Your output MUST be a JSON object with a single key 'sub_questions', which is a list of strings."
        )
        return await self._generate_initial_plan(prompt, correlation_id)

    async def _synthesize_final_report(
        self, goal: str, evidence_locker: List[ExecutionEvidence], correlation_id: str
    ) -> str:
        """Generates the final, comprehensive report from the gathered evidence."""
        high_quality_evidence = [
            ev for ev in evidence_locker if ev.get("quality") == "high"
        ]
        if not high_quality_evidence:
            return "Could not gather sufficient high-quality information to produce a report."

        evidence_context = "\n\n---\n\n".join(
            f"[Source {ev['step_number']}: {ev.get('source_url') or ev['action'].get('tool_id')}]\n{json.dumps(ev['outcome'], default=str, indent=2)}"
            for ev in high_quality_evidence
        )

        # Stage 1: Outline Generation
        outline_prompt = (
            f"User Goal: {goal}\n\n"
            f"Available Evidence:\n{evidence_context}\n\n"
            "Based on the goal and evidence, create a detailed markdown outline for a comprehensive report. The outline should cover all aspects of the goal."
        )
        outline_response = await self.genie.llm.generate(
            outline_prompt, provider_id=self.solver_llm_id
        )
        report_outline = outline_response.get("text", "Could not generate an outline.")

        # Stage 2: Report Generation
        report_prompt = (
            f"User Goal: {goal}\n\n"
            f"Report Outline:\n{report_outline}\n\n"
            f"Available Evidence:\n{evidence_context}\n\n"
            "You are a world-class writer. Using the provided evidence and adhering strictly to the provided outline, write the full, comprehensive report. "
            "Elaborate on each point in the outline to create a detailed, well-structured essay. "
            "You MUST cite sources from the evidence using markdown footnotes like [^1], [^2], etc. "
            "At the end of the report, list all cited sources under a '### Sources Cited:' heading."
        )
        final_report_response = await self.genie.llm.chat(
            [{"role": "user", "content": report_prompt}], provider_id=self.solver_llm_id
        )
        return final_report_response["message"].get(
            "content"
        ) or "The final synthesis failed to produce a report."
