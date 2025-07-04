# tests/unit/agents/test_deep_research_agent.py
from unittest.mock import AsyncMock, MagicMock

import pytest
from genie_tooling.agents.deep_research_agent import DeepResearchAgent
from genie_tooling.agents.types import (
    ExecutionEvidence,
    InitialResearchPlan,
    TacticalPlan,
    TacticalPlanStep,
)
from genie_tooling.interfaces import PromptInterface


@pytest.fixture()
def mock_genie_for_dra():
    """Provides a comprehensive mock Genie facade for DeepResearchAgent tests."""
    genie = MagicMock(name="MockGenieForDeepResearch")
    genie.observability = AsyncMock()
    genie.observability.trace_event = AsyncMock()
    genie.llm = AsyncMock()
    genie.llm.chat = AsyncMock()
    genie.llm.generate = AsyncMock()
    genie.llm.parse_output = AsyncMock()
    genie.execute_tool = AsyncMock()
    genie.run_command = AsyncMock()
    genie.prompts = AsyncMock(spec=PromptInterface)
    genie.prompts.render_prompt = AsyncMock(return_value="Rendered Prompt")
    genie._tool_manager = AsyncMock()
    genie._tool_manager.list_tools = AsyncMock(return_value=[])
    genie._tool_manager.get_formatted_tool_definition = AsyncMock(
        return_value="Tool Definition"
    )
    return genie


@pytest.fixture()
async def deep_research_agent(mock_genie_for_dra) -> DeepResearchAgent:
    """Provides an initialized DeepResearchAgent instance."""
    agent = DeepResearchAgent(
        genie=mock_genie_for_dra, agent_config={"min_high_quality_sources": 1}
    )
    return agent


@pytest.mark.asyncio()
class TestDeepResearchAgentLogic:
    async def test_generate_tactical_plan_success(
        self, deep_research_agent: DeepResearchAgent, mock_genie_for_dra
    ):
        """Test tactical plan generation."""
        agent = await deep_research_agent
        mock_plan = TacticalPlan(
            plan=[
                TacticalPlanStep(
                    thought="t", tool_id="search", params={}, output_variable_name="res"
                )
            ]
        )
        mock_genie_for_dra.llm.parse_output.return_value = mock_plan

        plan = await agent._generate_tactical_plan("question", "summary", "corr_id")

        assert plan is not None
        assert len(plan.plan) == 1
        mock_genie_for_dra.llm.chat.assert_awaited_once()
        mock_genie_for_dra.llm.parse_output.assert_awaited_once()

    async def test_execute_tactical_plan_for_question_success(
        self, deep_research_agent: DeepResearchAgent, mock_genie_for_dra
    ):
        """Test execution of a tactical plan."""
        agent = await deep_research_agent
        mock_plan = TacticalPlan(
            plan=[
                TacticalPlanStep(
                    thought="t", tool_id="search", params={}, output_variable_name="res"
                )
            ]
        )
        agent._generate_tactical_plan = AsyncMock(return_value=mock_plan)
        mock_genie_for_dra.execute_tool.return_value = {"results": ["found one"]}
        agent._evaluate_evidence = AsyncMock(return_value=True)

        evidence = await agent._execute_tactical_plan_for_question(
            "question", [], "corr_id"
        )

        assert len(evidence) == 1
        assert evidence[0]["quality"] == "high"
        mock_genie_for_dra.execute_tool.assert_awaited_once()

    async def test_execute_tactical_plan_placeholder_failure(
        self, deep_research_agent: DeepResearchAgent, mock_genie_for_dra
    ):
        """Test plan execution failing on placeholder resolution."""
        agent = await deep_research_agent
        mock_plan = TacticalPlan(
            plan=[
                TacticalPlanStep(
                    thought="t",
                    tool_id="read",
                    params={"url": "{{outputs.non_existent.url}}"},
                )
            ]
        )
        agent._generate_tactical_plan = AsyncMock(return_value=mock_plan)

        evidence = await agent._execute_tactical_plan_for_question(
            "question", [], "corr_id"
        )

        assert len(evidence) == 1
        assert evidence[0]["quality"] == "low"
        assert "placeholder" in evidence[0]["error"]
        mock_genie_for_dra.execute_tool.assert_not_called()

    async def test_generate_new_sub_questions(
        self, deep_research_agent: DeepResearchAgent, mock_genie_for_dra
    ):
        """Test the replanning logic."""
        agent = await deep_research_agent
        evidence = [
            ExecutionEvidence(quality="low", sub_question="q1", action={}, outcome={})
        ]
        mock_plan = InitialResearchPlan(sub_questions=["new question"])
        mock_genie_for_dra.llm.parse_output.return_value = mock_plan

        new_questions = await agent._generate_new_sub_questions(
            "goal", evidence, "corr_id"
        )
        assert new_questions == ["new question"]
        mock_genie_for_dra.llm.chat.assert_awaited_once()
        prompt_arg = mock_genie_for_dra.llm.chat.call_args.args[0][0]["content"]
        assert "q1" in prompt_arg

    async def test_run_exits_if_replan_is_futile(
        self, deep_research_agent: DeepResearchAgent, mock_genie_for_dra
    ):
        """Test that the main loop exits if replanning doesn't produce new questions."""
        agent = await deep_research_agent
        agent._generate_initial_plan = AsyncMock(return_value=["q1"])
        agent._execute_tactical_plan_for_question = AsyncMock(
            return_value=[ExecutionEvidence(quality="low")]
        )
        agent._is_sufficient_evidence = AsyncMock(return_value=False)
        agent._generate_new_sub_questions = AsyncMock(
            return_value=[]
        )  # No new questions
        agent._synthesize_final_report = AsyncMock(
            return_value="Synthesized from partial evidence."
        )

        result = await agent.run("test futile replan")

        agent._generate_new_sub_questions.assert_awaited_once()
        agent._synthesize_final_report.assert_awaited_once()
        assert result["status"] == "success"

    async def test_run_full_loop_success(
        self, deep_research_agent: DeepResearchAgent, mock_genie_for_dra
    ):
        """Test the main `run` method orchestrating the full loop."""
        agent = await deep_research_agent
        agent._generate_initial_plan = AsyncMock(
            return_value=["What is the capital of France?"]
        )
        agent._execute_tactical_plan_for_question = AsyncMock(
            return_value=[
                ExecutionEvidence(
                    quality="high",
                    outcome={"content": "Paris is the capital of France."},
                )
            ]
        )

        # We need it to run once, fail the check, and then pass on the second check.
        agent._is_sufficient_evidence = AsyncMock(side_effect=[False, True])
        agent._synthesize_final_report = AsyncMock(
            return_value="The capital is Paris."
        )

        result = await agent.run(goal="Find the capital of France.")

        assert result["status"] == "success"
        assert result["output"] == "The capital is Paris."
        agent._generate_initial_plan.assert_awaited_once()
        agent._execute_tactical_plan_for_question.assert_awaited_once()
        assert agent._is_sufficient_evidence.call_count == 2
        agent._synthesize_final_report.assert_awaited_once()

    async def test_run_replanning_loop(
        self, deep_research_agent: DeepResearchAgent, mock_genie_for_dra
    ):
        """Test the replanning logic within the main run loop."""
        agent = await deep_research_agent
        agent._generate_initial_plan = AsyncMock(return_value=["Initial Question"])
        agent._execute_tactical_plan_for_question = AsyncMock(
            side_effect=[
                [ExecutionEvidence(quality="low")],
                [ExecutionEvidence(quality="high")],
            ]
        )

        # the initial execution, the replan, and the second execution before
        # the final check passes and exits the loop.
        agent._is_sufficient_evidence = AsyncMock(side_effect=[False, False, False, True])
        agent._generate_new_sub_questions = AsyncMock(return_value=["New Question"])
        agent._synthesize_final_report = AsyncMock(
            return_value="Final report after replan"
        )

        await agent.run(goal="Test replanning")

        agent._generate_new_sub_questions.assert_awaited_once()
        assert agent._execute_tactical_plan_for_question.call_count == 2


@pytest.mark.asyncio()
class TestDeepResearchAgentEvidenceAndSynthesis:
    async def test_synthesize_final_report_no_evidence(
        self, deep_research_agent: DeepResearchAgent
    ):
        """Test that report synthesis handles having no high-quality evidence."""
        agent = await deep_research_agent
        report = await agent._synthesize_final_report("goal", [], "corr_id")
        assert "Could not gather sufficient" in report

    async def test_synthesize_final_report_success(
        self, deep_research_agent: DeepResearchAgent, mock_genie_for_dra
    ):
        """Test successful two-stage report synthesis."""
        agent = await deep_research_agent
        evidence = [
            ExecutionEvidence(
                quality="high",
                step_number=1,
                outcome={"content": "Content A"},
                action={"tool_id": "toolA"},
                sub_question="q1",
                source_url="http://a.com",
            )
        ]
        mock_genie_for_dra.llm.generate.return_value = {"text": "Generated Outline"}
        mock_genie_for_dra.llm.chat.return_value = {
            "message": {"content": "Final Report Body"}
        }

        report = await agent._synthesize_final_report("goal", evidence, "corr_id")
        assert report == "Final Report Body"
        # Check that both LLM calls (outline and report) were made
        mock_genie_for_dra.llm.generate.assert_awaited_once()
        mock_genie_for_dra.llm.chat.assert_awaited_once()
        # Check that the evidence was included in the prompts
        assert "Content A" in mock_genie_for_dra.llm.generate.call_args.args[0]
        assert "Generated Outline" in mock_genie_for_dra.llm.chat.call_args.args[0][0]["content"]

    async def test_evaluate_evidence_with_error(
        self, deep_research_agent: DeepResearchAgent
    ):
        agent = await deep_research_agent
        evidence = ExecutionEvidence(error="Tool failed")
        assert await agent._evaluate_evidence(evidence, "goal") is False

    async def test_evaluate_evidence_no_outcome(
        self, deep_research_agent: DeepResearchAgent
    ):
        agent = await deep_research_agent
        evidence = ExecutionEvidence(outcome=None)
        assert await agent._evaluate_evidence(evidence, "goal") is False

    async def test_evaluate_evidence_search_tool_with_results(
        self, deep_research_agent: DeepResearchAgent
    ):
        agent = await deep_research_agent
        evidence = ExecutionEvidence(
            action={"tool_id": "intelligent_search_aggregator_v1"},
            outcome={"results": [{"title": "found"}]},
        )
        assert await agent._evaluate_evidence(evidence, "goal") is True

    async def test_evaluate_evidence_content_too_short(
        self, deep_research_agent: DeepResearchAgent
    ):
        agent = await deep_research_agent
        evidence = ExecutionEvidence(outcome={"content": "short"})
        assert await agent._evaluate_evidence(evidence, "goal") is False

    async def test_evaluate_evidence_llm_relevance_check_pass(
        self, deep_research_agent: DeepResearchAgent, mock_genie_for_dra
    ):
        agent = await deep_research_agent
        mock_genie_for_dra.llm.generate.return_value = {"text": "yes, this is relevant"}
        long_content = "This is a long and detailed piece of text that is definitely over fifty characters long to pass the initial length check."
        evidence = ExecutionEvidence(
            outcome={"content": long_content},
            source_url="http://example.com",
        )
        assert await agent._evaluate_evidence(evidence, "goal") is True

    async def test_evaluate_evidence_llm_relevance_check_fail(
        self, deep_research_agent: DeepResearchAgent, mock_genie_for_dra
    ):
        agent = await deep_research_agent
        mock_genie_for_dra.llm.generate.return_value = {"text": "no."}
        long_content = "This is a long and detailed piece of text that is definitely over fifty characters long to pass the initial length check."
        evidence = ExecutionEvidence(
            outcome={"content": long_content},
            source_url="http://example.com",
        )
        assert await agent._evaluate_evidence(evidence, "goal") is False

    async def test_evaluate_evidence_llm_relevance_check_raises_error(
        self, deep_research_agent: DeepResearchAgent, mock_genie_for_dra
    ):
        agent = await deep_research_agent
        mock_genie_for_dra.llm.generate.side_effect = RuntimeError("LLM API error")
        long_content = "This is a long and detailed piece of text that is definitely over fifty characters long to pass the initial length check."
        evidence = ExecutionEvidence(
            outcome={"content": long_content},
            source_url="http://example.com",
        )
        assert await agent._evaluate_evidence(evidence, "goal") is False
