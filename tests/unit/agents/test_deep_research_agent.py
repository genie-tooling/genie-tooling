### tests/unit/agents/test_deep_research_agent.py
import asyncio
import json
from typing import Any, Dict, List, Optional, Type
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from genie_tooling.agents.deep_research_agent import DeepResearchAgent
from genie_tooling.agents.types import (
    AgentOutput,
    ExecutionEvidence,
    InitialResearchPlan,
    TacticalPlan,
)
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.interfaces import PromptInterface
from pydantic import BaseModel


@pytest.fixture()
def mock_genie_config() -> MiddlewareConfig:
    """Provides a basic MiddlewareConfig for agent tests."""
    return MiddlewareConfig()


@pytest.fixture()
def mock_genie_for_deep_research(
    mocker, mock_genie_config: MiddlewareConfig
) -> MagicMock:
    """Provides a comprehensive mock of the Genie facade for DeepResearchAgent tests."""
    genie_mock = MagicMock(name="MockGenieFacadeForDeepResearch")
    genie_mock._config = mock_genie_config

    # Observability
    genie_mock.observability = AsyncMock(name="MockObservabilityInterface")
    genie_mock.observability.trace_event = AsyncMock()

    # Prompts Interface
    prompts_mock = AsyncMock(spec=PromptInterface)
    prompts_mock.render_prompt = AsyncMock(return_value="Rendered Prompt Content")
    prompts_mock.render_chat_prompt = AsyncMock(
        return_value=[{"role": "user", "content": "Rendered Chat Prompt"}]
    )
    genie_mock.prompts = prompts_mock

    # LLM Interface
    genie_mock.llm = AsyncMock(name="MockLLMInterface")
    genie_mock.llm.chat = AsyncMock(
        return_value={"message": {"content": "{}"}}
    )  # Default empty JSON
    genie_mock.llm.generate = AsyncMock(return_value={"text": "yes"})  # Default for relevance
    genie_mock.llm.parse_output = AsyncMock()

    # Tool Execution
    genie_mock.execute_tool = AsyncMock(
        return_value={"status": "success", "data": "tool executed"}
    )
    return genie_mock


@pytest.fixture()
async def deep_research_agent(mock_genie_for_deep_research: MagicMock) -> DeepResearchAgent:
    """Provides an initialized DeepResearchAgent instance."""
    agent_config = {
        "min_high_quality_sources": 1,  # Set to 1 for easier testing
        "max_replanning_loops": 1,
        "max_tactical_steps": 3,
    }
    agent = DeepResearchAgent(genie=mock_genie_for_deep_research, agent_config=agent_config)
    return agent


@pytest.mark.asyncio()
class TestDeepResearchAgent:
    async def test_generate_initial_plan_success(
        self, deep_research_agent: DeepResearchAgent, mock_genie_for_deep_research: MagicMock
    ):
        agent = await deep_research_agent
        mock_plan = InitialResearchPlan(
            sub_questions=["What is X?", "How does Y relate to X?"]
        )
        mock_genie_for_deep_research.llm.parse_output.return_value = mock_plan

        questions = await agent._generate_initial_plan("goal", "corr-id-1")

        assert questions == ["What is X?", "How does Y relate to X?"]
        mock_genie_for_deep_research.llm.chat.assert_awaited_once()
        mock_genie_for_deep_research.llm.parse_output.assert_awaited_once_with(
            ANY, schema=InitialResearchPlan
        )

    async def test_generate_initial_plan_parse_fail(
        self, deep_research_agent: DeepResearchAgent, mock_genie_for_deep_research: MagicMock
    ):
        agent = await deep_research_agent
        mock_genie_for_deep_research.llm.parse_output.return_value = None
        questions = await agent._generate_initial_plan("goal", "corr-id-2")
        assert questions == []

    async def test_evaluate_evidence_high_quality(
        self, deep_research_agent: DeepResearchAgent, mock_genie_for_deep_research: MagicMock
    ):
        agent = await deep_research_agent
        mock_genie_for_deep_research.llm.generate.return_value = {"text": "yes"}
        evidence = ExecutionEvidence(
            source_details={"type": "web_page"},  # type: ignore
            outcome={"content": "This is very relevant and long content about the goal."},
            source_url="http://example.com"
        )  # type: ignore
        is_quality = await agent._evaluate_evidence(evidence, "goal")
        assert is_quality is True

    async def test_evaluate_evidence_low_quality(
        self, deep_research_agent: DeepResearchAgent, mock_genie_for_deep_research: MagicMock
    ):
        agent = await deep_research_agent
        evidence_with_error = ExecutionEvidence(error="Tool failed")  # type: ignore
        assert not await agent._evaluate_evidence(evidence_with_error, "goal")

        evidence_short_content = ExecutionEvidence(outcome={"content": "short"}, source_url="http://a.com")  # type: ignore
        assert not await agent._evaluate_evidence(evidence_short_content, "goal")

        mock_genie_for_deep_research.llm.generate.return_value = {"text": "no"}
        evidence_irrelevant = ExecutionEvidence(outcome={"content": "long but irrelevant content"}, source_url="http://b.com") # type: ignore
        assert not await agent._evaluate_evidence(evidence_irrelevant, "goal")


    async def test_synthesize_final_report(
        self, deep_research_agent: DeepResearchAgent, mock_genie_for_deep_research: MagicMock
    ):
        agent = await deep_research_agent
        evidence: List[ExecutionEvidence] = [
            {  # type: ignore
                "step_number": 1,
                "source_url": "http://source1.com",
                "action": {"tool_id": "tool1"},
                "outcome": {"data": "info 1"},
                "quality": "high",
            }
        ]
        mock_genie_for_deep_research.llm.generate.return_value = {"text": "## Outline"}
        mock_genie_for_deep_research.llm.chat.return_value = {
            "message": {"content": "Final report text."}
        }
        report = await agent._synthesize_final_report("goal", evidence, "corr-id-3")
        assert report == "Final report text."
        assert mock_genie_for_deep_research.llm.generate.call_count == 1
        assert mock_genie_for_deep_research.llm.chat.call_count == 1

    @pytest.mark.asyncio()
    async def test_run_full_loop_success(
        self, deep_research_agent: DeepResearchAgent, mock_genie_for_deep_research: MagicMock
    ):
        """Test the main `run` method orchestrating the full loop."""
        agent = await deep_research_agent
        # Mocking sub-methods to test the orchestration logic of `run`
        agent._generate_initial_plan = AsyncMock(
            return_value=["What is the capital of France?"]
        )
        agent._execute_tactical_plan_for_question = AsyncMock(
            return_value=[
                ExecutionEvidence(  # type: ignore
                    quality="high",
                    outcome={"content": "Paris is the capital of France."},
                )
            ]
        )
        agent._is_sufficient_evidence = AsyncMock(side_effect=[False, True])
        agent._synthesize_final_report = AsyncMock(return_value="The capital is Paris.")

        result = await agent.run(goal="Find the capital of France.")

        assert result["status"] == "success"
        assert result["output"] == "The capital is Paris."
        agent._generate_initial_plan.assert_awaited_once()
        agent._execute_tactical_plan_for_question.assert_awaited_once()
        assert agent._is_sufficient_evidence.call_count == 2
        agent._synthesize_final_report.assert_awaited_once()

    async def test_run_replanning_loop(self, deep_research_agent: DeepResearchAgent, mock_genie_for_deep_research: MagicMock):
        """Test the replanning logic within the main run loop."""
        agent = await deep_research_agent
        # Setup mocks for a replan scenario
        agent._generate_initial_plan = AsyncMock(return_value=["Initial Question"])

        agent._execute_tactical_plan_for_question = AsyncMock(side_effect=[
            [ExecutionEvidence(quality="low")],  # First call returns low quality
            [ExecutionEvidence(quality="high")]  # Second call (after replan) returns high quality
        ])

        # FIX: Evidence is insufficient on first three checks, sufficient on fourth.
        # This forces the agent to execute, replan, execute again, and then succeed.
        agent._is_sufficient_evidence = AsyncMock(side_effect=[False, False, False, True])
        agent._generate_new_sub_questions = AsyncMock(return_value=["New Question"])
        agent._synthesize_final_report = AsyncMock(return_value="Final report after replan")

        await agent.run(goal="Test replanning")

        # Assert that the replanning logic was triggered
        agent._generate_new_sub_questions.assert_awaited_once()
        # Tactical plan should be executed twice: once for initial, once for new question
        assert agent._execute_tactical_plan_for_question.call_count == 2