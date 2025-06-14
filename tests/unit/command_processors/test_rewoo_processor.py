# tests/unit/command_processors/test_rewoo_processor.py
import asyncio
import json
from typing import Any, Dict, List, Optional, Type
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest
from genie_tooling.agents.types import (
    AgentOutput,
    PlannedStep,
    ReActObservation,
)
from genie_tooling.command_processors.impl.rewoo_processor import (
    ReWOOCommandProcessorPlugin,
)
from genie_tooling.interfaces import PromptInterface
from pydantic import BaseModel, Field, ValidationError
import logging
logger = logging.getLogger(__name__)


# --- FIX: Define local Pydantic models that match the ReWOO processor's dynamic model structure ---
class ReWOOStep(BaseModel):
    # FIX: Add the missing step_number field.
    step_number: int
    thought: str
    tool_id: str
    params: str  # The model expects a JSON string
    output_variable_name: Optional[str] = None


class ReWOOPlan(BaseModel):
    plan: List[ReWOOStep]
    overall_reasoning: Optional[str] = None


# --- Fixtures ---
@pytest.fixture
def mock_genie_for_rewoo(mocker) -> MagicMock:
    """Provides a comprehensive mock of the Genie facade for ReWOO tests."""
    genie = mocker.MagicMock(name="MockGenieFacadeForReWOO")

    # LLM Mock
    genie.llm = AsyncMock(name="MockLLMInterface")
    genie.llm.chat = AsyncMock(return_value={"message": {"content": "{}"}})  # Default to empty JSON
    genie.llm.parse_output = AsyncMock()

    # Prompts Mock
    genie.prompts = AsyncMock(spec=PromptInterface)
    genie.prompts.render_prompt = AsyncMock(return_value="Rendered Prompt Content")

    # Tool Manager Mock
    genie._tool_manager = AsyncMock(name="MockToolManagerOnGenie")
    mock_tool_instance = MagicMock(identifier="default_mock_tool_id")
    genie._tool_manager.list_tools = AsyncMock(
        return_value=[mock_tool_instance]
    )
    genie._tool_manager.get_formatted_tool_definition = AsyncMock(
        return_value="Formatted Tool Def"
    )

    # Observability Mock
    genie.observability = AsyncMock(name="MockObservabilityInterface")
    genie.observability.trace_event = AsyncMock()

    # Tool Execution Mock
    genie.execute_tool = AsyncMock(return_value={"status": "success", "data": "tool executed"})

    return genie


@pytest.fixture
async def rewoo_processor() -> ReWOOCommandProcessorPlugin:
    """Provides an instance of the processor."""
    return ReWOOCommandProcessorPlugin()


# --- Tests ---
@pytest.mark.asyncio
class TestReWOOProcessorSetup:
    async def test_setup_success(self, rewoo_processor: ReWOOCommandProcessorPlugin, mock_genie_for_rewoo: MagicMock):
        """Test successful setup with a valid configuration."""
        processor = await rewoo_processor
        config = {
            "genie_facade": mock_genie_for_rewoo,
            "planner_llm_provider_id": "gemini-pro",
            "max_plan_retries": 2,
        }
        await processor.setup(config)
        assert processor._genie is mock_genie_for_rewoo
        assert processor._planner_llm_id == "gemini-pro"
        assert processor._max_plan_retries == 2

    async def test_setup_missing_genie_facade_raises_error(self, rewoo_processor: ReWOOCommandProcessorPlugin):
        """Test that setup raises a ValueError if the genie_facade is missing."""
        processor = await rewoo_processor
        with pytest.raises(ValueError, match="requires a 'genie_facade' instance in its config"):
            await processor.setup({})


@pytest.mark.asyncio
class TestReWOOProcessorPlanGeneration:
    async def test_generate_plan_success(
        self, rewoo_processor: ReWOOCommandProcessorPlugin, mock_genie_for_rewoo: MagicMock
    ):
        """Test the success path for generating a valid plan."""
        processor = await rewoo_processor
        await processor.setup({"genie_facade": mock_genie_for_rewoo})

        # FIX: Create a side effect for the mock that correctly instantiates the dynamic schema
        async def mock_parse_output(response: Any, schema: Type[BaseModel], **kwargs):
            if "DynamicReWOOPlan" in schema.__name__:
                plan_data = {
                    "plan": [
                        {"step_number": 1, "thought": "Reason", "tool_id": "tool1", "params": '{"p": 1}'}
                    ]
                }
                return schema(**plan_data)
            return None

        mock_genie_for_rewoo.llm.parse_output.side_effect = mock_parse_output

        plan_model, raw_output = await processor._generate_plan("goal", "tools_def", ["tool1"], "corr_id")

        assert plan_model is not None
        assert isinstance(plan_model, BaseModel)
        assert len(plan_model.plan) == 1  # type: ignore
        assert plan_model.plan[0].tool_id == "tool1"  # type: ignore
        mock_genie_for_rewoo.prompts.render_prompt.assert_awaited_once()
        mock_genie_for_rewoo.llm.chat.assert_awaited_once()
        mock_genie_for_rewoo.llm.parse_output.assert_awaited_once()

    async def test_generate_plan_retries_on_validation_error(
        self, rewoo_processor: ReWOOCommandProcessorPlugin, mock_genie_for_rewoo: MagicMock
    ):
        """Test that planning retries if the first LLM output fails Pydantic validation."""
        processor = await rewoo_processor
        await processor.setup({"genie_facade": mock_genie_for_rewoo, "max_plan_retries": 1})

        # FIX: Use a side effect to simulate failure then success with the correct dynamic type
        async def mock_parse_output_retry(response: Any, schema: Type[BaseModel], **kwargs):
            # Fail on the first call
            if mock_genie_for_rewoo.llm.parse_output.call_count == 1:
                raise ValidationError.from_exception_data("e", [])
            # Succeed on the second call
            if "DynamicReWOOPlan" in schema.__name__:
                plan_data = {"plan": [{"step_number": 1, "thought": "R", "tool_id": "t1", "params": '{"key":"val"}'}]}
                return schema(**plan_data)
            return None

        mock_genie_for_rewoo.llm.parse_output.side_effect = mock_parse_output_retry

        plan, raw_output = await processor._generate_plan("goal", "tools_def", ["t1"], "corr_id")

        assert plan is not None
        assert mock_genie_for_rewoo.llm.chat.call_count == 2
        assert mock_genie_for_rewoo.llm.parse_output.call_count == 2
        render_call_args = mock_genie_for_rewoo.prompts.render_prompt.call_args_list
        assert "PREVIOUS ATTEMPT FAILED" in render_call_args[1].kwargs["data"]["previous_attempt_feedback"]

    async def test_generate_plan_fails_after_all_retries(
        self, rewoo_processor: ReWOOCommandProcessorPlugin, mock_genie_for_rewoo: MagicMock
    ):
        """Test that planning returns None after exhausting all retries."""
        processor = await rewoo_processor
        await processor.setup({"genie_facade": mock_genie_for_rewoo, "max_plan_retries": 1})
        mock_genie_for_rewoo.llm.parse_output.side_effect = [
            ValidationError.from_exception_data("e", []),
            ValueError("Still wrong"),
        ]

        plan, raw_output = await processor._generate_plan("goal", "tools_def", ["t1"], "corr_id")

        assert plan is None
        assert mock_genie_for_rewoo.llm.chat.call_count == 2
        assert mock_genie_for_rewoo.llm.parse_output.call_count == 2


@pytest.mark.asyncio
class TestReWOOProcessorPlanExecution:
    async def test_execute_plan_success_with_placeholder(
        self, rewoo_processor: ReWOOCommandProcessorPlugin, mock_genie_for_rewoo: MagicMock
    ):
        """Test successful execution of a plan with a placeholder reference."""
        processor = await rewoo_processor
        await processor.setup({"genie_facade": mock_genie_for_rewoo})

        # Use the locally defined, correct Pydantic model for ReWOO
        plan = ReWOOPlan(
            plan=[
                ReWOOStep(step_number=1, thought="R1", tool_id="tool_A", params="{}", output_variable_name="outA"),
                ReWOOStep(step_number=2, thought="R2", tool_id="tool_B", params='{"input": "{{outputs.outA.result}}"}'),
            ]
        )
        mock_genie_for_rewoo.execute_tool.side_effect = [
            {"result": "output_from_A"},
            {"result": "final_output"},
        ]
        agent_result = await processor._execute_plan(plan, "goal", "corr_id")

        assert agent_result["status"] == "success"
        assert len(agent_result["evidence"]) == 2
        assert mock_genie_for_rewoo.execute_tool.call_count == 2
        # `execute_tool` is called with keyword arguments, not positional.
        second_call_kwargs = mock_genie_for_rewoo.execute_tool.call_args_list[1].kwargs
        assert second_call_kwargs["input"] == "output_from_A"

    async def test_execute_plan_placeholder_resolution_fails(
        self, rewoo_processor: ReWOOCommandProcessorPlugin, mock_genie_for_rewoo: MagicMock
    ):
        """Test that execution halts if a placeholder cannot be resolved."""
        processor = await rewoo_processor
        await processor.setup({"genie_facade": mock_genie_for_rewoo})

        # Use the locally defined, correct Pydantic model for ReWOO
        plan = ReWOOPlan(
            plan=[
                ReWOOStep(step_number=1, thought="R1", tool_id="tool_A", params='{"input": "{{outputs.non_existent.value}}"}'),
            ]
        )
        agent_result = await processor._execute_plan(plan, "goal", "corr_id")

        assert agent_result["status"] == "error"
        assert "Parameter resolution/parsing failed" in agent_result["evidence"][0]["error"]
        assert mock_genie_for_rewoo.execute_tool.call_count == 0


@pytest.mark.asyncio
class TestReWOOProcessorSummarization:
    async def test_process_step_result_for_evidence_web_page_long_content(
        self, rewoo_processor: ReWOOCommandProcessorPlugin, mock_genie_for_rewoo: MagicMock
    ):
        """Test that long web page content triggers LLM summarization for the evidence."""
        processor = await rewoo_processor
        await processor.setup({"genie_facade": mock_genie_for_rewoo})
        long_content = "a" * 10000
        tool_result = {"content": long_content, "url": "http://example.com"}
        step_model_dict = {"step_number": 1, "tool_id": "web_page_content_extractor", "reasoning": "Get details", "params": "{}"}
        mock_genie_for_rewoo.llm.generate.return_value = {"text": "Summarized content"}

        evidence = await processor._process_step_result_for_evidence(
            step_model_dict, tool_result, None, "goal", "corr_id"
        )

        assert evidence["detailed_summary_or_extraction"] == "Summarized content"
        mock_genie_for_rewoo.llm.generate.assert_awaited_once()

    async def test_process_step_result_for_evidence_search_tool(
        self, rewoo_processor: ReWOOCommandProcessorPlugin, mock_genie_for_rewoo: MagicMock
    ):
        """Test that search tool results are formatted into a readable summary."""
        processor = await rewoo_processor
        await processor.setup({"genie_facade": mock_genie_for_rewoo})
        tool_result = {
            "results": [
                {"title": "Title 1", "url": "url1", "snippet_or_summary": "Snippet 1"},
                {"title": "Title 2", "url": "url2", "snippet_or_summary": "Snippet 2"},
            ]
        }
        step_model_dict = {"step_number": 1, "tool_id": "intelligent_search_aggregator_v1", "params": '{"query": "test"}'}
        evidence = await processor._process_step_result_for_evidence(
            step_model_dict, tool_result, None, "goal", "corr_id"
        )
        summary = evidence["detailed_summary_or_extraction"]
        assert isinstance(summary, str)
        assert "Item 1:" in summary
        assert "Title: Title 1" in summary
        assert "URL/ID: url1" in summary
        assert "Snippet: Snippet 1" in summary
        assert "Item 2:" in summary