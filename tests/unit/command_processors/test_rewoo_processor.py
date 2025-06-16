# tests/unit/command_processors/test_rewoo_processor.py
import asyncio
import json
from typing import Any, Dict, List, Optional, Type
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genie_tooling.agents.types import AgentOutput
from genie_tooling.command_processors.impl.rewoo_processor import (
    ReWOOCommandProcessorPlugin,
)
from genie_tooling.input_validators.abc import InputValidator
from pydantic import BaseModel, Field, ValidationError


# Local Pydantic models for testing to ensure data structures are correct
class ReWOOStepForTest(BaseModel):
    step_number: int = 1
    thought: str = "Thinking..."
    tool_id: str
    params: Dict[str, Any] = Field(default_factory=dict)
    output_variable_name: Optional[str] = None


class ReWOOPlanForTest(BaseModel):
    plan: List[ReWOOStepForTest]
    overall_reasoning: Optional[str] = "Overall plan"


@pytest.fixture
def mock_genie_for_rewoo():
    """Provides a mock Genie facade for the ReWOO processor."""
    genie = MagicMock(name="MockGenieFacadeForReWOO")
    genie.observability = AsyncMock()
    genie.observability.trace_event = AsyncMock()
    genie.llm = AsyncMock()
    genie.llm.chat = AsyncMock()
    genie.llm.parse_output = AsyncMock()
    genie.execute_tool = AsyncMock()
    genie._tool_manager = AsyncMock()
    genie._tool_manager.list_tools = AsyncMock(return_value=[])
    genie._tool_manager.get_tool = AsyncMock()
    genie._tool_manager.get_formatted_tool_definition = AsyncMock(
        return_value="Formatted Tool"
    )
    genie._plugin_manager = AsyncMock()
    genie._plugin_manager.get_plugin_instance = AsyncMock()
    genie.prompts = AsyncMock()
    genie.prompts.render_prompt = AsyncMock(return_value="Rendered Prompt")
    return genie


@pytest.fixture
async def rewoo_processor(mock_genie_for_rewoo) -> ReWOOCommandProcessorPlugin:
    """Provides an initialized ReWOO processor."""
    processor = ReWOOCommandProcessorPlugin()
    await processor.setup({"genie_facade": mock_genie_for_rewoo})
    return processor


@pytest.mark.asyncio
class TestReWOOProcessor:
    async def test_generate_plan_success(
        self, rewoo_processor: ReWOOCommandProcessorPlugin, mock_genie_for_rewoo
    ):
        """Test successful plan generation from the LLM."""
        processor = await rewoo_processor
        mock_plan_data = ReWOOPlanForTest(plan=[ReWOOStepForTest(tool_id="tool1")])

        async def mock_parse_output_dynamic(
            response: Any, schema: Type[BaseModel], **kwargs
        ):
            """This mock now uses the schema it's given to create the return object."""
            return schema(**mock_plan_data.model_dump())

        mock_genie_for_rewoo.llm.parse_output = AsyncMock(
            side_effect=mock_parse_output_dynamic
        )

        plan, _ = await processor._generate_plan("goal", "tools", ["tool1"], "corr_id")

        assert plan is not None
        mock_genie_for_rewoo.llm.chat.assert_awaited_once()
        mock_genie_for_rewoo.llm.parse_output.assert_awaited_once()

    async def test_generate_plan_retries_on_failure(
        self, rewoo_processor: ReWOOCommandProcessorPlugin, mock_genie_for_rewoo
    ):
        """Test that planning retries if the LLM output fails validation."""
        processor = await rewoo_processor
        processor._max_plan_retries = 1
        mock_plan_data = ReWOOPlanForTest(plan=[ReWOOStepForTest(tool_id="t1")])
        call_count = 0

        async def mock_parse_output_with_failure(
            response: Any, schema: Type[BaseModel], **kwargs
        ):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValidationError.from_exception_data(
                    "Validation Error", [{"type": "missing", "loc": ("plan",)}]
                )
            return schema(**mock_plan_data.model_dump())

        mock_genie_for_rewoo.llm.parse_output = AsyncMock(
            side_effect=mock_parse_output_with_failure
        )

        await processor._generate_plan("goal", "tools", ["t1"], "corr_id")
        assert mock_genie_for_rewoo.llm.chat.call_count == 2
        assert mock_genie_for_rewoo.llm.parse_output.call_count == 2

    async def test_execute_plan_success(
        self, rewoo_processor: ReWOOCommandProcessorPlugin, mock_genie_for_rewoo
    ):
        """Test successful execution of a valid plan."""
        processor = await rewoo_processor
        mock_genie_for_rewoo.execute_tool.return_value = {"data": "success"}
        processor._is_high_quality_evidence = AsyncMock(return_value=True)

        plan_model = ReWOOPlanForTest(plan=[ReWOOStepForTest(tool_id="tool1")])
        result = await processor._execute_plan(plan_model, "goal", "corr_id")

        assert result["status"] == "success"
        mock_genie_for_rewoo.execute_tool.assert_awaited_once()

    async def test_execute_plan_tool_error(
        self, rewoo_processor: ReWOOCommandProcessorPlugin, mock_genie_for_rewoo
    ):
        """Test that a tool execution error halts the plan."""
        processor = await rewoo_processor
        mock_genie_for_rewoo.execute_tool.side_effect = RuntimeError("Tool failed")

        plan_model = ReWOOPlanForTest(plan=[ReWOOStepForTest(tool_id="tool1")])
        result = await processor._execute_plan(plan_model, "goal", "corr_id")

        assert result["status"] == "error"
        assert "Execution failed at step 1" in result["final_output"]

    async def test_process_command_integration(
        self, rewoo_processor: ReWOOCommandProcessorPlugin, mock_genie_for_rewoo
    ):
        """Test the main `process_command` orchestrator."""
        processor = await rewoo_processor
        
        # FIX: Configure agent to require only 1 source to prevent replanning loop
        processor._min_high_quality_sources = 1
        
        # FIX: Configure tool manager to return a tool so planning doesn't fail early.
        mock_tool_for_integration = MagicMock()
        mock_tool_for_integration.identifier = "t1"
        mock_genie_for_rewoo._tool_manager.list_tools.return_value = [
            mock_tool_for_integration
        ]

        # Mock the internal methods to test their orchestration
        mock_plan = ReWOOPlanForTest(plan=[ReWOOStepForTest(tool_id="t1")])

        processor._generate_plan = AsyncMock(return_value=(mock_plan, "raw_plan"))
        processor._execute_plan = AsyncMock(
            return_value={
                "status": "success",
                "final_output": "Plan executed successfully",
                "evidence": [
                    {
                        "step": {"tool_id": "t1"},
                        "outcome": {"data": "success"},
                        "error": None, # Ensure error key is present
                        "source_details": {
                            "type": "t",
                            "identifier": "i",
                            "title": "T",
                        },
                    }
                ],
            }
        )
        # FIX: Mock _is_high_quality_evidence to return True.
        processor._is_high_quality_evidence = AsyncMock(return_value=True)

        processor._synthesize_answer = AsyncMock(
            return_value=("Final Synthesized Answer", "raw_solver_output")
        )

        result = await processor.process_command("test command")
        assert result.get("final_answer") == "Final Synthesized Answer"
        processor._generate_plan.assert_awaited_once() # This should now pass
        processor._execute_plan.assert_awaited_once()
        processor._synthesize_answer.assert_awaited_once()