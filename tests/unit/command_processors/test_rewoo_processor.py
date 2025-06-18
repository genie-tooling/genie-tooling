# tests/unit/command_processors/test_rewoo_processor.py
import logging
from typing import Any, Dict, List, Optional, Type
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genie_tooling.command_processors.impl.rewoo_processor import (
    ReWOOCommandProcessorPlugin,
)
from genie_tooling.interfaces import PromptInterface
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


PROCESSOR_LOGGER_NAME = "genie_tooling.command_processors.impl.rewoo_processor"


@pytest.fixture()
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
    genie.prompts = AsyncMock(spec=PromptInterface)
    genie.prompts.render_prompt = AsyncMock(return_value="Rendered Prompt")
    return genie


@pytest.fixture()
def rewoo_processor() -> ReWOOCommandProcessorPlugin:
    """Provides a default, un-setup instance of the processor."""
    return ReWOOCommandProcessorPlugin()


@pytest.mark.asyncio()
class TestReWOOProcessor:
    async def test_rewoo_process_command_fails_without_facade(
        self,
        rewoo_processor: ReWOOCommandProcessorPlugin,
    ):
        """
        Tests that process_command fails gracefully if setup was called without a facade.
        """
        processor = rewoo_processor
        # Simulate setup without the facade
        await processor.setup(config=None)

        # Act
        response = await processor.process_command(command="research topic")

        # Assert
        assert "error" in response
        assert "not properly initialized" in response["error"]

    async def test_rewoo_setup_with_genie_facade(
        self,
        rewoo_processor: ReWOOCommandProcessorPlugin,
        mock_genie_for_rewoo: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ):
        """
        Tests that setup completes successfully when a valid genie_facade is provided.
        """
        processor = rewoo_processor
        config = {
            "genie_facade": mock_genie_for_rewoo,
            "max_plan_retries": 2,
        }
        caplog.set_level(logging.INFO, logger=PROCESSOR_LOGGER_NAME)

        # Act
        await processor.setup(config=config)

        # Assert
        assert processor._genie is mock_genie_for_rewoo
        assert processor._max_plan_retries == 2
        # Ensure no error was raised or logged about the facade
        assert "requires a 'genie_facade' instance" not in caplog.text

    async def test_generate_plan_success(
        self, rewoo_processor: ReWOOCommandProcessorPlugin, mock_genie_for_rewoo
    ):
        """Test successful plan generation from the LLM."""
        processor = rewoo_processor
        await processor.setup({"genie_facade": mock_genie_for_rewoo})

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
        processor = rewoo_processor
        await processor.setup(
            {"genie_facade": mock_genie_for_rewoo, "max_plan_retries": 1}
        )
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

    async def test_generate_plan_fails_all_retries(
        self, rewoo_processor: ReWOOCommandProcessorPlugin, mock_genie_for_rewoo
    ):
        """Test that planning returns None after all retries fail."""
        processor = rewoo_processor
        await processor.setup(
            {"genie_facade": mock_genie_for_rewoo, "max_plan_retries": 1}
        )

        mock_genie_for_rewoo.llm.parse_output = AsyncMock(
            side_effect=ValueError("Persistent parse error")
        )

        plan, raw_output = await processor._generate_plan(
            "goal", "tools", ["t1"], "corr_id"
        )

        assert plan is None
        assert raw_output is not None
        assert mock_genie_for_rewoo.llm.parse_output.call_count == 2

    async def test_generate_plan_invalid_tool_id(
        self, rewoo_processor: ReWOOCommandProcessorPlugin, mock_genie_for_rewoo
    ):
        """Test plan generation fails validation if an invalid tool_id is used."""
        processor = rewoo_processor
        await processor.setup(
            {"genie_facade": mock_genie_for_rewoo, "max_plan_retries": 0}
        )

        # LLM hallucinates a tool not in the candidate list
        mock_invalid_plan = ReWOOPlanForTest(
            plan=[ReWOOStepForTest(tool_id="invalid_tool_id")]
        )
        mock_genie_for_rewoo.llm.parse_output = AsyncMock(
            return_value=mock_invalid_plan
        )

        plan, _ = await processor._generate_plan(
            "goal", "tools", ["valid_tool_1"], "corr_id"
        )

        assert plan is None

    async def test_execute_plan_success(
        self, rewoo_processor: ReWOOCommandProcessorPlugin, mock_genie_for_rewoo
    ):
        """Test successful execution of a valid plan."""
        processor = rewoo_processor
        await processor.setup({"genie_facade": mock_genie_for_rewoo})

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
        processor = rewoo_processor
        await processor.setup({"genie_facade": mock_genie_for_rewoo})

        mock_genie_for_rewoo.execute_tool.side_effect = RuntimeError("Tool failed")

        plan_model = ReWOOPlanForTest(plan=[ReWOOStepForTest(tool_id="tool1")])
        result = await processor._execute_plan(plan_model, "goal", "corr_id")

        assert result["status"] == "error"
        assert "Execution failed at step 1" in result["final_output"]

    async def test_execute_plan_placeholder_resolution_fails(
        self, rewoo_processor: ReWOOCommandProcessorPlugin, mock_genie_for_rewoo
    ):
        """Test that a plan fails if placeholder resolution errors out."""
        processor = rewoo_processor
        await processor.setup({"genie_facade": mock_genie_for_rewoo})

        plan_model = ReWOOPlanForTest(
            plan=[
                ReWOOStepForTest(
                    tool_id="t1", params={"input": "{{outputs.non_existent.value}}"}
                )
            ]
        )
        result = await processor._execute_plan(plan_model, "goal", "corr_id")

        assert result["status"] == "error"
        # Corrected assertion to match actual error message structure
        assert "Error during step 1 preparation or validation" in result["final_output"]
        mock_genie_for_rewoo.execute_tool.assert_not_called()

    async def test_synthesize_answer_no_evidence(
        self, rewoo_processor: ReWOOCommandProcessorPlugin, mock_genie_for_rewoo
    ):
        """Test that the solver provides a graceful message with no evidence."""
        processor = rewoo_processor
        await processor.setup({"genie_facade": mock_genie_for_rewoo})

        mock_plan = ReWOOPlanForTest(plan=[])
        # Correctly mock the LLM chat response
        mock_genie_for_rewoo.llm.chat.return_value = {
            "message": {"content": "Synthesized based on errors: All plan steps failed."}
        }
        final_answer, _ = await processor._synthesize_answer(
            "goal", mock_plan, [], "corr_id"
        )
        assert "all plan steps failed" in final_answer.lower()

    async def test_synthesize_answer_llm_fails(
        self, rewoo_processor: ReWOOCommandProcessorPlugin, mock_genie_for_rewoo
    ):
        """Test that synthesis handles an LLM failure."""
        processor = rewoo_processor
        await processor.setup({"genie_facade": mock_genie_for_rewoo})

        mock_genie_for_rewoo.llm.chat.side_effect = Exception("Solver LLM error")
        mock_plan = ReWOOPlanForTest(plan=[ReWOOStepForTest(tool_id="t1")])
        evidence = [
            {
                "step": {"tool_id": "t1"},
                "error": None,
                "result": "res",
                "source_details": None,
                "detailed_summary_or_extraction": "res",
            }
        ]

        final_answer, _ = await processor._synthesize_answer(
            "goal", mock_plan, evidence, "corr_id"
        )
        assert "could not be synthesized" in final_answer
        assert "Solver LLM error" in final_answer

    async def test_process_command_integration(
        self, rewoo_processor: ReWOOCommandProcessorPlugin, mock_genie_for_rewoo
    ):
        """Test the main `process_command` orchestrator."""
        processor = rewoo_processor
        await processor.setup(
            {"genie_facade": mock_genie_for_rewoo, "min_high_quality_sources": 1}
        )

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
                        "error": None,  # Ensure error key is present
                        "source_details": {
                            "type": "t",
                            "identifier": "i",
                            "title": "T",
                        },
                    }
                ],
            }
        )
        processor._is_high_quality_evidence = AsyncMock(return_value=True)
        processor._synthesize_answer = AsyncMock(
            return_value=("Final Synthesized Answer", "raw_solver_output")
        )

        result = await processor.process_command("test command")
        assert result.get("final_answer") == "Final Synthesized Answer"
        processor._generate_plan.assert_awaited_once()  # This should now pass
        processor._execute_plan.assert_awaited_once()
        processor._synthesize_answer.assert_awaited_once()

    async def test_process_command_plan_generation_fails(
        self, rewoo_processor: ReWOOCommandProcessorPlugin, mock_genie_for_rewoo
    ):
        """Test the main `process_command` when planning fails completely."""
        processor = rewoo_processor
        await processor.setup({"genie_facade": mock_genie_for_rewoo})

        # Ensure list_tools returns something so the initial check passes
        mock_tool = MagicMock()
        mock_tool.identifier = "some_tool"
        mock_genie_for_rewoo._tool_manager.list_tools.return_value = [mock_tool]

        # Now mock the generate_plan method to fail
        processor._generate_plan = AsyncMock(
            return_value=(None, "raw planner failure output")
        )

        # Use patch to replace the real methods with mocks for this test
        with patch.object(
            processor, "_execute_plan", new_callable=AsyncMock
        ) as mock_execute_plan, patch.object(
            processor, "_synthesize_answer", new_callable=AsyncMock
        ) as mock_synthesize_answer:

            result = await processor.process_command("test command")

            assert "error" in result
            assert "Failed to generate a valid execution plan" in result["error"]
            assert (
                result.get("raw_response", {}).get("planner_llm_output")
                == "raw planner failure output"
            )

            mock_execute_plan.assert_not_called()
            mock_synthesize_answer.assert_not_called()

    async def test_generate_plan_pydantic_not_installed(
        self, rewoo_processor: ReWOOCommandProcessorPlugin, mock_genie_for_rewoo
    ):
        """Test that planning returns None if Pydantic is not available."""
        with patch(
            "genie_tooling.command_processors.impl.rewoo_processor.PYDANTIC_INSTALLED_FOR_REWOO_PROCESSOR",
            False,
        ):
            processor = rewoo_processor
            await processor.setup({"genie_facade": mock_genie_for_rewoo})

            plan, raw_output = await processor._generate_plan(
                "goal", "tools", ["t1"], "corr_id"
            )
            assert plan is None
            assert "cannot generate structured plan" in (raw_output or "")
