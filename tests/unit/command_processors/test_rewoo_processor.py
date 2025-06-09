### tests/unit/command_processors/impl/test_rewoo_processor.py
import json
from unittest.mock import ANY, AsyncMock, MagicMock
from typing import Any, List
import pytest
from genie_tooling.command_processors.impl.rewoo_processor import (
    ExecutionEvidence,
    ReWOOCommandProcessorPlugin,
    ReWOOPlan,
    ReWOOStep,
)
from genie_tooling.command_processors.types import CommandProcessorResponse
from genie_tooling.core.types import Plugin as CorePluginType
from genie_tooling.llm_providers.types import LLMChatResponse
from genie_tooling.tools.abc import Tool as ToolPlugin

# --- Mocks & Fixtures ---

class MockToolForReWOO(ToolPlugin, CorePluginType):
    """A simple mock Tool for testing."""
    def __init__(self, identifier: str, name: str, description: str):
        self._identifier = identifier
        self._name = name
        self._description = description

    @property
    def plugin_id(self) -> str:
        return self._identifier

    @property
    def identifier(self) -> str:
        return self._identifier

    async def get_metadata(self) -> dict:
        return {"identifier": self.identifier, "name": self._name, "description_llm": self._description}
    async def execute(self, params, key_provider, context=None) -> Any:
        return {"status": "success", "params_received": params}
    async def setup(self, config=None): pass
    async def teardown(self): pass


@pytest.fixture
def mock_genie_facade_for_rewoo(mocker) -> MagicMock:
    """A comprehensive mock of the Genie facade for the ReWOO processor."""
    genie = mocker.MagicMock(name="MockGenieFacadeForReWOO")

    # Mock sub-interfaces and their methods
    genie.prompts = AsyncMock(name="MockPromptInterface")
    genie.prompts.render_chat_prompt = AsyncMock(return_value=[{"role": "user", "content": "Planner prompt"}])
    genie.prompts.render_prompt = AsyncMock(return_value="Solver prompt")

    genie.llm = AsyncMock(name="MockLLMInterface")
    genie.llm.chat = AsyncMock(return_value=LLMChatResponse(message={"role": "assistant", "content": "{}"}))
    genie.llm.parse_output = AsyncMock()

    genie.observability = AsyncMock(name="MockObservabilityInterface")
    genie.observability.trace_event = AsyncMock()

    genie.execute_tool = AsyncMock(return_value={"tool_result": "success"})

    genie._tool_manager = AsyncMock(name="MockToolManager")
    genie._tool_manager.list_tools = AsyncMock(return_value=[])
    genie._tool_manager.get_formatted_tool_definition = AsyncMock(return_value="Formatted Tool Definition")

    return genie


@pytest.fixture
async def rewoo_processor(mock_genie_facade_for_rewoo: MagicMock) -> ReWOOCommandProcessorPlugin:
    """Provides an initialized ReWOOCommandProcessorPlugin."""
    processor = ReWOOCommandProcessorPlugin()
    await processor.setup({"genie_facade": mock_genie_facade_for_rewoo})
    return processor


# --- Test Cases ---

@pytest.mark.asyncio
class TestReWOOCommandProcessor:
    async def test_setup_success(self, rewoo_processor: ReWOOCommandProcessorPlugin, mock_genie_facade_for_rewoo: MagicMock):
        processor = await rewoo_processor
        assert processor._genie is mock_genie_facade_for_rewoo
        assert processor._tool_formatter_id == "compact_text_formatter_plugin_v1"
        assert processor._max_plan_retries == 1

    async def test_setup_no_genie_facade_raises_error(self):
        processor = ReWOOCommandProcessorPlugin()
        with pytest.raises(ValueError, match="requires a 'genie_facade' instance"):
            await processor.setup({})

    async def test_generate_plan_success(self, rewoo_processor: ReWOOCommandProcessorPlugin, mock_genie_facade_for_rewoo: MagicMock):
        processor = await rewoo_processor
        expected_plan = ReWOOPlan(plan=[ReWOOStep(thought="t", tool_id="get_file_line_count", params={"p": 1})])
        mock_genie_facade_for_rewoo.llm.parse_output.return_value = expected_plan

        plan, raw_output = await processor._generate_plan("goal", "tool defs", ["get_file_line_count"], "corr-id-1")

        assert plan is not None
        assert plan.plan[0].tool_id == "get_file_line_count"
        mock_genie_facade_for_rewoo.prompts.render_prompt.assert_awaited_once()
        mock_genie_facade_for_rewoo.llm.chat.assert_awaited_once()
        mock_genie_facade_for_rewoo.llm.parse_output.assert_awaited_once()

    async def test_generate_plan_fails_after_retries(self, rewoo_processor: ReWOOCommandProcessorPlugin, mock_genie_facade_for_rewoo: MagicMock):
        processor = await rewoo_processor
        processor._max_plan_retries = 1
        mock_genie_facade_for_rewoo.llm.parse_output.side_effect = ValueError("Parsing failed")

        plan, raw_output = await processor._generate_plan("goal", "tool defs", [], "corr-id-2")

        assert plan is None
        assert mock_genie_facade_for_rewoo.llm.parse_output.call_count == 2

    async def test_synthesize_answer_success(self, rewoo_processor: ReWOOCommandProcessorPlugin, mock_genie_facade_for_rewoo: MagicMock):
        processor = await rewoo_processor
        mock_genie_facade_for_rewoo.llm.chat.return_value = LLMChatResponse(message={"role": "assistant", "content": "Final synthesized answer."})
        plan = ReWOOPlan(plan=[])
        evidence: List[ExecutionEvidence] = []

        final_answer, raw_output = await processor._synthesize_answer("goal", plan, evidence, "corr-id-3")

        assert final_answer == "Final synthesized answer."
        mock_genie_facade_for_rewoo.prompts.render_prompt.assert_awaited_once()
        mock_genie_facade_for_rewoo.llm.chat.assert_awaited_once()

    async def test_process_command_end_to_end_success(self, rewoo_processor: ReWOOCommandProcessorPlugin, mock_genie_facade_for_rewoo: MagicMock):
        processor = await rewoo_processor
        plan_steps = [ReWOOStep(thought="t1", tool_id="tool_A", params={"p": "a"})]
        mock_plan = ReWOOPlan(plan=plan_steps)
        # Simulate the return from _generate_plan
        processor._generate_plan = AsyncMock(return_value=(mock_plan, '{"plan":...}'))
        mock_genie_facade_for_rewoo.execute_tool.return_value = {"result": "Tool A success"}
        # Simulate the return from _synthesize_answer
        processor._synthesize_answer = AsyncMock(return_value=("The final answer is success.", "Raw solver output"))

        response = await processor.process_command("Do the thing")

        assert isinstance(response, dict)
        assert response["final_answer"] == "The final answer is success."
        assert "plan" in response["llm_thought_process"]
        assert "evidence" in response["llm_thought_process"]
        mock_genie_facade_for_rewoo.execute_tool.assert_awaited_once_with("tool_A", p="a")
        assert "planner_llm_output" in response["raw_response"]
        assert "solver_llm_output" in response["raw_response"]

    async def test_process_command_plan_generation_fails(self, rewoo_processor: ReWOOCommandProcessorPlugin, mock_genie_facade_for_rewoo: MagicMock):
        processor = await rewoo_processor
        processor._generate_plan = AsyncMock(return_value=(None, "raw output on failure"))

        response = await processor.process_command("A goal")

        assert response["error"] == "Failed to generate a valid execution plan."
        assert response["raw_response"]["planner_llm_output"] == "raw output on failure"
        mock_genie_facade_for_rewoo.execute_tool.assert_not_called()

    async def test_process_command_tool_execution_fails(self, rewoo_processor: ReWOOCommandProcessorPlugin, mock_genie_facade_for_rewoo: MagicMock):
        processor = await rewoo_processor
        plan_steps = [
            ReWOOStep(thought="t1", tool_id="tool_A", params={}),
            ReWOOStep(thought="t2", tool_id="tool_B", params={}),
        ]
        mock_plan = ReWOOPlan(plan=plan_steps)
        processor._generate_plan = AsyncMock(return_value=(mock_plan, "{}"))
        mock_genie_facade_for_rewoo.execute_tool.side_effect = [Exception("Tool A crashed"), {"result": "Tool B success"}]
        processor._synthesize_answer = AsyncMock(return_value=("Answer based on partial data.", ""))

        response = await processor.process_command("Do two things")

        assert response["final_answer"] == "Answer based on partial data."
        assert mock_genie_facade_for_rewoo.execute_tool.call_count == 2
        thought_process_data = json.loads(response["llm_thought_process"])
        evidence = thought_process_data["evidence"]
        assert len(evidence) == 2
        assert "Error executing tool 'tool_A': Tool A crashed" in evidence[0]["error"]
        assert evidence[0]["result"] is None
        assert evidence[1]["error"] is None
        assert evidence[1]["result"] == {"result": "Tool B success"}
