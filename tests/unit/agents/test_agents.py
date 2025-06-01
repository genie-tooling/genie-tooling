### tests/unit/agents/test_agents.py
import logging
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock

import pytest

# from pydantic import BaseModel as PydanticBaseModel # Not needed directly in test file now
# from pydantic import Field as PydanticField       # Not needed directly in test file now
from genie_tooling.agents.base_agent import BaseAgent
from genie_tooling.agents.plan_and_execute_agent import (
    DEFAULT_PLANNER_SYSTEM_PROMPT_ID,
    PlanAndExecuteAgent,
)
from genie_tooling.agents.react_agent import (
    DEFAULT_REACT_MAX_ITERATIONS,
    DEFAULT_REACT_SYSTEM_PROMPT_ID,
    ReActAgent,
)

# Import Pydantic models from agents.types for mocking
from genie_tooling.agents.types import (
    PlanModelPydantic,
    PlannedStep,
    PlanStepModelPydantic,
)
from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig

# Forward declare Genie for type hinting in mocks
if False:  # TYPE_CHECKING
    pass

logger = logging.getLogger(__name__)



# --- Concrete BaseAgent for testing ---
class ConcreteBaseAgent(BaseAgent):
    async def run(self, goal: str, **kwargs: Any) -> Any:
        logger.info(f"ConcreteBaseAgent run called with goal: {goal}")
        return {"status": "concrete_run_executed", "goal": goal}


# --- Fixtures ---

@pytest.fixture
def mock_genie_config() -> MiddlewareConfig:
    """Provides a basic MiddlewareConfig for agent tests."""
    return MiddlewareConfig(
        features=FeatureSettings(
            llm="ollama",  # Default for agents if not specified
            llm_ollama_model_name="test_model_for_agents",
            hitl_approver="cli_hitl_approver",  # For PlanAndExecuteAgent HITL
            prompt_registry="file_system_prompt_registry",
            prompt_template_engine="jinja2_chat_formatter",
        )
    )

@pytest.fixture
def mock_genie(mocker, mock_genie_config: MiddlewareConfig) -> MagicMock:
    """Provides a MagicMock for the Genie facade."""
    genie_mock = MagicMock(name="MockGenieFacadeForAgents")

    genie_mock._config = mock_genie_config
    genie_mock._tool_manager = AsyncMock(name="MockToolManagerOnGenie")
    mock_tool_instance = MagicMock()
    mock_tool_instance.identifier = "default_mock_tool_id"
    genie_mock._tool_manager.list_tools = AsyncMock(return_value=[mock_tool_instance])
    genie_mock._tool_manager.get_formatted_tool_definition = AsyncMock(
        return_value="Formatted Tool Def"
    )

    genie_mock.observability = AsyncMock(name="MockObservabilityInterface")
    genie_mock.observability.trace_event = AsyncMock()

    genie_mock.prompts = AsyncMock(name="MockPromptInterface")
    genie_mock.prompts.render_chat_prompt = AsyncMock(
        return_value=[{"role": "system", "content": "Mock System Prompt"}]
    )

    genie_mock.llm = AsyncMock(name="MockLLMInterface")
    genie_mock.llm.chat = AsyncMock(
        return_value={
            "message": {"content": "LLM Response"},
            "usage": {"total_tokens": 10},
        }
    )

    # Use the imported Pydantic models for the default parsed plan
    default_parsed_plan = PlanModelPydantic(
        plan=[
            PlanStepModelPydantic(
                step_number=1, tool_id="tool1", params={"p": 1}, reasoning="Step 1 reason"
            )
        ],
        overall_reasoning="Overall plan reason",
    )
    genie_mock.llm.parse_output = AsyncMock(return_value=default_parsed_plan)

    genie_mock.execute_tool = AsyncMock(return_value={"result": "Tool Executed"})

    genie_mock.human_in_loop = AsyncMock(name="MockHITLInterface")
    genie_mock.human_in_loop.request_approval = AsyncMock(
        return_value={
            "status": "approved",
            "reason": None,
            "request_id": "mock_hitl_req_id",
        }
    )
    return genie_mock


# --- BaseAgent Tests ---

class TestBaseAgent:
    def test_base_agent_instantiation_success(self, mock_genie: MagicMock):
        agent_config = {"custom_param": "value"}
        agent = ConcreteBaseAgent(genie=mock_genie, agent_config=agent_config)
        assert agent.genie is mock_genie
        assert agent.agent_config == agent_config

    def test_base_agent_instantiation_no_config(self, mock_genie: MagicMock):
        agent = ConcreteBaseAgent(genie=mock_genie)
        assert agent.genie is mock_genie
        assert agent.agent_config == {}

    def test_base_agent_instantiation_no_genie_raises_error(self):
        with pytest.raises(ValueError, match="A Genie instance is required"):
            ConcreteBaseAgent(genie=None)  # type: ignore

    @pytest.mark.asyncio
    async def test_base_agent_teardown(self, mock_genie: MagicMock, caplog):
        caplog.set_level(logging.INFO)
        agent = ConcreteBaseAgent(genie=mock_genie)
        await agent.teardown()
        assert f"{agent.__class__.__name__} teardown initiated." in caplog.text


# --- ReActAgent Tests ---
class TestReActAgent:
    def test_react_agent_instantiation_defaults(self, mock_genie: MagicMock):
        agent = ReActAgent(genie=mock_genie)
        assert agent.max_iterations == DEFAULT_REACT_MAX_ITERATIONS
        assert agent.system_prompt_id == DEFAULT_REACT_SYSTEM_PROMPT_ID
        assert agent.llm_provider_id is None
        assert agent.tool_formatter_id == "compact_text_formatter_plugin_v1"
        assert agent.stop_sequences == ["Observation:"]
        assert agent.llm_retry_attempts == 1
        assert agent.llm_retry_delay == 2.0

    def test_react_agent_instantiation_custom_config(self, mock_genie: MagicMock):
        agent_config = {
            "max_iterations": 10,
            "system_prompt_id": "custom_react_prompt",
            "llm_provider_id": "custom_llm",
            "tool_formatter_id": "custom_formatter",
            "stop_sequences": ["Obs:"],
            "llm_retry_attempts": 2,
            "llm_retry_delay_seconds": 5.0,
        }
        agent = ReActAgent(genie=mock_genie, agent_config=agent_config)
        assert agent.max_iterations == 10
        assert agent.system_prompt_id == "custom_react_prompt"
        assert agent.llm_provider_id == "custom_llm"
        assert agent.tool_formatter_id == "custom_formatter"
        assert agent.stop_sequences == ["Obs:"]
        assert agent.llm_retry_attempts == 2
        assert agent.llm_retry_delay == 5.0

    @pytest.mark.parametrize(
        "llm_output, expected_thought, expected_action_str, expected_answer",
        [
            (
                'Thought: I need to use a tool.\nAction: MyTool[{"param": "value"}]',
                "I need to use a tool.",
                'MyTool[{"param": "value"}]',
                None,
            ),
            (
                "Thought: I have the answer.\nAnswer: The final answer is 42.",
                "I have the answer.",
                None,
                "The final answer is 42.",
            ),
            ("Thought: Just thinking.", "Just thinking.", None, None),
            ( # If only action is present, thought remains None, action is parsed
                "Action: AnotherTool[]",
                None,
                "AnotherTool[{}]",
                None,
            ),
            ( # If only answer is present, thought remains None, answer is parsed
                "Answer: Only an answer.",
                None,
                None,
                "Only an answer.",
            ),
            (
                "Thought: Malformed action.\nAction: BadTool[invalid_json", # Params not wrapped in {}, no closing ]
                "Malformed action.",
                None,
                None,
            ),
            (
                "Thought: Empty params.\nAction: EmptyParamTool[]",
                "Empty params.",
                "EmptyParamTool[{}]",
                None,
            ),
            (
                "Thought: Params no braces.\nAction: NoBraceTool[key:val]", # Params not wrapped in {}
                "Params no braces.",
                "NoBraceTool[{key:val}]", # Parser attempts to wrap
                None,
            ),
            ( # Fallback: entire output becomes thought
                "Completely unparseable output.",
                "Completely unparseable output.",
                None,
                None,
            ),
            (
                'thought: case test.\naction: CaseTool[{"p":1}]',
                "case test.",
                'CaseTool[{"p":1}]',
                None,
            ),
            ( # Action without [], so not a valid tool action format
                "Thought: Action with no params.\nAction: NoParamTool",
                "Action with no params.",
                None,
                None,
            ),
            (
                'Thought: Action with trailing text.\nAction: TrailTool[{"p":1}]\nSome other text.',
                "Action with trailing text.",
                'TrailTool[{"p":1}]',
                None,
            ),
        ],
    )
    def test_parse_llm_reason_act_output(
        self,
        mock_genie: MagicMock,
        llm_output,
        expected_thought,
        expected_action_str,
        expected_answer,
    ):
        agent = ReActAgent(genie=mock_genie)
        thought, action_str, final_answer = agent._parse_llm_reason_act_output(
            llm_output
        )
        assert thought == expected_thought
        assert action_str == expected_action_str
        assert final_answer == expected_answer

    @pytest.mark.asyncio
    async def test_run_successful_with_final_answer(self, mock_genie: MagicMock):
        agent = ReActAgent(genie=mock_genie)
        mock_genie.llm.chat.return_value = {
            "message": {"content": "Thought: I know the answer.\nAnswer: The answer is 42."}
        }
        result = await agent.run(goal="What is the meaning of life?")

        assert result["status"] == "success"
        assert result["output"] == "The answer is 42."
        assert len(result["history"]) == 1
        assert result["history"][0]["thought"] == "I know the answer."
        mock_genie.prompts.render_chat_prompt.assert_awaited_once()
        mock_genie.llm.chat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_max_iterations_reached(self, mock_genie: MagicMock):
        agent = ReActAgent(genie=mock_genie, agent_config={"max_iterations": 1})
        mock_genie.llm.chat.return_value = {
            "message": {"content": 'Thought: I need a tool.\nAction: SomeTool[{"q":"query"}]'}
        }
        mock_genie.execute_tool.return_value = {"tool_output": "some observation"}
        # Ensure SomeTool is in the list of tools for the agent
        mock_tool_instance = MagicMock()
        mock_tool_instance.identifier = "SomeTool"
        mock_genie._tool_manager.list_tools.return_value = [mock_tool_instance]


        result = await agent.run(goal="A complex task")

        assert result["status"] == "max_iterations_reached"
        assert "Max iterations (1) reached." in result["output"]  # type: ignore
        assert len(result["history"]) == 1  # One iteration completed
        assert mock_genie.llm.chat.call_count == 1
        mock_genie.execute_tool.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_llm_fails_all_retries(self, mock_genie: MagicMock):
        agent = ReActAgent(
            genie=mock_genie,
            agent_config={"llm_retry_attempts": 1, "llm_retry_delay_seconds": 0.01},
        )
        mock_genie.llm.chat.side_effect = RuntimeError("LLM is down")
        result = await agent.run(goal="Test LLM failure")

        assert result["status"] == "error"
        assert "LLM failed after retries: LLM is down" in result["output"]  # type: ignore
        assert mock_genie.llm.chat.call_count == 2  # 1 initial + 1 retry

    @pytest.mark.asyncio
    async def test_run_tool_execution_fails(self, mock_genie: MagicMock):
        agent = ReActAgent(genie=mock_genie)
        mock_genie.llm.chat.return_value = {
            "message": {"content": 'Thought: Use tool.\nAction: ErrorTool[{"p":1}]'}
        }
        mock_genie.execute_tool.side_effect = Exception("Tool crashed")
        # Ensure ErrorTool is in the list of tools for the agent
        mock_tool_instance = MagicMock()
        mock_tool_instance.identifier = "ErrorTool"
        mock_genie._tool_manager.list_tools.return_value = [mock_tool_instance]


        result = await agent.run(goal="Test tool failure")

        assert result["status"] == "max_iterations_reached" # It will retry until max_iterations
        assert len(result["history"]) > 0
        assert "Error executing tool 'ErrorTool': Tool crashed" in result["history"][-1][
            "observation"
        ]  # type: ignore
        mock_genie.execute_tool.assert_called()


# --- PlanAndExecuteAgent Tests ---
class TestPlanAndExecuteAgent:
    def test_plan_and_execute_agent_instantiation_defaults(
        self, mock_genie: MagicMock
    ):
        agent = PlanAndExecuteAgent(genie=mock_genie)
        assert agent.planner_system_prompt_id == DEFAULT_PLANNER_SYSTEM_PROMPT_ID
        assert agent.planner_llm_provider_id is None
        assert agent.tool_formatter_id == "compact_text_formatter_plugin_v1"
        assert agent.max_plan_retries == 1
        assert agent.max_step_retries == 0
        assert agent.replan_on_step_failure is False

    def test_plan_and_execute_agent_instantiation_custom_config(
        self, mock_genie: MagicMock
    ):
        agent_config = {
            "planner_system_prompt_id": "custom_planner",
            "planner_llm_provider_id": "custom_llm",
            "tool_formatter_id": "custom_formatter",
            "max_plan_retries": 2,
            "max_step_retries": 1,
            "replan_on_step_failure": True,
        }
        agent = PlanAndExecuteAgent(genie=mock_genie, agent_config=agent_config)
        assert agent.planner_system_prompt_id == "custom_planner"
        assert agent.planner_llm_provider_id == "custom_llm"
        assert agent.tool_formatter_id == "custom_formatter"
        assert agent.max_plan_retries == 2
        assert agent.max_step_retries == 1
        assert agent.replan_on_step_failure is True

    @pytest.mark.asyncio
    async def test_generate_plan_success(self, mock_genie: MagicMock):
        agent = PlanAndExecuteAgent(genie=mock_genie)
        # The mock_genie.llm.parse_output is already set up to return an instance
        # of PlanModelPydantic (via the default_parsed_plan in the fixture).
        plan = await agent._generate_plan(goal="Test plan generation")

        assert plan is not None
        assert len(plan) == 1
        assert plan[0]["tool_id"] == "tool1"
        mock_genie.prompts.render_chat_prompt.assert_awaited_once()
        mock_genie.llm.chat.assert_awaited_once()
        mock_genie.llm.parse_output.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_generate_plan_llm_parse_fails(self, mock_genie: MagicMock):
        agent = PlanAndExecuteAgent(
            genie=mock_genie, agent_config={"max_plan_retries": 0}
        )
        mock_genie.llm.parse_output.return_value = None  # Simulate parsing failure
        plan = await agent._generate_plan(goal="Test plan parse fail")

        assert plan is None
        mock_genie.llm.parse_output.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_plan_success(self, mock_genie: MagicMock):
        agent = PlanAndExecuteAgent(genie=mock_genie)
        plan: List[PlannedStep] = [
            {
                "step_number": 1,
                "tool_id": "tool_A",
                "params": {"p1": "v1"},
                "reasoning": "Step A",
            },
            {
                "step_number": 2,
                "tool_id": "tool_B",
                "params": {"p2": "v2"},
                "reasoning": "Step B",
            },
        ]
        mock_genie.execute_tool.side_effect = [
            {"result_A": "done"},
            {"result_B": "done"},
        ]
        result = await agent._execute_plan(plan, goal="Test execute success")

        assert result["status"] == "success"
        assert result["output"] == {"result_B": "done"}
        assert len(result["history"]) == 2  # type: ignore
        assert result["history"][0]["output"] == {"result_A": "done"}  # type: ignore
        assert mock_genie.execute_tool.call_count == 2
        mock_genie.human_in_loop.request_approval.assert_called()

    @pytest.mark.asyncio
    async def test_execute_plan_step_fails_no_replan(self, mock_genie: MagicMock):
        agent = PlanAndExecuteAgent(
            genie=mock_genie, agent_config={"replan_on_step_failure": False}
        )
        plan: List[PlannedStep] = [
            {"step_number": 1, "tool_id": "fail_tool", "params": {}, "reasoning": "Fail step"}
        ]
        mock_genie.execute_tool.side_effect = RuntimeError("Tool execution failed")
        result = await agent._execute_plan(plan, goal="Test step fail no replan")

        assert result["status"] == "error"
        assert "Execution failed at step 1: Error executing tool 'fail_tool'" in result[
            "output"
        ]  # type: ignore
        mock_genie.execute_tool.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_plan_step_fails_with_replan_success(
        self, mock_genie: MagicMock
    ):
        agent = PlanAndExecuteAgent(
            genie=mock_genie, agent_config={"replan_on_step_failure": True}
        )
        original_plan: List[PlannedStep] = [
            {"step_number": 1, "tool_id": "initial_tool", "params": {}, "reasoning": "Initial"}
        ]
        # Use the Pydantic models for the replan mock
        new_successful_plan_pydantic = PlanModelPydantic(
            plan=[PlanStepModelPydantic(step_number=1, tool_id="replan_tool", params={"rp":1}, reasoning="Replanned")],
            overall_reasoning="Replanned successfully"
        )
        # The agent's _generate_plan converts this to List[PlannedStep]
        new_successful_plan_typeddict: List[PlannedStep] = [
            {"step_number": 1, "tool_id": "replan_tool", "params": {"rp": 1}, "reasoning": "Replanned"}
        ]

        mock_genie.execute_tool.side_effect = [
            RuntimeError("Initial tool failed"),
            {"replan_result": "success"},
        ]
        # Mock _generate_plan to return the List[PlannedStep] structure
        agent._generate_plan = AsyncMock(return_value=new_successful_plan_typeddict) # type: ignore

        result = await agent._execute_plan(original_plan, goal="Test replan success")

        assert result["status"] == "success"
        assert result["output"] == {"replan_result": "success"}
        agent._generate_plan.assert_awaited_once()
        assert mock_genie.execute_tool.call_count == 2

    @pytest.mark.asyncio
    async def test_run_overall_success(self, mock_genie: MagicMock):
        agent = PlanAndExecuteAgent(genie=mock_genie)
        # Ensure _generate_plan is mocked to return a valid plan for the run method
        # The fixture already sets up mock_genie.llm.parse_output to return a PlanModelPydantic instance.
        # agent._generate_plan will use this.
        result = await agent.run(goal="Achieve this goal")

        assert result["status"] == "success" # CORRECTED
        assert result["output"] == {"result": "Tool Executed"} # From the default_parsed_plan's single step
        mock_genie.llm.parse_output.assert_called()
        mock_genie.execute_tool.assert_called()

    @pytest.mark.asyncio
    async def test_run_initial_plan_generation_fails(self, mock_genie: MagicMock):
        agent = PlanAndExecuteAgent(genie=mock_genie)
        mock_genie.llm.parse_output.return_value = None # This will cause _generate_plan to return None
        result = await agent.run(goal="Test initial plan fail")

        assert result["status"] == "error"
        assert "Failed to generate a plan" in result["output"]  # type: ignore
        mock_genie.execute_tool.assert_not_called()
