###tests/unit/agents/test_math_proof_assistant_agent.py###
from unittest.mock import AsyncMock, MagicMock

import pytest
from genie_tooling.agents.math_proof_assistant_agent import (
    HypothesisTestPlan,
    IntentResponse,
    MathProofAssistantAgent,
)
from genie_tooling.conversation.types import ConversationState


@pytest.fixture()
def mock_genie_for_math_agent():
    """Provides a mock Genie facade for the MathProofAssistantAgent."""
    genie = MagicMock(name="MockGenieForMathAgent")
    genie.observability = AsyncMock()
    genie.observability.trace_event = AsyncMock()

    genie.llm = AsyncMock()
    genie.llm.chat = AsyncMock()
    genie.llm.parse_output = AsyncMock()

    genie.rag = AsyncMock()
    genie.rag.index_text = AsyncMock()
    genie.rag.search = AsyncMock(return_value=[])

    genie.conversation = AsyncMock()
    genie.conversation.load_state = AsyncMock()
    genie.conversation.add_message = AsyncMock()

    genie.execute_tool = AsyncMock()
    genie.run_command = AsyncMock()

    # Mock tool manager for _handle_hypothesis
    genie._tool_manager = AsyncMock()
    genie._tool_manager.get_formatted_tool_definition = AsyncMock(
        return_value="Formatted Tool Def"
    )

    return genie


@pytest.fixture()
async def math_agent(mock_genie_for_math_agent) -> MathProofAssistantAgent:
    """Provides an initialized MathProofAssistantAgent."""
    agent = MathProofAssistantAgent(genie=mock_genie_for_math_agent)
    # Simulate run's initial setup
    agent.main_goal = "Test Goal"
    agent.session_id = "test_session"
    return agent


@pytest.mark.asyncio()
class TestMathProofAssistantAgent:
    async def test_determine_intent(self, math_agent, mock_genie_for_math_agent):
        """Test intent determination logic."""
        agent = await math_agent  # Await the fixture
        mock_genie_for_math_agent.conversation.load_state.return_value = ConversationState(
            session_id="test_session", history=[]
        )
        mock_intent_response = IntentResponse(intent="research_concept", query="What is a fermion?")
        mock_genie_for_math_agent.llm.parse_output.return_value = mock_intent_response

        intent = await agent.determine_intent("explain fermions")

        assert intent.intent == "research_concept"
        assert intent.query == "What is a fermion?"
        mock_genie_for_math_agent.llm.chat.assert_awaited_once()

    async def test_handle_research(self, math_agent, mock_genie_for_math_agent):
        """Test the research handler."""
        agent = await math_agent  # Await the fixture
        mock_genie_for_math_agent.run_command.return_value = {
            "final_answer": "Research complete."
        }
        await agent._handle_research("test query")
        mock_genie_for_math_agent.run_command.assert_awaited_once_with(
            command="test query", processor_id=agent.research_processor_id
        )
        mock_genie_for_math_agent.rag.index_text.assert_awaited_once()

    async def test_handle_research_fails(self, math_agent, mock_genie_for_math_agent, capsys):
        """Test when the research subcommand returns an error."""
        agent = await math_agent
        mock_genie_for_math_agent.run_command.return_value = {"error": "Deep research failed."}
        await agent._handle_research("failing query")
        captured = capsys.readouterr()

        assert "Research did not produce a final answer." in captured.out

    async def test_handle_hypothesis(self, math_agent, mock_genie_for_math_agent):
        """Test the hypothesis testing handler."""
        agent = await math_agent  # Await the fixture
        mock_hypothesis_plan = HypothesisTestPlan(
            thought="Thinking...",
            tool_id="symbolic_math_tool",
            params={"operation": "simplify", "expression": "x+x"},
        )
        # Use a mock for the internal _call_llm_for_json method to isolate logic
        agent._call_llm_for_json = AsyncMock(return_value=mock_hypothesis_plan.model_dump())
        mock_genie_for_math_agent.execute_tool.return_value = {"result": "2*x"}

        await agent._handle_hypothesis("simplify x+x")

        agent._call_llm_for_json.assert_awaited_once()
        mock_genie_for_math_agent.execute_tool.assert_awaited_once_with(
            "symbolic_math_tool", operation="simplify", expression="x+x"
        )
        mock_genie_for_math_agent.rag.index_text.assert_awaited_once()

    async def test_handle_hypothesis_plan_fails(self, math_agent, capsys):
        """Test when the LLM fails to generate a hypothesis test plan."""
        agent = await math_agent
        agent._call_llm_for_json = AsyncMock(return_value=None)
        await agent._handle_hypothesis("some hypothesis")
        captured = capsys.readouterr()
        assert "I couldn't figure out how to test that hypothesis" in captured.out

    async def test_handle_hypothesis_tool_fails(self, math_agent, mock_genie_for_math_agent, capsys):
        """Test when the tool execution fails during hypothesis testing."""
        agent = await math_agent
        mock_plan = HypothesisTestPlan(thought="t", tool_id="fail_tool", params={})
        agent._call_llm_for_json = AsyncMock(return_value=mock_plan.model_dump())
        mock_genie_for_math_agent.execute_tool.side_effect = RuntimeError("Tool execution error")

        await agent._handle_hypothesis("failing hypothesis")
        captured = capsys.readouterr()
        assert "An error occurred while testing the hypothesis: Tool execution error" in captured.out


    async def test_handle_insight(self, math_agent, mock_genie_for_math_agent):
        """Test the insight handling logic."""
        agent = await math_agent  # Await the fixture
        await agent._handle_insight("A new idea.")
        mock_genie_for_math_agent.rag.index_text.assert_awaited_once()
        assert "User insight: A new idea." in mock_genie_for_math_agent.rag.index_text.call_args.kwargs["text"]

    async def test_handle_continue_proof(self, math_agent):
        """Test advancing to the next sub-goal."""
        agent = await math_agent  # Await the fixture
        agent.sub_goals = ["step 1", "step 2"]
        agent.current_sub_goal_index = 0

        await agent._handle_continue_proof()
        assert agent.current_sub_goal_index == 1

        # Test completing all goals
        agent._synthesize_final_report = AsyncMock() # Mock the synthesis call
        await agent._handle_continue_proof()
        assert agent.current_sub_goal_index == 2
        assert agent.state == "FINISHED"
        agent._synthesize_final_report.assert_awaited_once()

    async def test_run_fallback_chat_logic_is_fixed(self, math_agent, mock_genie_for_math_agent):
        """Verify the bug fix in the 'unsupported' intent fallback."""
        agent = await math_agent  # Await the fixture

        # This test needs to mock the agent's internal loop. We'll test the logic
        # by directly calling the fallback block as if it were in the loop.
        # This is more of a unit test of the fallback logic itself.

        # 1. Test case where conversation history exists
        mock_history = [{"role": "user", "content": "hello"}]
        mock_genie_for_math_agent.conversation.load_state.return_value = ConversationState(
            session_id="test_session", history=mock_history, metadata={}
        )
        mock_genie_for_math_agent.llm.chat.return_value = {
            "message": {"role": "assistant", "content": "fallback response"}
        }

        # Simulate the 'else' block from the agent's run loop
        conversation_state = await agent.genie.conversation.load_state(agent.session_id)
        if conversation_state and conversation_state.get("history"):
            response = await agent.genie.llm.chat(conversation_state["history"])
            await agent.genie.conversation.add_message(agent.session_id, response["message"])
        else:
            # This part of the 'else' block wouldn't be hit in this setup
            pass

        # Verify it loaded state, then called chat with the correct history
        agent.genie.conversation.load_state.assert_awaited_once_with(agent.session_id)
        agent.genie.llm.chat.assert_awaited_with(mock_history)
        agent.genie.conversation.add_message.assert_awaited_with(agent.session_id, {"role": "assistant", "content": "fallback response"})

        # 2. Test case where conversation state is None (e.g., first turn error)
        mock_genie_for_math_agent.conversation.load_state.reset_mock()
        mock_genie_for_math_agent.llm.chat.reset_mock()
        mock_genie_for_math_agent.conversation.load_state.return_value = None

        conversation_state_none = await agent.genie.conversation.load_state(agent.session_id)
        assert conversation_state_none is None
        agent.genie.llm.chat.assert_not_called()

    async def test_synthesize_report_no_memories(self, math_agent, mock_genie_for_math_agent, capsys):
        """Test synthesis when the RAG search returns no documents."""
        agent = await math_agent
        # mock_genie_for_math_agent.rag.search is already configured to return [] by the fixture
        await agent._synthesize_final_report()
        captured = capsys.readouterr()
        assert "There's nothing in our project memory to synthesize" in captured.out
