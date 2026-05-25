"""F4 — ReActAgent persists state to a configured checkpointer between iterations
and supports resume_from_run_id to pick up where a crashed run left off."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from genie_tooling.agents.checkpointer.impl.in_memory import (
    InMemoryAgentCheckpointerPlugin,
)
from genie_tooling.agents.react_agent import ReActAgent


def _mock_tool(identifier: str):
    t = MagicMock()
    t.identifier = identifier
    t.get_metadata = AsyncMock(
        return_value={
            "identifier": identifier,
            "name": identifier,
            "description_llm": f"The {identifier} tool.",
            "input_schema": {"type": "object", "properties": {}},
        }
    )
    return t


async def _build_genie_with_checkpointer():
    cp = InMemoryAgentCheckpointerPlugin()
    await cp.setup()
    genie = MagicMock()
    genie.checkpointer = cp
    genie.tools = MagicMock()
    genie.tools.list = AsyncMock(return_value=[_mock_tool("calculator_tool")])
    genie.observability = MagicMock()
    genie.observability.trace_event = AsyncMock()
    genie.execute_tool = AsyncMock(return_value={"result": 42})
    genie.human_in_loop = MagicMock()
    return genie, cp


@pytest.mark.asyncio()
async def test_react_native_run_persists_checkpoint_per_iteration():
    """Each iteration should save a CheckpointState; final state has status="completed"."""
    genie, cp = await _build_genie_with_checkpointer()

    # 1st turn: tool call. 2nd turn: final answer.
    turn_1 = {
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "calculator_tool", "arguments": "{}"},
                }
            ],
        }
    }
    turn_2 = {"message": {"role": "assistant", "content": "42"}}
    genie.llm = MagicMock()
    genie.llm.chat = AsyncMock(side_effect=[turn_1, turn_2])

    agent = ReActAgent(
        genie=genie,
        agent_config={"use_native_tool_use": True, "max_iterations": 5},
    )
    result = await agent.run(goal="compute")
    assert result["status"] == "success"

    # Find the run that got saved
    runs = await cp.list_runs(agent_id="react_agent")
    assert len(runs) == 1
    final = await cp.load_checkpoint(runs[0].run_id)
    assert final is not None
    assert final.status == "completed"
    assert final.goal == "compute"
    assert isinstance(final.state_blob.get("scratchpad"), list)


@pytest.mark.asyncio()
async def test_react_resume_continues_from_saved_iteration():
    """If we manually save a partial state with run_id=X and then call run(resume_from_run_id=X),
    the agent picks up where the state left off."""
    genie, cp = await _build_genie_with_checkpointer()
    # No tools needed for this resume test — just one LLM turn that emits final answer.

    # Seed a "previous run" state.
    import time

    from genie_tooling.agents.checkpointer.types import CheckpointState

    prior = CheckpointState(
        run_id="prior-run-1",
        agent_id="react_agent",
        iteration=2,
        goal="continue this",
        state_blob={
            "scratchpad": [
                {"thought": "step 1", "action": "calc", "observation": "did calc"},
                {"thought": "step 2", "action": "lookup", "observation": "got value"},
            ],
            "mode": "react",
        },
        status="running",
        created_at=time.time() - 30,
        updated_at=time.time() - 10,
    )
    await cp.save_checkpoint(prior)

    # The model immediately answers.
    final_turn = {"message": {"role": "assistant", "content": "done"}}
    genie.llm = MagicMock()
    genie.llm.chat = AsyncMock(return_value=final_turn)

    agent = ReActAgent(
        genie=genie,
        agent_config={"use_native_tool_use": True, "max_iterations": 5},
    )
    result = await agent.run(goal="continue this", resume_from_run_id="prior-run-1")
    assert result["status"] == "success"

    after = await cp.load_checkpoint("prior-run-1")
    assert after.status == "completed"
    # The resumed run should preserve historic scratchpad (we don't trim it)
    # plus the new "Answer" observation.
    assert len(after.state_blob["scratchpad"]) >= 3


@pytest.mark.asyncio()
async def test_react_no_checkpointer_means_no_save_no_crash():
    """Genie without a checkpointer still runs cleanly."""
    genie = MagicMock()
    genie.checkpointer = None
    genie.tools = MagicMock()
    genie.tools.list = AsyncMock(return_value=[_mock_tool("calculator_tool")])
    genie.observability = MagicMock()
    genie.observability.trace_event = AsyncMock()
    genie.execute_tool = AsyncMock()
    genie.human_in_loop = MagicMock()

    final_turn = {"message": {"role": "assistant", "content": "hello"}}
    genie.llm = MagicMock()
    genie.llm.chat = AsyncMock(return_value=final_turn)

    agent = ReActAgent(
        genie=genie,
        agent_config={"use_native_tool_use": True, "max_iterations": 1},
    )
    result = await agent.run(goal="say hi")
    assert result["status"] == "success"
