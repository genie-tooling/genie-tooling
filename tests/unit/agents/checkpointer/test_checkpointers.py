"""Phase 6A.2 — agent checkpointer tests for both implementations."""
from __future__ import annotations

import time

import pytest
import pytest_asyncio
from genie_tooling.agents.checkpointer import (
    AgentCheckpointerPlugin,
    CheckpointState,
)
from genie_tooling.agents.checkpointer.impl.in_memory import (
    InMemoryAgentCheckpointerPlugin,
)
from genie_tooling.agents.checkpointer.impl.sqlite import (
    SQLiteAgentCheckpointerPlugin,
)


@pytest_asyncio.fixture(params=["in_memory", "sqlite"])
async def checkpointer(request, tmp_path) -> AgentCheckpointerPlugin:
    if request.param == "in_memory":
        cp = InMemoryAgentCheckpointerPlugin()
        await cp.setup()
    else:
        cp = SQLiteAgentCheckpointerPlugin()
        await cp.setup({"db_path": str(tmp_path / "test_cp.sqlite")})
    yield cp
    await cp.teardown()


def _state(run_id="r1", iteration=0, status="running", agent_id="react", **overrides):
    now = time.time()
    return CheckpointState(
        run_id=run_id,
        agent_id=agent_id,
        iteration=iteration,
        goal="test goal",
        state_blob={"scratchpad": [{"thought": "t", "action": "a", "observation": "o"}]},
        status=status,
        created_at=overrides.get("created_at", now),
        updated_at=overrides.get("updated_at", now),
        attribution_tags=overrides.get("attribution_tags") or {"team": "platform"},
        user_identity=overrides.get("user_identity"),
        correlation_id=overrides.get("correlation_id"),
        error=overrides.get("error"),
    )


@pytest.mark.asyncio()
async def test_save_and_load_roundtrip(checkpointer):
    await checkpointer.save_checkpoint(_state())
    loaded = await checkpointer.load_checkpoint("r1")
    assert loaded is not None
    assert loaded.run_id == "r1"
    assert loaded.agent_id == "react"
    assert loaded.iteration == 0
    assert loaded.state_blob["scratchpad"][0]["thought"] == "t"
    assert loaded.attribution_tags == {"team": "platform"}


@pytest.mark.asyncio()
async def test_load_missing_returns_none(checkpointer):
    assert await checkpointer.load_checkpoint("nonexistent") is None


@pytest.mark.asyncio()
async def test_upsert_on_same_run_id_replaces_state(checkpointer):
    await checkpointer.save_checkpoint(_state(iteration=0))
    await checkpointer.save_checkpoint(_state(iteration=5, status="running"))
    loaded = await checkpointer.load_checkpoint("r1")
    assert loaded.iteration == 5


@pytest.mark.asyncio()
async def test_list_runs_filters_by_agent_and_status(checkpointer):
    await checkpointer.save_checkpoint(_state(run_id="a1", agent_id="react", status="running"))
    await checkpointer.save_checkpoint(_state(run_id="a2", agent_id="react", status="completed"))
    await checkpointer.save_checkpoint(_state(run_id="b1", agent_id="planner", status="running"))

    react_runs = await checkpointer.list_runs(agent_id="react")
    assert {r.run_id for r in react_runs} == {"a1", "a2"}

    running = await checkpointer.list_runs(status="running")
    assert {r.run_id for r in running} == {"a1", "b1"}

    react_completed = await checkpointer.list_runs(agent_id="react", status="completed")
    assert {r.run_id for r in react_completed} == {"a2"}


@pytest.mark.asyncio()
async def test_list_runs_filters_by_attribution_tag(checkpointer):
    await checkpointer.save_checkpoint(
        _state(run_id="a1", attribution_tags={"team": "platform", "incident": "SEV2-1"})
    )
    await checkpointer.save_checkpoint(
        _state(run_id="a2", attribution_tags={"team": "platform", "incident": "SEV2-2"})
    )
    await checkpointer.save_checkpoint(
        _state(run_id="a3", attribution_tags={"team": "search"})
    )

    platform = await checkpointer.list_runs(attribution_tag={"team": "platform"})
    assert {r.run_id for r in platform} == {"a1", "a2"}

    incident_1 = await checkpointer.list_runs(attribution_tag={"incident": "SEV2-1"})
    assert {r.run_id for r in incident_1} == {"a1"}


@pytest.mark.asyncio()
async def test_delete_checkpoint(checkpointer):
    await checkpointer.save_checkpoint(_state())
    await checkpointer.delete_checkpoint("r1")
    assert await checkpointer.load_checkpoint("r1") is None


@pytest.mark.asyncio()
async def test_list_runs_orders_by_updated_at_desc(checkpointer):
    now = time.time()
    await checkpointer.save_checkpoint(
        _state(run_id="old", created_at=now - 100, updated_at=now - 100)
    )
    await checkpointer.save_checkpoint(
        _state(run_id="new", created_at=now, updated_at=now)
    )
    runs = await checkpointer.list_runs()
    assert [r.run_id for r in runs[:2]] == ["new", "old"]


@pytest.mark.asyncio()
async def test_state_blob_json_roundtrip_preserves_structure(checkpointer):
    """Complex state_blob with nested dicts/lists survives serialization."""
    complex_blob = {
        "scratchpad": [
            {"thought": "step 1", "action": "tool_a", "observation": {"result": [1, 2, 3]}},
            {"thought": "step 2", "action": "tool_b", "observation": None},
        ],
        "plan": [{"id": "s1", "tool": "x"}, {"id": "s2", "tool": "y"}],
        "outputs": {"s1": {"value": "ok"}, "s2": None},
    }
    s = _state()
    s.state_blob = complex_blob
    await checkpointer.save_checkpoint(s)
    loaded = await checkpointer.load_checkpoint("r1")
    assert loaded.state_blob == complex_blob
