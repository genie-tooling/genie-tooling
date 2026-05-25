"""Unit tests for ReActAgent's per-action HITL gate (Phase 4 — B3).

Verifies:
  * When hitl_per_action=False (default), tools execute without approval.
  * When hitl_per_action=True, every tool call requests approval first.
  * A "denied" approval becomes an observation in the scratchpad; the agent
    sees the denial and continues reasoning rather than executing the tool.
  * A "approved" approval lets the tool execute as normal.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from genie_tooling.agents.react_agent import ReActAgent


def _mock_tool(identifier: str):
    t = MagicMock()
    t.identifier = identifier
    return t


def _mock_genie_with_tools(*tool_ids: str):
    g = MagicMock()
    g.tools = MagicMock()
    g.tools.list = AsyncMock(return_value=[_mock_tool(tid) for tid in tool_ids])
    g.tools.get_definition = AsyncMock(return_value="tool def text")
    g.observability = MagicMock()
    g.observability.trace_event = AsyncMock()

    async def render_prompt(name=None, template_content=None, **kw):
        return "rendered prompt"
    g.prompts = MagicMock()
    g.prompts.render_prompt = render_prompt

    g.execute_tool = AsyncMock(return_value={"result": 42})
    g.human_in_loop = MagicMock()
    g.human_in_loop.request_approval = AsyncMock()
    return g


@pytest.mark.asyncio
async def test_no_gate_when_hitl_per_action_disabled():
    """Default: agent calls execute_tool directly without ever touching HITL."""
    genie = _mock_genie_with_tools("calculator_tool")
    # The agent must produce an Action then an Answer; we orchestrate the
    # LLM to do exactly that in two turns.
    responses = [
        # Turn 1: pick a tool
        {"message": {"content": "Thought: try calculator\nAction: calculator_tool[{}]"}},
        # Turn 2: terminate
        {"message": {"content": "Thought: done\nAnswer: 42"}},
    ]
    genie.llm = MagicMock()
    genie.llm.chat = AsyncMock(side_effect=responses)

    agent = ReActAgent(genie=genie, agent_config={"max_iterations": 3})
    result = await agent.run(goal="test")
    assert result["status"] == "success"
    # HITL approver should NEVER have been called.
    genie.human_in_loop.request_approval.assert_not_awaited()
    # Tool DID execute
    genie.execute_tool.assert_awaited()


@pytest.mark.asyncio
async def test_gate_calls_approver_with_tool_details():
    """When hitl_per_action=True, the approval request payload carries
    enough detail for the approver to make an informed decision."""
    genie = _mock_genie_with_tools("calculator_tool")
    responses = [
        {"message": {"content": "Thought: math\nAction: calculator_tool[{\"x\": 5}]"}},
        {"message": {"content": "Thought: done\nAnswer: 42"}},
    ]
    genie.llm = MagicMock()
    genie.llm.chat = AsyncMock(side_effect=responses)
    genie.human_in_loop.request_approval = AsyncMock(
        return_value={"status": "approved", "approver_id": "auto", "reason": "ok"}
    )

    agent = ReActAgent(
        genie=genie,
        agent_config={"max_iterations": 3, "hitl_per_action": True},
    )
    result = await agent.run(goal="test")
    assert result["status"] == "success"
    genie.human_in_loop.request_approval.assert_awaited()
    req = genie.human_in_loop.request_approval.await_args.args[0]
    # The approval request carries actionable detail
    assert req["data_to_approve"]["tool_id"] == "calculator_tool"
    assert req["data_to_approve"]["params"] == {"x": 5}
    assert "iteration" in req["data_to_approve"]
    assert "calculator_tool" in req["prompt"]


@pytest.mark.asyncio
async def test_denial_becomes_observation_and_agent_continues():
    """A denied approval must NOT execute the tool. The denial must appear
    in the scratchpad so the LLM can reason about it on the next turn."""
    genie = _mock_genie_with_tools("calculator_tool")
    responses = [
        # Turn 1: agent tries to call the tool, gets denied
        {"message": {"content": "Thought: try math\nAction: calculator_tool[{}]"}},
        # Turn 2: agent observes the denial and gives up
        {"message": {"content": "Thought: denied, give up\nAnswer: I cannot proceed"}},
    ]
    genie.llm = MagicMock()
    genie.llm.chat = AsyncMock(side_effect=responses)
    genie.human_in_loop.request_approval = AsyncMock(
        return_value={
            "status": "denied",
            "approver_id": "policy_v1",
            "reason": "tool requires elevated role",
        }
    )

    agent = ReActAgent(
        genie=genie,
        agent_config={"max_iterations": 3, "hitl_per_action": True},
    )
    result = await agent.run(goal="test")
    assert result["status"] == "success"
    # Tool MUST NOT have executed when the approval was denied.
    genie.execute_tool.assert_not_awaited()
    # The scratchpad records the denial as an observation.
    history = result.get("history") or []
    history_str = " ".join(str(h) for h in history).lower()
    assert "denied" in history_str
    assert "policy_v1" in history_str
    assert "tool requires elevated role" in history_str


@pytest.mark.asyncio
async def test_denial_emits_react_tool_hitl_denied_trace_event():
    """Audit signal: a denied tool call emits a specific trace event so
    forensics can find every denied attempt."""
    genie = _mock_genie_with_tools("calculator_tool")
    responses = [
        {"message": {"content": "Thought: math\nAction: calculator_tool[{}]"}},
        {"message": {"content": "Thought: stop\nAnswer: done"}},
    ]
    genie.llm = MagicMock()
    genie.llm.chat = AsyncMock(side_effect=responses)
    genie.human_in_loop.request_approval = AsyncMock(
        return_value={"status": "denied", "approver_id": "ap", "reason": "no"}
    )

    agent = ReActAgent(
        genie=genie,
        agent_config={"max_iterations": 3, "hitl_per_action": True},
    )
    await agent.run(goal="test")
    denied_events = [
        call
        for call in genie.observability.trace_event.await_args_list
        if call.args and call.args[0] == "react_agent.tool.hitl_denied"
    ]
    assert denied_events, "expected react_agent.tool.hitl_denied trace event"
    payload = denied_events[0].args[1]
    assert payload["tool_id"] == "calculator_tool"
    assert payload["approver_id"] == "ap"
    assert payload["reason"] == "no"
