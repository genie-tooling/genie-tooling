"""M8 — parallel tool calls in the native ReActAgent loop.

Modern providers (OpenAI, Anthropic, Gemini) can emit multiple
``tool_calls`` in a single assistant turn. The native loop must execute
each of them and feed each result back keyed to the right ``tool_call_id``
before the next round.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

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


@pytest.mark.asyncio
async def test_native_loop_handles_two_parallel_tool_calls_per_turn():
    """Single assistant message contains two tool_calls → both execute
    sequentially → both results sent back with correct tool_call_ids."""
    calc_tool = _mock_tool("calculator_tool")
    lookup_tool = _mock_tool("lookup_tool")

    genie = MagicMock()
    genie.tools = MagicMock()
    genie.tools.list = AsyncMock(return_value=[calc_tool, lookup_tool])
    genie.observability = MagicMock()
    genie.observability.trace_event = AsyncMock()
    genie.human_in_loop = MagicMock()

    # Each tool returns a distinct response so we can verify dispatch.
    async def execute(tool_id, context=None, **params):
        if tool_id == "calculator_tool":
            return {"result": 100}
        if tool_id == "lookup_tool":
            return {"value": "speed_of_light"}
        raise AssertionError(f"unexpected tool {tool_id}")
    genie.execute_tool = AsyncMock(side_effect=execute)

    # Turn 1: model asks for both tools in parallel.
    # Turn 2: model synthesises the final answer.
    turn_1 = {
        "message": {
            "role": "assistant",
            "content": "calling both tools in parallel",
            "tool_calls": [
                {
                    "id": "call_calc",
                    "type": "function",
                    "function": {
                        "name": "calculator_tool",
                        "arguments": json.dumps({"a": 50, "b": 50}),
                    },
                },
                {
                    "id": "call_lookup",
                    "type": "function",
                    "function": {
                        "name": "lookup_tool",
                        "arguments": json.dumps({"name": "speed_of_light"}),
                    },
                },
            ],
        }
    }
    turn_2 = {
        "message": {
            "role": "assistant",
            "content": "the answer is 100 and the constant is speed_of_light",
        }
    }
    genie.llm = MagicMock()
    genie.llm.chat = AsyncMock(side_effect=[turn_1, turn_2])

    agent = ReActAgent(
        genie=genie, agent_config={"use_native_tool_use": True, "max_iterations": 3}
    )
    result = await agent.run(goal="get both values")

    # Both tools were executed
    assert genie.execute_tool.await_count == 2
    executed_tools = {call.args[0] for call in genie.execute_tool.await_args_list}
    assert executed_tools == {"calculator_tool", "lookup_tool"}

    # The second LLM call received TWO tool messages, each keyed to its
    # originating tool_call_id (this is the contract OpenAI/Anthropic
    # require for parallel tool calls).
    second_call_msgs = genie.llm.chat.await_args_list[1].kwargs["messages"]
    tool_msgs = [m for m in second_call_msgs if m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    ids = {m["tool_call_id"] for m in tool_msgs}
    assert ids == {"call_calc", "call_lookup"}

    # Final answer surfaced to the caller
    assert result["status"] == "success"
    assert "100" in result["output"]
    assert "speed_of_light" in result["output"]


@pytest.mark.asyncio
async def test_native_loop_parallel_calls_preserve_order_of_execution():
    """Order matters for audit: the loop must execute tool_calls in the
    order the model emitted them."""
    a_tool = _mock_tool("tool_a")
    b_tool = _mock_tool("tool_b")

    execution_log = []

    async def execute(tool_id, context=None, **params):
        execution_log.append(tool_id)
        return {"ok": True, "from": tool_id}

    genie = MagicMock()
    genie.tools = MagicMock()
    genie.tools.list = AsyncMock(return_value=[a_tool, b_tool])
    genie.observability = MagicMock()
    genie.observability.trace_event = AsyncMock()
    genie.execute_tool = AsyncMock(side_effect=execute)
    genie.human_in_loop = MagicMock()

    turn_1 = {
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "1", "type": "function", "function": {"name": "tool_b", "arguments": "{}"}},
                {"id": "2", "type": "function", "function": {"name": "tool_a", "arguments": "{}"}},
                {"id": "3", "type": "function", "function": {"name": "tool_b", "arguments": "{}"}},
            ],
        }
    }
    turn_2 = {"message": {"role": "assistant", "content": "done"}}
    genie.llm = MagicMock()
    genie.llm.chat = AsyncMock(side_effect=[turn_1, turn_2])

    agent = ReActAgent(
        genie=genie, agent_config={"use_native_tool_use": True, "max_iterations": 3}
    )
    await agent.run(goal="ordered")
    assert execution_log == ["tool_b", "tool_a", "tool_b"]


@pytest.mark.asyncio
async def test_native_loop_partial_failure_in_parallel_does_not_short_circuit():
    """If one tool in a parallel batch fails, the others still execute
    and all results (success + error) are sent back. The agent decides
    what to do on the next turn."""
    a_tool = _mock_tool("good_tool")
    b_tool = _mock_tool("bad_tool")

    async def execute(tool_id, context=None, **params):
        if tool_id == "bad_tool":
            raise RuntimeError("kaboom")
        return {"ok": True}

    genie = MagicMock()
    genie.tools = MagicMock()
    genie.tools.list = AsyncMock(return_value=[a_tool, b_tool])
    genie.observability = MagicMock()
    genie.observability.trace_event = AsyncMock()
    genie.execute_tool = AsyncMock(side_effect=execute)
    genie.human_in_loop = MagicMock()

    turn_1 = {
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "1", "type": "function", "function": {"name": "good_tool", "arguments": "{}"}},
                {"id": "2", "type": "function", "function": {"name": "bad_tool", "arguments": "{}"}},
            ],
        }
    }
    turn_2 = {"message": {"role": "assistant", "content": "noted both"}}
    genie.llm = MagicMock()
    genie.llm.chat = AsyncMock(side_effect=[turn_1, turn_2])

    agent = ReActAgent(
        genie=genie, agent_config={"use_native_tool_use": True, "max_iterations": 3}
    )
    result = await agent.run(goal="parallel with one failure")

    # Both tools attempted
    assert genie.execute_tool.await_count == 2
    # Both tool messages reached the next turn
    tool_msgs = [
        m for m in genie.llm.chat.await_args_list[1].kwargs["messages"]
        if m.get("role") == "tool"
    ]
    assert len(tool_msgs) == 2
    # The failed call's tool message contains the error
    bad = next(m for m in tool_msgs if m["tool_call_id"] == "2")
    assert "kaboom" in bad["content"]
    # The successful one is JSON
    good = next(m for m in tool_msgs if m["tool_call_id"] == "1")
    assert "true" in good["content"].lower() or "ok" in good["content"]

    assert result["status"] == "success"
