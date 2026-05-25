"""F3 — verify ReActAgent threads attribution_tags + budget_scope through
to LLM calls and to genie.execute_tool's context."""
from __future__ import annotations

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


@pytest.mark.asyncio()
async def test_attribution_tags_and_budget_scope_reach_llm_chat():
    """Native loop: caller passes attribution_tags + budget_scope via input_context;
    they must show up as kwargs on every genie.llm.chat call."""
    tool = _mock_tool("calculator_tool")

    genie = MagicMock()
    genie.tools = MagicMock()
    genie.tools.list = AsyncMock(return_value=[tool])
    genie.observability = MagicMock()
    genie.observability.trace_event = AsyncMock()
    genie.execute_tool = AsyncMock(return_value={"result": 42})
    genie.human_in_loop = MagicMock()

    # 1st turn: model emits a tool call; 2nd turn: final answer.
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
        agent_config={"use_native_tool_use": True, "max_iterations": 3},
    )
    await agent.run(
        goal="compute it",
        input_context={
            "attribution_tags": {"team": "platform", "incident": "SEV2-9"},
            "budget_scope": "incident:SEV2-9",
            "user_identity": {"sub": "alice"},
        },
    )

    # Both LLM calls saw the framework-level kwargs
    for call in genie.llm.chat.await_args_list:
        kw = call.kwargs
        assert kw["attribution_tags"] == {"team": "platform", "incident": "SEV2-9"}
        assert kw["budget_scope"] == "incident:SEV2-9"

    # The tool call context propagated them
    tool_call = genie.execute_tool.await_args_list[0]
    ctx = tool_call.kwargs["context"]
    assert ctx["attribution_tags"] == {"team": "platform", "incident": "SEV2-9"}
    assert ctx["budget_scope"] == "incident:SEV2-9"
    assert ctx["user_identity"] == {"sub": "alice"}
    assert ctx["caller_chain"] == ["ReActAgent.native_tool_use"]


@pytest.mark.asyncio()
async def test_no_attribution_means_no_kwargs_passed():
    """When caller doesn't supply attribution, the agent shouldn't synthesise
    fake values — the kwargs simply aren't there."""
    tool = _mock_tool("noop_tool")

    genie = MagicMock()
    genie.tools = MagicMock()
    genie.tools.list = AsyncMock(return_value=[tool])
    genie.observability = MagicMock()
    genie.observability.trace_event = AsyncMock()
    genie.execute_tool = AsyncMock(return_value={"ok": True})
    genie.human_in_loop = MagicMock()

    turn_1 = {"message": {"role": "assistant", "content": "done"}}
    genie.llm = MagicMock()
    genie.llm.chat = AsyncMock(return_value=turn_1)

    agent = ReActAgent(
        genie=genie,
        agent_config={"use_native_tool_use": True, "max_iterations": 1},
    )
    await agent.run(goal="hello")

    kw = genie.llm.chat.await_args.kwargs
    assert "attribution_tags" not in kw
    assert "budget_scope" not in kw
