"""Unit tests for M7 — ReActAgent's native tool-use loop."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from genie_tooling.agents.react_agent import ReActAgent


def _mock_tool(identifier: str, description: str = "", schema: dict | None = None):
    t = MagicMock()
    t.identifier = identifier
    t.get_metadata = AsyncMock(
        return_value={
            "identifier": identifier,
            "name": identifier,
            "description_llm": description or f"The {identifier} tool.",
            "input_schema": schema or {"type": "object", "properties": {}},
        }
    )
    return t


def _make_genie(tool_ids=("calculator_tool",), tool_response=None):
    g = MagicMock()
    g.tools = MagicMock()
    g.tools.list = AsyncMock(return_value=[_mock_tool(t) for t in tool_ids])
    g.tools.get_definition = AsyncMock(return_value="def text")
    g.observability = MagicMock()
    g.observability.trace_event = AsyncMock()
    g.execute_tool = AsyncMock(return_value=tool_response or {"result": 42})
    g.human_in_loop = MagicMock()
    g.human_in_loop.request_approval = AsyncMock()
    g.llm = MagicMock()
    g.llm.chat = AsyncMock()
    return g


@pytest.mark.asyncio()
async def test_native_path_returns_final_answer_immediately():
    """When the LLM responds with no tool_calls, treat its text as the
    final answer."""
    genie = _make_genie()
    genie.llm.chat = AsyncMock(
        return_value={
            "message": {"role": "assistant", "content": "the answer is 42"},
            "finish_reason": "stop",
        }
    )

    agent = ReActAgent(
        genie=genie, agent_config={"use_native_tool_use": True, "max_iterations": 3}
    )
    result = await agent.run(goal="what is 6 times 7")
    assert result["status"] == "success"
    assert result["output"] == "the answer is 42"
    # No tools were called
    genie.execute_tool.assert_not_awaited()


@pytest.mark.asyncio()
async def test_native_path_executes_tool_and_loops_to_answer():
    """Two-turn loop: tool_call → tool result → final answer."""
    genie = _make_genie(tool_response={"result": 42})

    responses = [
        # Turn 1: LLM asks for the calculator
        {
            "message": {
                "role": "assistant",
                "content": "let me calculate",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "calculator_tool",
                            "arguments": json.dumps({"num1": 6, "num2": 7, "operation": "multiply"}),
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        },
        # Turn 2: LLM gives the final answer
        {
            "message": {"role": "assistant", "content": "the answer is 42"},
            "finish_reason": "stop",
        },
    ]
    genie.llm.chat = AsyncMock(side_effect=responses)

    agent = ReActAgent(
        genie=genie, agent_config={"use_native_tool_use": True, "max_iterations": 3}
    )
    result = await agent.run(goal="what is 6 times 7")
    assert result["status"] == "success"
    assert result["output"] == "the answer is 42"
    # Tool was executed exactly once
    genie.execute_tool.assert_awaited_once()
    call = genie.execute_tool.await_args
    assert call.args[0] == "calculator_tool"
    # The agent's tool params arrive as kwargs, alongside a context= kwarg
    # carrying the caller_chain (C1 provenance).
    assert call.kwargs["num1"] == 6
    assert call.kwargs["num2"] == 7
    assert call.kwargs["operation"] == "multiply"
    # Phase 6A.3/6A.6: tool context carries caller_chain + correlation_id;
    # attribution_tags/budget_scope only included when caller passes them.
    ctx = call.kwargs["context"]
    assert ctx["caller_chain"] == ["ReActAgent.native_tool_use"]
    assert "correlation_id" in ctx


@pytest.mark.asyncio()
async def test_native_path_passes_tool_specs_in_chat_call():
    """The LLM chat call must include tools= in OpenAI-function-spec shape."""
    genie = _make_genie()
    genie.llm.chat = AsyncMock(
        return_value={"message": {"role": "assistant", "content": "done"}}
    )
    agent = ReActAgent(
        genie=genie, agent_config={"use_native_tool_use": True, "max_iterations": 1}
    )
    await agent.run(goal="test")
    call_kwargs = genie.llm.chat.await_args.kwargs
    assert "tools" in call_kwargs
    tools_arg = call_kwargs["tools"]
    assert tools_arg[0]["type"] == "function"
    assert tools_arg[0]["function"]["name"] == "calculator_tool"
    assert call_kwargs.get("tool_choice") == "auto"


@pytest.mark.asyncio()
async def test_native_path_handles_invalid_tool_name_in_observation():
    """Model hallucinates a tool name → result becomes an error observation
    so the model can retry on the next turn."""
    genie = _make_genie()
    responses = [
        {
            "message": {
                "role": "assistant",
                "content": "trying a tool",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "nonexistent_tool",
                            "arguments": "{}",
                        },
                    }
                ],
            }
        },
        {
            "message": {"role": "assistant", "content": "ok giving up"},
        },
    ]
    genie.llm.chat = AsyncMock(side_effect=responses)
    agent = ReActAgent(
        genie=genie, agent_config={"use_native_tool_use": True, "max_iterations": 3}
    )
    result = await agent.run(goal="test")
    assert result["status"] == "success"
    # The non-existent tool was NOT executed
    genie.execute_tool.assert_not_awaited()
    # The error observation made it into the conversation context
    # (verifiable via the second chat call's messages)
    second_call = genie.llm.chat.await_args_list[1]
    msgs = second_call.kwargs["messages"]
    tool_msg = [m for m in msgs if m.get("role") == "tool"][-1]
    assert "not registered" in tool_msg["content"]


@pytest.mark.asyncio()
async def test_native_path_honors_hitl_per_action_denial():
    """When hitl_per_action=True and approval is denied, the tool must NOT
    execute and the denial becomes the tool result the model sees."""
    genie = _make_genie()
    genie.human_in_loop.request_approval = AsyncMock(
        return_value={"status": "denied", "approver_id": "policy_v1", "reason": "blocked by policy"}
    )
    responses = [
        {
            "message": {
                "role": "assistant",
                "content": "calling tool",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "calculator_tool", "arguments": "{}"},
                    }
                ],
            }
        },
        {"message": {"role": "assistant", "content": "guess I can't"}},
    ]
    genie.llm.chat = AsyncMock(side_effect=responses)
    agent = ReActAgent(
        genie=genie,
        agent_config={
            "use_native_tool_use": True,
            "max_iterations": 3,
            "hitl_per_action": True,
        },
    )
    result = await agent.run(goal="test")
    assert result["status"] == "success"
    # Tool did NOT execute because of denial
    genie.execute_tool.assert_not_awaited()
    # Denial appears in the second turn's tool message
    second_call = genie.llm.chat.await_args_list[1]
    msgs = second_call.kwargs["messages"]
    tool_msg = [m for m in msgs if m.get("role") == "tool"][-1]
    assert "denied" in tool_msg["content"].lower()
    assert "policy_v1" in tool_msg["content"]


@pytest.mark.asyncio()
async def test_native_path_hits_max_iterations_when_model_loops_forever():
    """If the model keeps requesting tool calls without ever emitting an
    answer, the loop terminates at max_iterations."""
    genie = _make_genie()
    loop_response = {
        "message": {
            "role": "assistant",
            "content": "again",
            "tool_calls": [
                {
                    "id": "call_x",
                    "type": "function",
                    "function": {"name": "calculator_tool", "arguments": "{}"},
                }
            ],
        }
    }
    genie.llm.chat = AsyncMock(return_value=loop_response)
    agent = ReActAgent(
        genie=genie, agent_config={"use_native_tool_use": True, "max_iterations": 2}
    )
    result = await agent.run(goal="test")
    assert result["status"] == "max_iterations_reached"
    assert genie.execute_tool.await_count == 2  # called once per iteration


@pytest.mark.asyncio()
async def test_default_mode_remains_regex_loop():
    """Backward-compat: use_native_tool_use defaults to False. The agent
    falls through to the regex-parsing loop and the chat call does NOT
    include the tools= kwarg."""
    genie = _make_genie()

    # Need to mock genie.prompts.render_prompt for the regex path.
    async def _render(name=None, template_content=None, **kw):
        return "rendered prompt"
    genie.prompts = MagicMock()
    genie.prompts.render_prompt = _render

    # LLM emits a final answer immediately in the regex format.
    genie.llm.chat = AsyncMock(
        return_value={"message": {"content": "Thought: done\nAnswer: 42"}}
    )

    agent = ReActAgent(genie=genie, agent_config={"max_iterations": 1})
    assert agent.use_native_tool_use is False
    result = await agent.run(goal="test")
    assert result["status"] == "success"
    # The chat call must NOT include the tools= kwarg in regex mode.
    call_kwargs = genie.llm.chat.await_args.kwargs
    assert "tools" not in call_kwargs
