"""
Regression test for F15: ReActAgent and PlanAndExecuteAgent must work without
a user-supplied template in genie.prompts.render_prompt — they should fall
back to their bundled DEFAULT_* constants instead of failing with
'Failed to render ReAct prompt.'.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from genie_tooling.agents.plan_and_execute_agent import (
    DEFAULT_PLANNER_SYSTEM_PROMPT,
    PlanAndExecuteAgent,
)
from genie_tooling.agents.react_agent import (
    DEFAULT_REACT_SYSTEM_PROMPT,
    ReActAgent,
)


def _make_mock_genie_for_react():
    g = MagicMock()
    g.tools = MagicMock()
    g.tools.list = AsyncMock(return_value=[])
    g.tools.get_definition = AsyncMock(return_value=None)
    g.observability = MagicMock()
    g.observability.trace_event = AsyncMock()
    # Registry has NO template under react_agent_system_prompt_v1 -> None.
    # Built-in default fallback render is then invoked; mock returns a
    # canned rendered string so the agent can proceed.
    async def render_prompt(name=None, template_content=None, **kw):
        if name and not template_content:
            return None  # registry lookup miss
        return "<<rendered ReAct prompt>>"
    g.prompts = MagicMock()
    g.prompts.render_prompt = render_prompt
    # The LLM emits a final Answer so the agent terminates in one iteration.
    g.llm = MagicMock()
    g.llm.chat = AsyncMock(
        return_value={"message": {"content": "Thought: ok\nAnswer: done"}}
    )
    return g


def _make_mock_genie_for_plan_and_execute():
    g = MagicMock()
    g.tools = MagicMock()
    g.tools.list = AsyncMock(return_value=[])
    g.tools.get_definition = AsyncMock(return_value=None)
    g.observability = MagicMock()
    g.observability.trace_event = AsyncMock()

    async def render_chat_prompt(name=None, template_content=None, **kw):
        if name and not template_content:
            return None  # registry lookup miss
        return [{"role": "user", "content": "<<rendered planner prompt>>"}]
    g.prompts = MagicMock()
    g.prompts.render_chat_prompt = render_chat_prompt

    # Planner LLM returns a valid one-step plan as JSON.
    g.llm = MagicMock()
    g.llm.chat = AsyncMock(
        return_value={
            "message": {
                "content": (
                    '{"plan":[{"step_number":1,"tool_id":"noop","params":{},'
                    '"reasoning":"trivial","output_variable_name":null}],'
                    '"overall_reasoning":"trivial"}'
                )
            }
        }
    )

    from genie_tooling.agents.types import PlanModelPydantic, PlanStepModelPydantic

    g.llm.parse_output = AsyncMock(
        return_value=PlanModelPydantic(
            plan=[
                PlanStepModelPydantic(
                    step_number=1,
                    tool_id="noop",
                    params={},
                    reasoning="trivial",
                    output_variable_name=None,
                )
            ],
            overall_reasoning="trivial",
        )
    )

    # The agent ends in 'error' status because there's no actual `noop` tool
    # to execute — we don't care; this test only proves the prompt fallback
    # path ran without short-circuiting on the registry miss.
    g.execute_tool = AsyncMock(side_effect=Exception("no such tool"))
    g.human_in_loop = MagicMock()
    g.human_in_loop.request_approval = AsyncMock(
        return_value={"status": "approved", "approver_id": "test"}
    )
    return g


def test_react_agent_has_a_default_system_prompt_constant():
    """The constant must exist, be non-empty, and reference the three vars
    the agent passes (goal, scratchpad, tool_definitions)."""
    assert isinstance(DEFAULT_REACT_SYSTEM_PROMPT, str)
    assert DEFAULT_REACT_SYSTEM_PROMPT.strip()
    for required in ("{{ goal }}", "{{ scratchpad }}", "{{ tool_definitions }}"):
        assert required in DEFAULT_REACT_SYSTEM_PROMPT, (
            f"Default ReAct prompt is missing required Jinja variable {required!r}"
        )


def test_plan_and_execute_has_a_default_system_prompt_constant():
    assert isinstance(DEFAULT_PLANNER_SYSTEM_PROMPT, str)
    assert DEFAULT_PLANNER_SYSTEM_PROMPT.strip()
    for required in ("{{ goal }}", "{{ tool_definitions }}"):
        assert required in DEFAULT_PLANNER_SYSTEM_PROMPT, (
            f"Default planner prompt is missing required Jinja variable {required!r}"
        )


@pytest.mark.asyncio
async def test_react_agent_falls_back_to_default_prompt_when_registry_misses():
    """If genie.prompts.render_prompt returns None for the configured ID,
    the agent must invoke the same method again with template_content set
    to the built-in default — and proceed normally rather than returning
    {'status': 'error', 'output': 'Failed to render ReAct prompt.'}."""
    genie = _make_mock_genie_for_react()
    agent = ReActAgent(genie=genie, agent_config={"max_iterations": 1})
    result = await agent.run(goal="test")
    assert result["status"] == "success", (
        f"agent should have completed successfully via default prompt fallback; "
        f"got status={result['status']!r}, output={result['output']!r}"
    )
    assert "done" in str(result["output"]).lower()


@pytest.mark.asyncio
async def test_plan_and_execute_falls_back_to_default_when_registry_misses():
    """Same as above for PlanAndExecuteAgent's planner prompt."""
    genie = _make_mock_genie_for_plan_and_execute()
    agent = PlanAndExecuteAgent(
        genie=genie, agent_config={"max_plan_retries": 0, "max_step_retries": 0}
    )
    result = await agent.run(goal="test")
    # The plan generation must have succeeded — the noop step then fails on
    # execution because there's no real tool, which is expected and not what
    # this test is verifying. The key signal: status is NOT "error" with
    # output "PlannerPromptRenderingFailed" or similar plan-stage error.
    assert result["status"] != "error" or "render" not in str(result.get("output", "")).lower(), (
        f"agent failed at the plan-rendering stage despite the built-in default "
        f"prompt being available; got: {result!r}"
    )
