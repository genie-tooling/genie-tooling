"""Phase 6A.3 — InMemoryBudgetEnforcerPlugin tests.

Validates token / cost / call-count / wall-clock caps and per-provider tracking.
"""
from __future__ import annotations

import asyncio

import pytest

from genie_tooling.budget import BudgetExceeded, BudgetSpec
from genie_tooling.budget.impl.in_memory import InMemoryBudgetEnforcerPlugin


@pytest.mark.asyncio
async def test_token_cap_triggers_after_threshold():
    p = InMemoryBudgetEnforcerPlugin()
    await p.setup()
    await p.set_budget("s1", BudgetSpec(max_tokens=100))

    # Under cap
    await p.check_and_charge_llm_call("s1", prompt_tokens=40, completion_tokens=20, provider_id="openai")
    snap = await p.get_usage("s1")
    assert snap.tokens == 60
    assert snap.by_provider["openai"] == 60

    # Exceeds cap (60 + 50 = 110 > 100)
    with pytest.raises(BudgetExceeded) as ei:
        await p.check_and_charge_llm_call("s1", prompt_tokens=30, completion_tokens=20, provider_id="anthropic")
    assert "tokens" in ei.value.scope
    assert ei.value.limit == 100.0


@pytest.mark.asyncio
async def test_cost_cap():
    p = InMemoryBudgetEnforcerPlugin()
    await p.setup()
    await p.set_budget("s1", BudgetSpec(max_cost_usd=1.0))

    await p.check_and_charge_llm_call("s1", 10, 5, "openai", cost_usd=0.5)
    with pytest.raises(BudgetExceeded) as ei:
        await p.check_and_charge_llm_call("s1", 1, 1, "openai", cost_usd=0.6)
    assert "cost_usd" in ei.value.scope


@pytest.mark.asyncio
async def test_tool_call_cap():
    p = InMemoryBudgetEnforcerPlugin()
    await p.setup()
    await p.set_budget("s1", BudgetSpec(max_tool_calls=2))

    await p.check_and_charge_tool_call("s1")
    await p.check_and_charge_tool_call("s1")
    with pytest.raises(BudgetExceeded):
        await p.check_and_charge_tool_call("s1")


@pytest.mark.asyncio
async def test_llm_call_cap():
    p = InMemoryBudgetEnforcerPlugin()
    await p.setup()
    await p.set_budget("s1", BudgetSpec(max_llm_calls=1))

    await p.check_and_charge_llm_call("s1", 1, 1, "openai")
    with pytest.raises(BudgetExceeded):
        await p.check_and_charge_llm_call("s1", 1, 1, "openai")


@pytest.mark.asyncio
async def test_wall_clock_cap_after_simulated_elapse():
    p = InMemoryBudgetEnforcerPlugin()
    await p.setup()
    await p.set_budget("s1", BudgetSpec(max_wall_clock_seconds=0.01))

    await p.check_and_charge_llm_call("s1", 1, 1, "openai")
    await asyncio.sleep(0.02)
    with pytest.raises(BudgetExceeded) as ei:
        await p.check_and_charge_llm_call("s1", 1, 1, "openai")
    assert "wall_clock" in ei.value.scope


@pytest.mark.asyncio
async def test_per_provider_attribution():
    p = InMemoryBudgetEnforcerPlugin()
    await p.setup()
    await p.set_budget("s1", BudgetSpec(max_tokens=10_000))
    await p.check_and_charge_llm_call("s1", 100, 50, "openai")
    await p.check_and_charge_llm_call("s1", 80, 20, "anthropic")
    await p.check_and_charge_llm_call("s1", 30, 30, "openai")
    snap = await p.get_usage("s1")
    assert snap.by_provider["openai"] == 210  # 150+60
    assert snap.by_provider["anthropic"] == 100
    assert snap.tokens == 310


@pytest.mark.asyncio
async def test_global_default_used_when_no_specific_scope_set():
    p = InMemoryBudgetEnforcerPlugin()
    await p.setup({"global_spec": {"max_tokens": 50}})
    # No set_budget call for "s1" — falls through to global
    await p.check_and_charge_llm_call("s1", 25, 25, "openai")
    with pytest.raises(BudgetExceeded):
        await p.check_and_charge_llm_call("s1", 1, 1, "openai")


@pytest.mark.asyncio
async def test_clear_resets_usage():
    p = InMemoryBudgetEnforcerPlugin()
    await p.setup()
    await p.set_budget("s1", BudgetSpec(max_tokens=100))
    await p.check_and_charge_llm_call("s1", 50, 30, "openai")
    await p.clear("s1")
    snap = await p.get_usage("s1")
    assert snap is None
    # Cap still applies after clear
    await p.check_and_charge_llm_call("s1", 99, 0, "openai")
    with pytest.raises(BudgetExceeded):
        await p.check_and_charge_llm_call("s1", 2, 0, "openai")


@pytest.mark.asyncio
async def test_no_cap_means_unlimited():
    p = InMemoryBudgetEnforcerPlugin()
    await p.setup()
    # No set_budget call, no global_spec → all charges succeed
    for _ in range(100):
        await p.check_and_charge_llm_call("s1", 100, 100, "openai")
    snap = await p.get_usage("s1")
    assert snap.tokens == 20_000


@pytest.mark.asyncio
async def test_separate_scopes_are_independent():
    p = InMemoryBudgetEnforcerPlugin()
    await p.setup()
    await p.set_budget("team_a", BudgetSpec(max_tokens=100))
    await p.set_budget("team_b", BudgetSpec(max_tokens=100))
    await p.check_and_charge_llm_call("team_a", 90, 0, "openai")
    # team_b is fresh
    await p.check_and_charge_llm_call("team_b", 90, 0, "openai")
    # team_a is near cap
    with pytest.raises(BudgetExceeded):
        await p.check_and_charge_llm_call("team_a", 20, 0, "openai")
    # team_b still has headroom for now
    snap_b = await p.get_usage("team_b")
    assert snap_b.tokens == 90
