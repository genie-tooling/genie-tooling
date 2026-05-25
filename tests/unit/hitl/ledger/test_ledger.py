"""Phase 6A.7 — HITL approval ledger tests."""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from genie_tooling.hitl.ledger import HITLLedgerPlugin, LedgerEntry, LedgerQuery
from genie_tooling.hitl.ledger.impl.in_memory import InMemoryHITLLedgerPlugin
from genie_tooling.hitl.ledger.impl.sqlite import SQLiteHITLLedgerPlugin
from genie_tooling.hitl.manager import HITLManager


@pytest_asyncio.fixture(params=["in_memory", "sqlite"])
async def ledger(request, tmp_path) -> HITLLedgerPlugin:
    if request.param == "in_memory":
        led = InMemoryHITLLedgerPlugin()
        await led.setup()
    else:
        led = SQLiteHITLLedgerPlugin()
        await led.setup({"db_path": str(tmp_path / "ledger.sqlite")})
    yield led
    await led.teardown()


def _entry(request_id="r1", status="approved", tool_id="kubectl_apply", **extras):
    now = time.time()
    return LedgerEntry(
        request_id=request_id,
        decision_id=extras.get("decision_id", "d1"),
        correlation_id=extras.get("correlation_id"),
        tool_id=tool_id,
        params={"namespace": "prod"},
        tool_metadata={"side_effects": "write"},
        status=status,
        approver_id=extras.get("approver_id", "webhook:human1"),
        reason=extras.get("reason"),
        requested_at=extras.get("requested_at", now - 1),
        decided_at=extras.get("decided_at", now),
        user_identity=extras.get("user_identity"),
        attribution_tags=extras.get("attribution_tags") or {"team": "platform"},
    )


@pytest.mark.asyncio()
async def test_record_and_get_roundtrip(ledger):
    e = _entry()
    await ledger.record(e)
    got = await ledger.get("r1")
    assert got is not None
    assert got.tool_id == "kubectl_apply"
    assert got.status == "approved"
    assert got.params == {"namespace": "prod"}
    assert got.attribution_tags == {"team": "platform"}


@pytest.mark.asyncio()
async def test_search_by_status(ledger):
    await ledger.record(_entry("r1", status="approved"))
    await ledger.record(_entry("r2", status="denied"))
    await ledger.record(_entry("r3", status="approved"))

    approved = await ledger.search(LedgerQuery(status="approved"))
    assert {e.request_id for e in approved} == {"r1", "r3"}


@pytest.mark.asyncio()
async def test_search_by_tool_id_and_attribution(ledger):
    await ledger.record(_entry("a1", tool_id="kubectl_apply", attribution_tags={"team": "platform"}))
    await ledger.record(_entry("a2", tool_id="kubectl_apply", attribution_tags={"team": "search"}))
    await ledger.record(_entry("a3", tool_id="slack_post", attribution_tags={"team": "platform"}))

    results = await ledger.search(LedgerQuery(tool_id="kubectl_apply", attribution_tag={"team": "platform"}))
    assert {e.request_id for e in results} == {"a1"}


@pytest.mark.asyncio()
async def test_search_by_time_window(ledger):
    now = time.time()
    await ledger.record(_entry("old", decided_at=now - 1000))
    await ledger.record(_entry("recent", decided_at=now))

    recent = await ledger.search(LedgerQuery(since=now - 500))
    assert {e.request_id for e in recent} == {"recent"}


@pytest.mark.asyncio()
async def test_search_orders_decided_at_desc(ledger):
    now = time.time()
    await ledger.record(_entry("old", decided_at=now - 100))
    await ledger.record(_entry("new", decided_at=now))
    results = await ledger.search(LedgerQuery(limit=10))
    assert [e.request_id for e in results[:2]] == ["new", "old"]


@pytest.mark.asyncio()
async def test_upsert_replaces_on_same_request_id(ledger):
    await ledger.record(_entry("r1", status="ask_human"))
    await ledger.record(_entry("r1", status="approved", reason="final approval"))
    got = await ledger.get("r1")
    assert got.status == "approved"
    assert got.reason == "final approval"


# ---- HITLManager integration ----


class _StubApprover:
    plugin_id = "stub_approver"
    description = "stub"

    async def setup(self, config=None):
        pass

    async def teardown(self):
        pass

    async def request_approval(self, request):
        return {
            "request_id": request.get("request_id"),
            "status": "approved",
            "approver_id": "stub:auto",
            "reason": "stub",
            "timestamp": time.time(),
        }


@pytest.mark.asyncio()
async def test_hitl_manager_writes_to_ledger(tmp_path):
    led = SQLiteHITLLedgerPlugin()
    await led.setup({"db_path": str(tmp_path / "led.sqlite")})

    approver = _StubApprover()
    pm = MagicMock()

    async def _load(plugin_id, config):
        if plugin_id == "stub_approver":
            return approver
        if plugin_id == "sqlite_hitl_ledger_v1":
            return led
        return None

    pm.get_plugin_instance = AsyncMock(side_effect=_load)

    mgr = HITLManager(
        pm,
        default_approver_id="stub_approver",
        approver_configurations={},
        ledger_id="sqlite_hitl_ledger_v1",
        ledger_configurations={"sqlite_hitl_ledger_v1": {"db_path": str(tmp_path / "led.sqlite")}},
    )

    await mgr.request_approval(
        {
            "request_id": "req_xyz",
            "prompt": "do it?",
            "data_to_approve": {
                "tool_id": "kubectl_apply",
                "params": {"namespace": "prod"},
                "tool_metadata": {"side_effects": "write"},
            },
            "context": {"attribution_tags": {"team": "platform"}, "decision_id": "dec_abc"},
            "timeout_seconds": 60,
        }
    )

    entry = await led.get("req_xyz")
    assert entry is not None
    assert entry.status == "approved"
    assert entry.tool_id == "kubectl_apply"
    assert entry.decision_id == "dec_abc"
    assert entry.attribution_tags == {"team": "platform"}
    await led.teardown()
