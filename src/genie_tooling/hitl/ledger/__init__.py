"""Phase 6A.7 — durable approval ledger.

Webhook approvals are fire-and-forget. ``HITLLedgerPlugin`` persists every
HITL decision so audit can later answer "show me every approved tool call
for tenant X last week" — joinable to the parent ``DecisionRecord`` via
``decision_id`` / ``correlation_id``.

Ships two implementations:
* ``in_memory_hitl_ledger_v1`` — list-backed, single-process; tests & dev.
* ``sqlite_hitl_ledger_v1`` — stdlib sqlite3 over asyncio.to_thread.
A Postgres impl is the intended production target (designed but not bundled).
"""
from .abc import HITLLedgerPlugin
from .types import LedgerEntry, LedgerQuery

__all__ = ["HITLLedgerPlugin", "LedgerEntry", "LedgerQuery"]
