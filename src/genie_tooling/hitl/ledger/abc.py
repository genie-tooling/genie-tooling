"""Abstract protocol for the HITL approval ledger."""
from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

from genie_tooling.core.types import Plugin

from .types import LedgerEntry, LedgerQuery


@runtime_checkable
class HITLLedgerPlugin(Plugin, Protocol):
    """Persistent record of HITL approval decisions.

    The HITLManager writes every decision (approve/deny/error/timeout) to the
    configured ledger plugin. Audit / governance teams query it after the fact:
    "every approved kubectl_delete this week", "every denied request for tenant
    X", etc. Joins to ``DecisionRecord`` via ``decision_id`` /
    ``correlation_id``.
    """

    async def record(self, entry: LedgerEntry) -> None: ...

    async def get(self, request_id: str) -> Optional[LedgerEntry]: ...

    async def search(self, query: LedgerQuery) -> List[LedgerEntry]: ...
