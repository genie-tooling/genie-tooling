"""InMemoryHITLLedgerPlugin: list-backed reference impl."""
from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any, Dict, List, Optional

from genie_tooling.hitl.ledger.abc import HITLLedgerPlugin
from genie_tooling.hitl.ledger.types import LedgerEntry, LedgerQuery


class InMemoryHITLLedgerPlugin(HITLLedgerPlugin):
    plugin_id: str = "in_memory_hitl_ledger_v1"
    description: str = (
        "In-process HITL approval ledger. Single-process only — does NOT survive "
        "process restart. Use sqlite_hitl_ledger_v1 for durable storage."
    )

    _entries: List[LedgerEntry]
    _index: Dict[str, int]
    _lock: asyncio.Lock

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._entries = []
        self._index = {}
        self._lock = asyncio.Lock()

    async def record(self, entry: LedgerEntry) -> None:
        async with self._lock:
            stored = replace(entry)
            if entry.request_id in self._index:
                self._entries[self._index[entry.request_id]] = stored
            else:
                self._index[entry.request_id] = len(self._entries)
                self._entries.append(stored)

    async def get(self, request_id: str) -> Optional[LedgerEntry]:
        async with self._lock:
            idx = self._index.get(request_id)
            return replace(self._entries[idx]) if idx is not None else None

    async def search(self, query: LedgerQuery) -> List[LedgerEntry]:
        async with self._lock:
            out: List[LedgerEntry] = []
            for entry in self._entries:
                if query.tool_id and entry.tool_id != query.tool_id:
                    continue
                if query.status and entry.status != query.status:
                    continue
                if query.approver_id_prefix and not (entry.approver_id or "").startswith(query.approver_id_prefix):
                    continue
                if query.since is not None and entry.decided_at < query.since:
                    continue
                if query.until is not None and entry.decided_at > query.until:
                    continue
                if query.attribution_tag:
                    e_tags = entry.attribution_tags or {}
                    if not all(e_tags.get(k) == v for k, v in query.attribution_tag.items()):
                        continue
                out.append(replace(entry))
            out.sort(key=lambda e: e.decided_at, reverse=True)
            return out[: query.limit]

    async def teardown(self) -> None:
        self._entries.clear()
        self._index.clear()
