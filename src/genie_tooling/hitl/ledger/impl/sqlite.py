"""SQLiteHITLLedgerPlugin: stdlib sqlite3 over asyncio.to_thread."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional

from genie_tooling.hitl.ledger.abc import HITLLedgerPlugin
from genie_tooling.hitl.ledger.types import LedgerEntry, LedgerQuery

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS hitl_ledger (
    request_id        TEXT PRIMARY KEY,
    decision_id       TEXT,
    correlation_id    TEXT,
    tool_id           TEXT,
    params_json       TEXT,
    tool_metadata_json TEXT,
    status            TEXT NOT NULL,
    approver_id       TEXT,
    reason            TEXT,
    requested_at      REAL NOT NULL,
    decided_at        REAL NOT NULL,
    user_identity_json TEXT,
    attribution_tags_json TEXT,
    extra_json        TEXT
);
CREATE INDEX IF NOT EXISTS idx_hitl_ledger_tool_decided
    ON hitl_ledger (tool_id, decided_at DESC);
CREATE INDEX IF NOT EXISTS idx_hitl_ledger_status_decided
    ON hitl_ledger (status, decided_at DESC);
CREATE INDEX IF NOT EXISTS idx_hitl_ledger_decision
    ON hitl_ledger (decision_id);
"""


class SQLiteHITLLedgerPlugin(HITLLedgerPlugin):
    plugin_id: str = "sqlite_hitl_ledger_v1"
    description: str = (
        "SQLite-backed HITL approval ledger. Single-host production-grade "
        "durability. Use postgres_hitl_ledger_v1 for multi-host."
    )

    _db_path: str
    _conn_lock: threading.Lock

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._db_path = str(cfg.get("db_path", "./genie_hitl_ledger.sqlite"))
        self._conn_lock = threading.Lock()
        await asyncio.to_thread(self._init_db)
        logger.info(f"{self.plugin_id}: ready (db_path={self._db_path!r})")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False, isolation_level="DEFERRED")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        parent = os.path.dirname(self._db_path)
        if parent and parent != ":memory:" and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        with self._conn_lock:
            conn = self._connect()
            try:
                for stmt in [s.strip() for s in _SCHEMA.split(";") if s.strip()]:
                    conn.execute(stmt)
                conn.commit()
            finally:
                conn.close()

    async def record(self, entry: LedgerEntry) -> None:
        await asyncio.to_thread(self._record_sync, entry)

    def _record_sync(self, e: LedgerEntry) -> None:
        with self._conn_lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO hitl_ledger (
                        request_id, decision_id, correlation_id, tool_id, params_json,
                        tool_metadata_json, status, approver_id, reason, requested_at,
                        decided_at, user_identity_json, attribution_tags_json, extra_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(request_id) DO UPDATE SET
                        decision_id=excluded.decision_id,
                        correlation_id=excluded.correlation_id,
                        tool_id=excluded.tool_id,
                        params_json=excluded.params_json,
                        tool_metadata_json=excluded.tool_metadata_json,
                        status=excluded.status,
                        approver_id=excluded.approver_id,
                        reason=excluded.reason,
                        decided_at=excluded.decided_at,
                        user_identity_json=excluded.user_identity_json,
                        attribution_tags_json=excluded.attribution_tags_json,
                        extra_json=excluded.extra_json
                    """,
                    (
                        e.request_id,
                        e.decision_id,
                        e.correlation_id,
                        e.tool_id,
                        json.dumps(e.params, default=str) if e.params is not None else None,
                        json.dumps(e.tool_metadata) if e.tool_metadata is not None else None,
                        e.status,
                        e.approver_id,
                        e.reason,
                        e.requested_at,
                        e.decided_at,
                        json.dumps(e.user_identity) if e.user_identity is not None else None,
                        json.dumps(e.attribution_tags) if e.attribution_tags is not None else None,
                        json.dumps(e.extra, default=str) if e.extra is not None else None,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    async def get(self, request_id: str) -> Optional[LedgerEntry]:
        return await asyncio.to_thread(self._get_sync, request_id)

    def _get_sync(self, request_id: str) -> Optional[LedgerEntry]:
        with self._conn_lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "SELECT request_id, decision_id, correlation_id, tool_id, params_json, "
                    "tool_metadata_json, status, approver_id, reason, requested_at, decided_at, "
                    "user_identity_json, attribution_tags_json, extra_json "
                    "FROM hitl_ledger WHERE request_id = ?",
                    (request_id,),
                )
                row = cur.fetchone()
                return self._row_to_entry(row) if row else None
            finally:
                conn.close()

    async def search(self, query: LedgerQuery) -> List[LedgerEntry]:
        return await asyncio.to_thread(self._search_sync, query)

    def _search_sync(self, q: LedgerQuery) -> List[LedgerEntry]:
        clauses: List[str] = []
        params: List[Any] = []
        if q.tool_id:
            clauses.append("tool_id = ?")
            params.append(q.tool_id)
        if q.status:
            clauses.append("status = ?")
            params.append(q.status)
        if q.approver_id_prefix:
            clauses.append("approver_id LIKE ?")
            params.append(f"{q.approver_id_prefix}%")
        if q.since is not None:
            clauses.append("decided_at >= ?")
            params.append(q.since)
        if q.until is not None:
            clauses.append("decided_at <= ?")
            params.append(q.until)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(q.limit)
        with self._conn_lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    f"SELECT request_id, decision_id, correlation_id, tool_id, params_json, "
                    f"tool_metadata_json, status, approver_id, reason, requested_at, decided_at, "
                    f"user_identity_json, attribution_tags_json, extra_json "
                    f"FROM hitl_ledger {where} ORDER BY decided_at DESC LIMIT ?",
                    params,
                )
                rows = cur.fetchall()
            finally:
                conn.close()
        entries = [self._row_to_entry(r) for r in rows]
        if q.attribution_tag:
            entries = [
                e for e in entries
                if e.attribution_tags
                and all(e.attribution_tags.get(k) == v for k, v in q.attribution_tag.items())
            ]
        return entries

    @staticmethod
    def _row_to_entry(row) -> LedgerEntry:
        return LedgerEntry(
            request_id=row[0],
            decision_id=row[1],
            correlation_id=row[2],
            tool_id=row[3],
            params=json.loads(row[4]) if row[4] else None,
            tool_metadata=json.loads(row[5]) if row[5] else None,
            status=row[6],
            approver_id=row[7],
            reason=row[8],
            requested_at=float(row[9]),
            decided_at=float(row[10]),
            user_identity=json.loads(row[11]) if row[11] else None,
            attribution_tags=json.loads(row[12]) if row[12] else None,
            extra=json.loads(row[13]) if row[13] else None,
        )

    async def teardown(self) -> None:
        pass
