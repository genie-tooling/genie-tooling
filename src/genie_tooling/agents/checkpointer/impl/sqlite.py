"""SQLiteAgentCheckpointerPlugin: stdlib-sqlite-backed checkpointer.

Uses ``sqlite3`` from the stdlib (no new deps) and runs the synchronous
DB calls inside ``asyncio.to_thread`` so the asyncio event loop stays
responsive.

Schema is a single ``agent_checkpoints`` table; the ``state_blob`` column
is JSON serialized. Indexes cover ``(agent_id, updated_at DESC)`` for
listing.

For multi-worker production deployments, swap to the Postgres-backed
implementation (TBD). The protocol contract is the same.

Config::

    db_path: str   (default: "./genie_checkpoints.sqlite")
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional

from genie_tooling.agents.checkpointer.abc import AgentCheckpointerPlugin
from genie_tooling.agents.checkpointer.types import CheckpointMeta, CheckpointState

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_checkpoints (
    run_id            TEXT PRIMARY KEY,
    agent_id          TEXT NOT NULL,
    iteration         INTEGER NOT NULL,
    goal              TEXT,
    state_blob        TEXT NOT NULL,
    status            TEXT NOT NULL,
    created_at        REAL NOT NULL,
    updated_at        REAL NOT NULL,
    attribution_tags  TEXT,
    user_identity     TEXT,
    correlation_id    TEXT,
    error             TEXT
);
CREATE INDEX IF NOT EXISTS idx_agent_checkpoints_agent_updated
    ON agent_checkpoints (agent_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_checkpoints_status_updated
    ON agent_checkpoints (status, updated_at DESC);
"""


class SQLiteAgentCheckpointerPlugin(AgentCheckpointerPlugin):
    plugin_id: str = "sqlite_agent_checkpointer_v1"
    description: str = (
        "SQLite-backed agent checkpointer. Uses stdlib sqlite3 in a worker "
        "thread. Survives process restart. Single-host only — for multi-host "
        "deployments use postgres_agent_checkpointer_v1."
    )

    _db_path: str
    _conn_lock: threading.Lock

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._db_path = str(cfg.get("db_path", "./genie_checkpoints.sqlite"))
        # ":memory:" is a valid path for tests
        self._conn_lock = threading.Lock()
        await asyncio.to_thread(self._init_db)
        logger.info(f"{self.plugin_id}: ready (db_path={self._db_path!r})")

    def _connect(self) -> sqlite3.Connection:
        # check_same_thread=False — we serialise via _conn_lock
        conn = sqlite3.connect(
            self._db_path,
            check_same_thread=False,
            isolation_level="DEFERRED",
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        # ensure parent dir exists (for relative paths in CWDs that don't have it)
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

    async def save_checkpoint(self, state: CheckpointState) -> None:
        await asyncio.to_thread(self._save_sync, state)

    def _save_sync(self, state: CheckpointState) -> None:
        with self._conn_lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO agent_checkpoints (
                        run_id, agent_id, iteration, goal, state_blob, status,
                        created_at, updated_at, attribution_tags, user_identity,
                        correlation_id, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        iteration=excluded.iteration,
                        goal=excluded.goal,
                        state_blob=excluded.state_blob,
                        status=excluded.status,
                        updated_at=excluded.updated_at,
                        attribution_tags=excluded.attribution_tags,
                        user_identity=excluded.user_identity,
                        correlation_id=excluded.correlation_id,
                        error=excluded.error
                    """,
                    (
                        state.run_id,
                        state.agent_id,
                        state.iteration,
                        state.goal,
                        json.dumps(state.state_blob, default=str),
                        state.status,
                        state.created_at,
                        state.updated_at,
                        json.dumps(state.attribution_tags) if state.attribution_tags else None,
                        json.dumps(state.user_identity) if state.user_identity else None,
                        state.correlation_id,
                        state.error,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    async def load_checkpoint(self, run_id: str) -> Optional[CheckpointState]:
        return await asyncio.to_thread(self._load_sync, run_id)

    def _load_sync(self, run_id: str) -> Optional[CheckpointState]:
        with self._conn_lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "SELECT run_id, agent_id, iteration, goal, state_blob, status, "
                    "created_at, updated_at, attribution_tags, user_identity, "
                    "correlation_id, error FROM agent_checkpoints WHERE run_id = ?",
                    (run_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                return CheckpointState(
                    run_id=row[0],
                    agent_id=row[1],
                    iteration=int(row[2]),
                    goal=row[3] or "",
                    state_blob=json.loads(row[4]),
                    status=row[5],
                    created_at=float(row[6]),
                    updated_at=float(row[7]),
                    attribution_tags=json.loads(row[8]) if row[8] else None,
                    user_identity=json.loads(row[9]) if row[9] else None,
                    correlation_id=row[10],
                    error=row[11],
                )
            finally:
                conn.close()

    async def list_runs(
        self,
        agent_id: Optional[str] = None,
        status: Optional[str] = None,
        attribution_tag: Optional[Dict[str, str]] = None,
        limit: int = 100,
    ) -> List[CheckpointMeta]:
        return await asyncio.to_thread(
            self._list_sync, agent_id, status, attribution_tag, limit
        )

    def _list_sync(
        self,
        agent_id: Optional[str],
        status: Optional[str],
        attribution_tag: Optional[Dict[str, str]],
        limit: int,
    ) -> List[CheckpointMeta]:
        clauses: List[str] = []
        params: List[Any] = []
        if agent_id:
            clauses.append("agent_id = ?")
            params.append(agent_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        with self._conn_lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    f"SELECT run_id, agent_id, iteration, status, created_at, "
                    f"updated_at, attribution_tags FROM agent_checkpoints {where} "
                    f"ORDER BY updated_at DESC LIMIT ?",
                    params,
                )
                rows = cur.fetchall()
            finally:
                conn.close()
        out: List[CheckpointMeta] = []
        for r in rows:
            tags_blob = r[6]
            tags = json.loads(tags_blob) if tags_blob else None
            if attribution_tag:
                if tags is None or not all(tags.get(k) == v for k, v in attribution_tag.items()):
                    continue
            out.append(
                CheckpointMeta(
                    run_id=r[0],
                    agent_id=r[1],
                    iteration=int(r[2]),
                    status=r[3],
                    created_at=float(r[4]),
                    updated_at=float(r[5]),
                    attribution_tags=tags,
                )
            )
        return out

    async def delete_checkpoint(self, run_id: str) -> None:
        await asyncio.to_thread(self._delete_sync, run_id)

    def _delete_sync(self, run_id: str) -> None:
        with self._conn_lock:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM agent_checkpoints WHERE run_id = ?", (run_id,))
                conn.commit()
            finally:
                conn.close()

    async def teardown(self) -> None:
        # Nothing to close — connections are per-call.
        pass
