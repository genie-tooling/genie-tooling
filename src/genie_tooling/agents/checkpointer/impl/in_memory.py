"""InMemoryAgentCheckpointerPlugin: dict-backed reference impl."""
from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any, Dict, List, Optional

from genie_tooling.agents.checkpointer.abc import AgentCheckpointerPlugin
from genie_tooling.agents.checkpointer.types import CheckpointMeta, CheckpointState


class InMemoryAgentCheckpointerPlugin(AgentCheckpointerPlugin):
    plugin_id: str = "in_memory_agent_checkpointer_v1"
    description: str = (
        "Dict-backed agent checkpointer. Single-process only — does NOT survive "
        "process restart. Suitable for tests and local dev. For production, use "
        "sqlite_agent_checkpointer_v1 (single host) or postgres_agent_checkpointer_v1 "
        "(multi-host)."
    )

    _runs: Dict[str, CheckpointState]
    _lock: asyncio.Lock

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._runs = {}
        self._lock = asyncio.Lock()

    async def save_checkpoint(self, state: CheckpointState) -> None:
        async with self._lock:
            # Defensive copy so callers can't mutate stored state in place.
            self._runs[state.run_id] = replace(state)

    async def load_checkpoint(self, run_id: str) -> Optional[CheckpointState]:
        async with self._lock:
            existing = self._runs.get(run_id)
            return replace(existing) if existing else None

    async def list_runs(
        self,
        agent_id: Optional[str] = None,
        status: Optional[str] = None,
        attribution_tag: Optional[Dict[str, str]] = None,
        limit: int = 100,
    ) -> List[CheckpointMeta]:
        async with self._lock:
            metas: List[CheckpointMeta] = []
            for s in self._runs.values():
                if agent_id and s.agent_id != agent_id:
                    continue
                if status and s.status != status:
                    continue
                if attribution_tag:
                    s_tags = s.attribution_tags or {}
                    if not all(s_tags.get(k) == v for k, v in attribution_tag.items()):
                        continue
                metas.append(
                    CheckpointMeta(
                        run_id=s.run_id,
                        agent_id=s.agent_id,
                        iteration=s.iteration,
                        status=s.status,
                        created_at=s.created_at,
                        updated_at=s.updated_at,
                        attribution_tags=dict(s.attribution_tags) if s.attribution_tags else None,
                    )
                )
            metas.sort(key=lambda m: m.updated_at, reverse=True)
            return metas[:limit]

    async def delete_checkpoint(self, run_id: str) -> None:
        async with self._lock:
            self._runs.pop(run_id, None)

    async def teardown(self) -> None:
        self._runs.clear()
