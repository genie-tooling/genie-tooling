"""InMemoryBudgetEnforcerPlugin: thread-safe in-process budget enforcement.

Suitable for single-process deployments and tests. For multi-process /
multi-worker deployments swap to a Redis-backed or Postgres-backed
enforcer (one of those ships as a follow-up plugin).
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Optional

from genie_tooling.budget.abc import BudgetEnforcerPlugin
from genie_tooling.budget.exceptions import BudgetExceeded
from genie_tooling.budget.types import BudgetSnapshot, BudgetSpec

logger = logging.getLogger(__name__)


class InMemoryBudgetEnforcerPlugin(BudgetEnforcerPlugin):
    plugin_id: str = "in_memory_budget_enforcer_v1"
    description: str = (
        "In-process budget enforcement. Suitable for single-process deployments "
        "and tests. Caps per scope on tokens / cost / tool-calls / LLM-calls / "
        "wall-clock; raises BudgetExceeded when a cap is hit."
    )

    _specs: Dict[str, BudgetSpec]
    _snaps: Dict[str, BudgetSnapshot]
    _lock: asyncio.Lock

    async def setup(self, config: Optional[Dict[str, object]] = None) -> None:
        self._specs = {}
        self._snaps = {}
        self._lock = asyncio.Lock()
        cfg = config or {}
        # Optional default cap applied to any scope without a more-specific
        # spec; convenient for "no scope wired up but I want a global cap".
        global_spec = cfg.get("global_spec") or {}
        if isinstance(global_spec, dict) and global_spec:
            self._specs["global"] = BudgetSpec(**{k: v for k, v in global_spec.items() if k in BudgetSpec.__dataclass_fields__})

    async def set_budget(self, scope: str, spec: BudgetSpec) -> None:
        async with self._lock:
            self._specs[scope] = spec

    async def check_and_charge_llm_call(
        self,
        scope: str,
        prompt_tokens: int,
        completion_tokens: int,
        provider_id: str,
        cost_usd: float = 0.0,
    ) -> None:
        async with self._lock:
            snap = self._snap_locked(scope)
            tokens = (prompt_tokens or 0) + (completion_tokens or 0)
            snap.tokens += tokens
            snap.cost_usd += float(cost_usd or 0.0)
            snap.llm_calls += 1
            snap.by_provider[provider_id] = snap.by_provider.get(provider_id, 0) + tokens
            snap.last_event_at = time.time()
            if snap.started_at is None:
                snap.started_at = snap.last_event_at
            self._enforce_locked(scope, snap)

    async def check_and_charge_tool_call(self, scope: str) -> None:
        async with self._lock:
            snap = self._snap_locked(scope)
            snap.tool_calls += 1
            snap.last_event_at = time.time()
            if snap.started_at is None:
                snap.started_at = snap.last_event_at
            self._enforce_locked(scope, snap)

    async def get_usage(self, scope: str) -> Optional[BudgetSnapshot]:
        async with self._lock:
            return self._snaps.get(scope)

    async def clear(self, scope: str) -> None:
        async with self._lock:
            self._snaps.pop(scope, None)

    async def teardown(self) -> None:
        self._specs.clear()
        self._snaps.clear()

    # --- helpers (assume lock held) ---

    def _snap_locked(self, scope: str) -> BudgetSnapshot:
        snap = self._snaps.get(scope)
        if snap is None:
            snap = BudgetSnapshot(scope=scope)
            self._snaps[scope] = snap
        return snap

    def _enforce_locked(self, scope: str, snap: BudgetSnapshot) -> None:
        spec = self._specs.get(scope) or self._specs.get("global")
        if spec is None:
            return
        if spec.max_tokens is not None and snap.tokens > spec.max_tokens:
            raise BudgetExceeded(
                scope=f"{scope}:tokens",
                limit=float(spec.max_tokens),
                current=float(snap.tokens),
                reason=f"token cap reached ({snap.tokens}>{spec.max_tokens}) for scope={scope}",
            )
        if spec.max_cost_usd is not None and snap.cost_usd > spec.max_cost_usd:
            raise BudgetExceeded(
                scope=f"{scope}:cost_usd",
                limit=spec.max_cost_usd,
                current=snap.cost_usd,
                reason=f"USD cap reached (${snap.cost_usd:.2f}>${spec.max_cost_usd:.2f}) for scope={scope}",
            )
        if spec.max_tool_calls is not None and snap.tool_calls > spec.max_tool_calls:
            raise BudgetExceeded(
                scope=f"{scope}:tool_calls",
                limit=float(spec.max_tool_calls),
                current=float(snap.tool_calls),
                reason=f"tool-call cap reached ({snap.tool_calls}>{spec.max_tool_calls}) for scope={scope}",
            )
        if spec.max_llm_calls is not None and snap.llm_calls > spec.max_llm_calls:
            raise BudgetExceeded(
                scope=f"{scope}:llm_calls",
                limit=float(spec.max_llm_calls),
                current=float(snap.llm_calls),
                reason=f"LLM-call cap reached ({snap.llm_calls}>{spec.max_llm_calls}) for scope={scope}",
            )
        if spec.max_wall_clock_seconds is not None and snap.wall_clock_seconds > spec.max_wall_clock_seconds:
            raise BudgetExceeded(
                scope=f"{scope}:wall_clock",
                limit=spec.max_wall_clock_seconds,
                current=snap.wall_clock_seconds,
                reason=f"wall-clock cap reached ({snap.wall_clock_seconds:.1f}s>{spec.max_wall_clock_seconds:.1f}s) for scope={scope}",
            )
