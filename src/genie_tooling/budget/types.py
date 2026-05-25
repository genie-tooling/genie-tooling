"""Budget primitives."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class BudgetSpec:
    """Caps for one logical scope (a session, tenant, run, or "global").

    Any field left None = no cap. The first cap that gets exceeded triggers
    BudgetExceeded; checks are inexpensive enough that callers can declare
    multiple caps freely.
    """

    max_tokens: Optional[int] = None
    """Cumulative prompt + completion tokens."""

    max_cost_usd: Optional[float] = None
    """Cumulative LLM cost in USD (requires a per-model price table)."""

    max_tool_calls: Optional[int] = None
    """Total ``genie.execute_tool`` invocations under this scope."""

    max_llm_calls: Optional[int] = None
    """Total ``genie.llm.chat`` + ``generate`` invocations."""

    max_wall_clock_seconds: Optional[float] = None
    """Wall-clock budget from first observation to now."""


@dataclass
class BudgetSnapshot:
    """Current usage for one scope. Returned from ``BudgetEnforcerPlugin.get_usage``."""

    scope: str
    tokens: int = 0
    cost_usd: float = 0.0
    tool_calls: int = 0
    llm_calls: int = 0
    started_at: Optional[float] = None
    last_event_at: Optional[float] = None
    by_provider: Dict[str, int] = field(default_factory=dict)
    """Per-provider token totals (provider_id → tokens)."""

    @property
    def wall_clock_seconds(self) -> float:
        if self.started_at is None or self.last_event_at is None:
            return 0.0
        return max(0.0, self.last_event_at - self.started_at)
