"""Abstract protocol for budget-enforcement plugins."""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from genie_tooling.core.types import Plugin

from .types import BudgetSnapshot, BudgetSpec


@runtime_checkable
class BudgetEnforcerPlugin(Plugin, Protocol):
    """Phase 6A.3 — budget enforcement.

    Implementations track usage by scope (free-form string identifier — typically
    ``"session:<sid>"``, ``"tenant:<tid>"``, ``"global"``, or a custom tag) and
    refuse further work when a cap is hit.

    Required methods:

    * ``set_budget(scope, spec)`` — install (or replace) caps for the scope.
    * ``check_and_charge_llm_call(scope, prompt_tokens, completion_tokens, provider_id, cost_usd)`` —
      called by ``LLMInterface`` after a successful LLM round-trip. Raises
      :class:`BudgetExceeded` if the call pushes any cap over.
    * ``check_and_charge_tool_call(scope)`` — called before / after
      ``genie.execute_tool``. Raises if the count cap is hit.
    * ``get_usage(scope)`` — current snapshot. Pure read.
    * ``clear(scope)`` — reset accumulator (start a new run under the same scope).

    The framework treats ``scope=None`` calls as no-op so plugins remain
    optional — they only enforce when the caller surfaces a scope.
    """

    async def set_budget(self, scope: str, spec: BudgetSpec) -> None: ...

    async def check_and_charge_llm_call(
        self,
        scope: str,
        prompt_tokens: int,
        completion_tokens: int,
        provider_id: str,
        cost_usd: float = 0.0,
    ) -> None:
        """Charge an LLM call. Raises ``BudgetExceeded`` if it overruns."""
        ...

    async def check_and_charge_tool_call(self, scope: str) -> None:
        """Charge one tool execution. Raises ``BudgetExceeded`` if over."""
        ...

    async def get_usage(self, scope: str) -> Optional[BudgetSnapshot]:
        """Return the current snapshot or None if scope is unknown."""
        ...

    async def clear(self, scope: str) -> None: ...
