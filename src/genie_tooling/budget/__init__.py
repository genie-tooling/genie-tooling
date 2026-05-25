"""Phase 6A.3 — Budget enforcement.

The token_usage subsystem records LLM cost passively. ``budget`` adds
*enforcement*: hard caps per session/tenant on tokens, cost, tool calls,
and wall-clock. When a cap is hit, the framework raises
``BudgetExceeded`` so a runaway agent fails loud instead of burning the
month's API budget.

Plugins implement ``BudgetEnforcerPlugin``. Caps are checked at every
LLM call and tool execution; the in-memory backend ships by default.
"""
from .abc import BudgetEnforcerPlugin
from .exceptions import BudgetExceeded
from .types import BudgetSnapshot, BudgetSpec

__all__ = ["BudgetEnforcerPlugin", "BudgetExceeded", "BudgetSnapshot", "BudgetSpec"]
