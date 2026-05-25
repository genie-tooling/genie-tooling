"""Budget enforcement exceptions."""
from __future__ import annotations


class BudgetExceeded(Exception):
    """Raised by a BudgetEnforcerPlugin when a configured cap is hit.

    Attributes:
        scope: the cap identifier that was hit (e.g. "session:abc:tokens").
        limit: the cap value.
        current: the current usage at the moment of refusal.
        reason: human-readable description for logs and audit records.
    """

    def __init__(self, scope: str, limit: float, current: float, reason: str = ""):
        self.scope = scope
        self.limit = limit
        self.current = current
        self.reason = reason or f"budget exceeded for {scope}: {current}/{limit}"
        super().__init__(self.reason)
