"""HITL ledger types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class LedgerEntry:
    """A single HITL decision, ready to be persisted."""

    request_id: str
    decision_id: Optional[str]  # joins to DecisionRecord
    correlation_id: Optional[str]
    tool_id: Optional[str]
    params: Optional[Dict[str, Any]]
    tool_metadata: Optional[Dict[str, Any]]
    status: str  # approved / denied / ask_human / error / timeout / pending
    approver_id: Optional[str]
    reason: Optional[str]
    requested_at: float
    decided_at: float
    user_identity: Optional[Dict[str, Any]] = None
    attribution_tags: Optional[Dict[str, str]] = None
    extra: Optional[Dict[str, Any]] = None


@dataclass
class LedgerQuery:
    """Filter for ``HITLLedgerPlugin.search``. Any None field is unconstrained."""

    tool_id: Optional[str] = None
    status: Optional[str] = None
    approver_id_prefix: Optional[str] = None
    attribution_tag: Optional[Dict[str, str]] = None
    since: Optional[float] = None
    until: Optional[float] = None
    limit: int = 100
