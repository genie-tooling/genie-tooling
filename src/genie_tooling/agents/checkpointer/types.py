"""Types for the agent checkpointer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

RunStatus = Literal["running", "completed", "failed", "aborted", "max_iterations", "user_stopped"]


@dataclass
class CheckpointState:
    """A single snapshot of an agent run.

    Persisted at every iteration boundary. ``state_blob`` is the
    agent-specific payload — for ReAct it's the scratchpad + iteration index;
    for PlanAndExecute it's the plan + step outputs. Round-trips through JSON.
    """

    run_id: str
    agent_id: str
    iteration: int
    goal: str
    state_blob: Dict[str, Any]
    status: RunStatus
    created_at: float
    updated_at: float
    attribution_tags: Optional[Dict[str, str]] = None
    user_identity: Optional[Dict[str, Any]] = None
    correlation_id: Optional[str] = None
    error: Optional[str] = None


@dataclass
class CheckpointMeta:
    """Lightweight metadata for run listings (without the full state_blob)."""

    run_id: str
    agent_id: str
    iteration: int
    status: RunStatus
    created_at: float
    updated_at: float
    attribution_tags: Optional[Dict[str, str]] = None
