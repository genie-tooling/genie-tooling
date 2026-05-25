"""Progress event types."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

ProgressLevel = Literal["info", "warning", "error"]
ProgressPhase = Literal[
    "started", "iterating", "tool_call", "tool_result", "thinking", "completed", "failed"
]


@dataclass
class ProgressEvent:
    """A single progress update from an agent run."""

    run_id: str
    agent_id: str
    phase: ProgressPhase
    message: str
    timestamp: float
    iteration: Optional[int] = None
    level: ProgressLevel = "info"
    tool_id: Optional[str] = None
    attribution_tags: Optional[Dict[str, str]] = None
    extra: Optional[Dict[str, Any]] = None
