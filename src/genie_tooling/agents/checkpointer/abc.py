"""Abstract protocol for the agent checkpointer."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from genie_tooling.core.types import Plugin

from .types import CheckpointMeta, CheckpointState


@runtime_checkable
class AgentCheckpointerPlugin(Plugin, Protocol):
    """Persists agent run state to durable storage.

    Implementations must be safe under concurrent access from a single process
    (Postgres-backed implementations should additionally be safe across worker
    processes via row-level locks or advisory locks).
    """

    async def save_checkpoint(self, state: CheckpointState) -> None:
        """Insert-or-update the checkpoint keyed by ``state.run_id``.

        Subsequent saves for the same ``run_id`` MUST replace the prior row /
        record. Implementations are free to keep a history table for audit;
        this method's contract is only "latest state is now this."
        """
        ...

    async def load_checkpoint(self, run_id: str) -> Optional[CheckpointState]:
        """Return the latest state for the run, or None if unknown."""
        ...

    async def list_runs(
        self,
        agent_id: Optional[str] = None,
        status: Optional[str] = None,
        attribution_tag: Optional[Dict[str, str]] = None,
        limit: int = 100,
    ) -> List[CheckpointMeta]:
        """List recent runs. ``attribution_tag`` filter matches if ALL provided
        key/value pairs are present in the run's attribution_tags dict."""
        ...

    async def delete_checkpoint(self, run_id: str) -> None:
        """Permanently remove a run's checkpoint(s)."""
        ...
