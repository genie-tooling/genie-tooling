"""Phase 6A.2 — durable agent state.

ReActAgent / PlanAndExecuteAgent loops are process-local by default.
``AgentCheckpointerPlugin`` lets long-running flows persist their scratchpad
to a durable store and resume after process restart — critical for SRE on-call
playbooks that may run 30+ minutes and need to survive worker rescheduling.

The package ships two implementations:

* ``in_memory_agent_checkpointer_v1`` — dict-backed, useful for tests and
  single-process dev.
* ``sqlite_agent_checkpointer_v1`` — stdlib ``sqlite3`` over
  ``asyncio.to_thread``. Suitable for single-host production runs and as
  a stepping-stone toward Postgres for multi-worker deployments.

A Postgres-backed implementation (``postgres_agent_checkpointer_v1``) is the
intended production target — designed but not bundled here. It requires the
``asyncpg`` driver and is therefore opt-in via a future ``postgres`` extra.
"""
from .abc import AgentCheckpointerPlugin
from .types import CheckpointMeta, CheckpointState, RunStatus

__all__ = ["AgentCheckpointerPlugin", "CheckpointMeta", "CheckpointState", "RunStatus"]
