# Agent Checkpointing (Phase 6A.2)

Long-running agents shouldn't lose all progress to a worker restart or
pod reschedule. `AgentCheckpointerPlugin` persists agent scratchpad
state to a durable store after every iteration. A crashed run resumes
from the last completed iteration.

Two bundled implementations:

* **`in_memory_agent_checkpointer_v1`** — dict-backed. Useful for tests
  and single-process dev. **Does not survive process restart.**
* **`sqlite_agent_checkpointer_v1`** — stdlib `sqlite3` over
  `asyncio.to_thread`. Survives restart on a single host.
* (Postgres backend is the intended production target for multi-worker
  deployments; designed but not shipped — write the plugin against
  the same `AgentCheckpointerPlugin` protocol.)

## Quick setup

```python
from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig

cfg = MiddlewareConfig(
    features=FeatureSettings(
        agent_checkpointer="sqlite_agent_checkpointer",
    ),
    agent_checkpointer_configurations={
        "sqlite_agent_checkpointer_v1": {
            "db_path": "/var/lib/genie/checkpoints.sqlite",
        },
    },
)
genie = await Genie.create(config=cfg)
```

## Saving — happens automatically

`ReActAgent` (both regex and native loops) saves a `CheckpointState`
at every iteration boundary while configured. The agent's run id
defaults to the same UUID as the correlation id, so audit + checkpoint
queries share a key.

State blob shape (for ReAct):

```python
{
    "scratchpad": [
        {"thought": "...", "action": "...", "observation": "..."},
        ...
    ],
    "mode": "react",
}
```

Status transitions: `running` → (`completed` | `max_iterations` | `failed`).

## Resuming a crashed run

Pass `resume_from_run_id` to `agent.run(...)`:

```python
agent = ReActAgent(genie=genie, agent_config={"use_native_tool_use": True})
result = await agent.run(
    goal="continue investigation",
    resume_from_run_id="abc-123-...",
)
```

If the checkpointer has state for that run, the agent loads the
scratchpad, sets its run id to the resume target, and continues from
the next iteration. Final state is saved under the same run id so the
audit history is unbroken.

## Querying runs

```python
runs = await genie.checkpointer.list_runs(
    agent_id="react_agent",
    status="running",
    attribution_tag={"team": "platform"},
)
for r in runs:
    print(r.run_id, r.iteration, r.updated_at)

# Drill into one
state = await genie.checkpointer.load_checkpoint(runs[0].run_id)
```

Common operational pattern: a "resume orphaned runs" sweep at process
startup picks up any `status="running"` runs whose `updated_at` is older
than the expected iteration time, and re-invokes them with `resume_from_run_id`.

## What's NOT persisted

* The conversation state (use `genie.conversation` for that).
* In-flight tool calls — if a tool call is interrupted, the resumed run
  will re-execute its own decision about what to do next given the
  scratchpad. Idempotent tools handle this gracefully; non-idempotent
  tools should be gated by the permission model (`requires_approval: true`).
* Cached embeddings, RAG state, etc.

## Comparison with conversation forking

`genie.conversation.fork(session_id)` clones a *conversation* (user/
assistant messages) into a new session id — for **parallel investigation
branches**. Checkpointing persists an *agent's internal scratchpad* —
for **resuming after crash**. The two are orthogonal.
