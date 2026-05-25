# Budget Enforcement (Phase 6A.3)

Hard caps per scope on LLM tokens, USD cost, tool calls, LLM calls, and
wall-clock time. When a cap is hit, the framework raises
`BudgetExceeded` so a runaway agent fails loud instead of burning the
month's API budget.

## Quick setup

```python
from genie_tooling.budget import BudgetSpec
from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig

cfg = MiddlewareConfig(
    features=FeatureSettings(
        budget_enforcer="in_memory_budget_enforcer",
        token_usage_recorder="in_memory_token_recorder",
    ),
)
genie = await Genie.create(config=cfg)
```

## Setting a cap

```python
await genie.budget.set_budget(
    "incident:SEV2-1234",
    BudgetSpec(
        max_tokens=50_000,
        max_cost_usd=2.00,
        max_tool_calls=30,
        max_llm_calls=20,
        max_wall_clock_seconds=600.0,
    ),
)
```

Any field left `None` = no cap. Multiple caps stack — the first one to
overflow triggers `BudgetExceeded`.

## Charging the scope

The framework charges automatically once the scope is supplied per call.

### From direct LLM calls

```python
try:
    response = await genie.llm.chat(
        messages=[...],
        budget_scope="incident:SEV2-1234",
        attribution_tags={"incident": "SEV2-1234"},
    )
except BudgetExceeded as e:
    # e.scope, e.limit, e.current, e.reason
    logger.error(f"Budget refused the call: {e}")
```

### From an agent run

Pass it once in `input_context` — `ReActAgent` threads it to every LLM
call AND every `genie.execute_tool` call inside the loop:

```python
result = await react_agent.run(
    goal="...",
    input_context={
        "budget_scope": "incident:SEV2-1234",
        "attribution_tags": {"incident": "SEV2-1234"},
    },
)
```

### From tool calls

```python
result = await genie.execute_tool(
    "some_tool",
    context={"budget_scope": "incident:SEV2-1234"},
    **params,
)
```

## Inspecting usage

```python
snap = await genie.budget.get_usage("incident:SEV2-1234")
if snap:
    print(f"tokens={snap.tokens} cost_usd={snap.cost_usd:.2f}")
    print(f"by provider: {snap.by_provider}")
    print(f"wall clock: {snap.wall_clock_seconds:.1f}s")
```

## Multi-tenant pattern

Use any free-form string as the scope. Conventions that work well:

* `"session:<sid>"` — per user session
* `"tenant:<tid>"` — per tenant
* `"incident:<incident_id>"` — per incident triage
* `"pr:<pr_id>"` — per PR review session
* `"global"` — fallback used by `InMemoryBudgetEnforcerPlugin` when no
  per-scope spec is set (configure via `global_spec` in the plugin config).

## Backend choice

* **`in_memory_budget_enforcer_v1`** — single-process. Suitable for
  tests and single-worker deployments.
* For multi-worker production, write a Redis- or Postgres-backed
  enforcer against the same `BudgetEnforcerPlugin` protocol. The cap
  semantics are unchanged; only the storage moves out of memory.
