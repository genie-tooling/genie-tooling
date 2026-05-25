# E31 — Budget Enforcement and Audit Ledger

Source: [`examples/E31_budget_and_audit.py`](https://github.com/genie-tooling/genie-tooling/blob/main/examples/E31_budget_and_audit.py)

End-to-end: per-session budget cap on tokens / cost / LLM calls, plus a
durable HITL approval ledger keyed by `decision_id`.

See [Budget guide](../guides/budget.md) for the full reference.

## What it shows

* `features.budget_enforcer="in_memory_budget_enforcer"` enables hard
  caps via the `genie.budget` interface.
* `default_hitl_ledger_id="sqlite_hitl_ledger_v1"` persists every HITL
  approval decision joinable by `decision_id`.
* `await genie.budget.set_budget("incident:SEV2-1234", BudgetSpec(...))`
  installs the caps.
* `await genie.llm.chat(..., budget_scope="incident:SEV2-1234",
  attribution_tags={"incident": "SEV2-1234"})` makes the framework
  charge the scope; `BudgetExceeded` raises past the cap.

## Required setup

```bash
poetry install
```

(Local Ollama is used for the LLM call in the demo — adjust the model
name in the script to one you have running locally.)

## Run

```bash
poetry run python examples/E31_budget_and_audit.py
```

The script prints token / call totals for the scope after the run. Look
at the SQLite ledger file (path printed by the script) to see the
captured HITL decisions.
