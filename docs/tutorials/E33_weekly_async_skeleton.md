# E33 — Weekly Async Planner Skeleton

Source: [`examples/E33_weekly_async_skeleton.py`](https://github.com/genie-tooling/genie-tooling/blob/main/examples/E33_weekly_async_skeleton.py)

Wires the full corporate-harness stack for the Weekly Async planning
use case (the "first production use case" in
[PHASE_6_PLAN.md §9](https://github.com/genie-tooling/genie-tooling/blob/main/PHASE_6_PLAN.md)).

This example is the **configuration template**, not the agent itself.
The actual ReActAgent loop that calls Linear / Slack / Notion is left
as a thin Python wrapper around `agent.run(...)` — see the standalone
**`docs/WEEKLY_ASYNC_AGENT.md`** plan document for the deployment
runbook.

## What's wired

* **`features.hitl_approver_chain=["claude_code_permissions", "webhook_approval_v1"]`**
  — permission plugin in front, webhook as the human fallback.
* **`features.budget_enforcer="in_memory_budget_enforcer"`** — hard caps
  per team-run scope.
* **`features.agent_checkpointer="sqlite_agent_checkpointer"`** —
  durable scratchpad so a crashed worker resumes from the last iteration.
* **`features.progress_sinks=["webhook_progress_sink"]`** —
  thread-progress updates to a Slack interactive message.
* **`extension_configurations.mcp_composition`** — Linear, Slack,
  Notion MCP servers ingested with bundled overlays.
* **Permission policy** auto-allows `mcp_notion_create_page` under
  the `/team-plans/` parent and `slack_post_message` to `#*-async`
  channels; asks for approval on everything else write/destructive.
* **Webhook routing** sends destructive to PagerDuty oncall, Notion
  writes to team approvers, default to a general channel.

## Required setup

```bash
poetry install --extras "mcp anthropic"
export ANTHROPIC_API_KEY=sk-ant-...
export SLACK_BOT_TOKEN=xoxb-...
export LINEAR_API_KEY=lin_...
export NOTION_API_KEY=secret_...
```

## Run

```bash
poetry run python examples/E33_weekly_async_skeleton.py
```

The script lists the discovered MCP tools (~30 across the 3 servers)
and runs a placeholder per-team loop. Replace the placeholder with
your actual ReActAgent invocation.
