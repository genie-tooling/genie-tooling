# Permissions — Claude-Code-style HITL (Phase 6A.5b)

The **`claude_code_permissions_v1`** approver is a deterministic
three-tier (allow / ask / deny) permission model with glob match on tool
ID and parameters, modelled on the Claude Code permission system. It's
designed to sit at the front of a `HITLManager` chain: the policy
handles most requests deterministically and delegates the rest to a
human approver (webhook, CLI, etc.) via the `ask_human` status.

## Quick setup

```python
from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig

cfg = MiddlewareConfig(
    environment="production",
    features=FeatureSettings(
        # Chain: policy plugin first; webhook as the human fallback.
        hitl_approver_chain=["claude_code_permissions", "webhook_approval_v1"],
    ),
    hitl_approver_configurations={
        "claude_code_permissions_v1": {
            "policy_path": "./permissions.yml",
        },
        "webhook_approval_v1": {
            "default_url": "https://hooks.example.com/approve",
        },
    },
)
```

## permissions.yml structure

```yaml
defaults:
  # Decision when no rule matches AND the side-effects default also says "auto".
  on_no_match: ask        # allow / ask / deny

  # Decision for each side-effects category when no rule matches.
  side_effects_defaults:
    none: allow
    read: allow
    write: ask
    destructive: ask
    unknown: ask

# First-match-wins rules.
rules:
  # Explicit allow with exact match on parameters.
  - id: ALLOW_KUBECTL_READS
    match:
      tool_id: kubectl_tool_v1
      params_match:
        operation: get           # exact equality
    decision: allow

  # Glob match on a param value.
  - id: ALLOW_TEAM_PLAN_CREATE
    match:
      tool_id_in: ["mcp_notion_create_page"]
      params_match:
        parent_id: "team_plans_*"   # fnmatch glob
    decision: allow

  # Explicit deny that overrides defaults.
  - id: DENY_PRODUCTION_DELETES
    match:
      side_effects_in: ["destructive"]
      params_match:
        namespace: "prod-*"
    decision: deny

  # Ask the human approver in the chain.
  - id: ASK_ON_WRITES_OUTSIDE_TEAM_PLANS
    match:
      side_effects_in: ["write"]
    decision: ask
```

## Match keys

Top-level `match` dict:

| Key | Behaviour |
|---|---|
| `tool_id` | Exact match against `data_to_approve['tool_id']`. |
| `tool_id_in` | Membership test; supports `fnmatch` globs (`slack:list*`). |
| `side_effects` | Exact match against `tool_metadata['side_effects']`. |
| `side_effects_in` | Membership against `tool_metadata['side_effects']`. |
| `params_match` | Sub-match against `data_to_approve['params']` (see below). |
| `user_identity` | Sub-match against `request.context['user_identity']`. |

**`params_match`** — each key must be present in `params`; values match
exactly OR as `fnmatch` globs when the spec is a string containing
`*`, `?`, or `[`.

## Where tool metadata comes from

Tools declare `side_effects` / `requires_approval` / `idempotent` in
their `get_metadata()`. The `@tool` decorator accepts them as kwargs:

```python
@tool(side_effects="destructive", requires_approval=True, idempotent=False)
async def delete_namespace(name: str) -> dict:
    """..."""
```

`genie.run_command(...)`, `ReActAgent`, and `PlanAndExecuteAgent` all
populate `data_to_approve["tool_metadata"]` automatically before
calling HITL.

## Per-session always-allow

Wrapper UIs can register "always allow for this session" patterns —
the equivalent of Claude Code's per-session `Always allow this tool`
button:

```python
approver = await genie.plugins.get_instance("claude_code_permissions_v1")
approver.add_session_allow(
    session_id="s-42",
    tool_id="slack_post_message",
    params_match={"channel": "#engineering*"},
)
```

Session overrides are cleared with `approver.clear_session_allows(session_id)`.

## The `ask_human` status

When a rule emits `ask` (or a side-effects default resolves to `ask`),
the approver returns `status="ask_human"` instead of approved/denied.
`HITLManager` walks the chain — the next approver (typically a webhook)
gets the request. If the chain is exhausted with `ask_human`, the
manager downgrades to `denied` with a clear reason.
