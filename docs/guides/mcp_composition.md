# MCP Composition (Phase 6B.1)

The composition layer ingests N official MCP servers, applies
side-effect overlays, and exposes the discovered tools as a single
policy-controlled tool surface. Think of it as a **corporate MCP
gateway**: every external integration goes through your audit + HITL
+ redaction layer before it reaches the agent.

## Quick setup

```python
from pathlib import Path
from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig

# Use the bundled overlay catalog (Phase 6B.2):
OVERLAYS = (
    Path(__file__).parent / ".." / ".." / "src"
    / "genie_tooling" / "tools" / "impl" / "mcp_overlays"
).resolve()

cfg = MiddlewareConfig(
    features=FeatureSettings(
        hitl_approver_chain=["claude_code_permissions", "webhook_approval_v1"],
    ),
    extension_configurations={
        "mcp_composition": {
            "overlays_dir": str(OVERLAYS),
            "servers": [
                {
                    "name": "slack",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-slack"],
                    "env": {"SLACK_BOT_TOKEN": "xoxb-..."},
                },
                {
                    "name": "notion",
                    "command": "npx",
                    "args": ["-y", "@notionhq/notion-mcp-server"],
                    "env": {"NOTION_API_KEY": "secret_..."},
                },
                {
                    "name": "linear",
                    "command": "npx",
                    "args": ["-y", "@linear/mcp-server"],
                    "env": {"LINEAR_API_KEY": "lin_..."},
                },
            ],
        },
    },
)
genie = await Genie.create(config=cfg)
# Discovered tools are now in genie.tools.list().
```

Each MCP tool surfaces as `mcp_<server_name>_<tool_name>` (e.g.
`mcp_slack_postMessage`).

## Overlay files

Each `<server>.yml` in the overlays directory maps remote tool names
→ metadata patches:

```yaml
overlays:
  postMessage:
    side_effects: write
    requires_approval: true
    redact_response_fields:
      - content.messages.user.email   # scrub before the LLM sees it
  listChannels:
    side_effects: read
    idempotent: true
    cacheable: true
    cache_ttl_seconds: 300
  "*":
    side_effects: unknown    # catch-all for tools we haven't classified
```

The bundled catalog (`src/genie_tooling/tools/impl/mcp_overlays/`) ships
overlays for: Slack, GitHub, Notion, Linear, JIRA, AWS-API, Filesystem,
Postgres, Sentry, Datadog, Grafana, Prometheus, Google Drive, Gmail.
Copy any of them to your own directory if you want stricter or looser
policy.

## Side-effects vocabulary

* `none` — pure computation.
* `read` — external read only.
* `write` — external write, reversible or scoped.
* `destructive` — irreversible or high blast radius.
* `unknown` — fallback; the permission plugin treats this as "ask".

## Response redaction

`redact_response_fields` is a list of dotted paths into the response
that get scrubbed (replaced with `"[REDACTED]"`) before the LLM sees
them. Use `*` as a wildcard at any level.

```yaml
overlays:
  postMessage:
    redact_response_fields:
      - content.messages.user.email
      - content.messages.user.phone
      - content.team.access_token       # never leak the bot token back
      - "secrets.*"                     # all keys at this level
```

## Inline overlays (alternative to overlays_dir)

```yaml
servers:
  - name: custom
    command: ...
    overlays:
      do_thing:
        side_effects: write
        requires_approval: true
```

Inline beats file when both are present.

## Re-exporting tools via MCP

Pair the composition layer with the MCP server bootstrap to *re-export*
the curated, policy-controlled tool surface to other MCP clients
(Claude Desktop, IDE plugins). Run:

```bash
genie-mcp-serve --config corporate_gateway.yml
```

…with a config that enables both `mcp_composition` and
`extension_configurations.mcp_server.enabled=True`. The result is a
single MCP endpoint that fronts your audit + HITL layer.

## Linting overlays in CI

```bash
genie-lint src/genie_tooling/tools/impl/mcp_overlays/ --kind overlays
```

…catches:

* Invalid `side_effects` values
* `side_effects=destructive` paired with `requires_approval=false` (dangerous)
* Malformed YAML
