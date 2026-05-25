# MCP Overlay Catalog

Phase 6B.2 — curated YAML overlays for popular official MCP servers.

Each file declares per-tool metadata that the upstream MCP server can't
self-report: `side_effects` classification, `requires_approval` policy
hints, response redaction paths, cacheability, etc.

To use one:

```python
from genie_tooling.config.models import MiddlewareConfig

cfg = MiddlewareConfig(
    extension_configurations={
        "mcp_composition": {
            "overlays_dir": "/path/to/this/dir",
            "servers": [
                {
                    "name": "slack",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-slack"],
                    "env": {"SLACK_BOT_TOKEN": "xoxb-..."},
                },
                # ...
            ],
        }
    },
)
```

The composition plugin looks up `slack.yml` in the overlays directory
and applies the overlay to each discovered Slack MCP tool. You can also
inline overlays per-server in the config — inline beats file.

## Side-effects vocabulary

* `none` — pure computation; no I/O.
* `read` — external read only; safe to auto-allow in most cases.
* `write` — external write that's reversible or scoped (post a message,
  create a draft).
* `destructive` — irreversible or high-blast-radius (delete a resource,
  drop a table, force-merge a PR).

## Editing overlays

These overlays are *opinionated defaults*. If your team needs stricter
or looser policy, copy a file out to your repo and reference it via
`overlays_dir` — `MCPCompositionPlugin` reads the dir at startup.

Default tone:
- **Optimistic on reads**: list / get / search → `read` + idempotent.
- **Conservative on writes**: anything that mutates an external system →
  `write` + `requires_approval: true` unless we can identify a sub-scope
  that's clearly safe.
- **Always-deny-worth-flagging on destructive**: drop / delete / force
  → `destructive` + `requires_approval: true`.

Tools we don't know about default to `side_effects: unknown` which the
permissions plugin treats as `ask`.

## Catalog

The shipped overlays are version-pinned against each upstream MCP server.
When the upstream server changes (adds tools, renames them), the
overlay may go out of date — the policy plugin will then fall back to
`side_effects: unknown` for unrecognized tool names, which is the safe
default. Pin a known-good upstream version when this matters.
