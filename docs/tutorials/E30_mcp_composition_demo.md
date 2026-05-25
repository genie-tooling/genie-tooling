# E30 — Multi-Server MCP Composition

Source: [`examples/E30_mcp_composition_demo.py`](https://github.com/genie-tooling/genie-tooling/blob/main/examples/E30_mcp_composition_demo.py)

Configures the MCP composition layer with one or more official MCP
servers, applies the bundled overlay catalog, and lists what tools
appeared in the unified policy-controlled surface.

See [MCP Composition guide](../guides/mcp_composition.md) for the full
configuration reference.

## What it shows

* `extension_configurations.mcp_composition.servers` lists the MCP
  servers to start as subprocesses (filesystem ships enabled; Slack is
  commented out — uncomment + set `SLACK_BOT_TOKEN` to enable).
* `overlays_dir` points at the bundled overlay catalog; each
  `<server>.yml` declares per-tool side-effects.
* After `Genie.create()`, `genie.tools.list()` includes every
  discovered MCP tool with the right `side_effects` metadata applied.

## Required setup

```bash
poetry install --extras mcp
# Optional, only if you uncomment the Slack server:
export SLACK_BOT_TOKEN=xoxb-...
```

## Run

```bash
poetry run python examples/E30_mcp_composition_demo.py
```

Output: a list of MCP tool identifiers with `side_effects` /
`requires_approval` annotations.
