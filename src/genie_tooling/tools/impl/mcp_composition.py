"""MCPCompositionPlugin — multi-server MCP ingestion with overlay registry.

Phase 6B.1: connects to N MCP servers, applies side-effect overlays to the
discovered tools, and exposes them all under a single composition plugin.
This is the "corporate MCP gateway" pattern — ingest official servers (Slack,
GitHub, Notion, Linear, AWS-API), apply HITL/audit/redaction overlays, and
expose a unified policy-controlled tool surface.

Config::

    extension_configurations:
      mcp_composition:
        # Optional path to a directory of overlay YAML files. Each filename
        # matches a server name (e.g. `slack.yml`, `github.yml`).
        overlays_dir: "./mcp_overlays"

        # Or inline overlays per server.
        servers:
          - name: slack
            command: npx
            args: ["-y", "@modelcontextprotocol/server-slack"]
            env: {SLACK_BOT_TOKEN: "xoxb-..."}
            # Inline overlay (overrides overlays_dir if both set).
            overlays:
              postMessage:
                side_effects: write
                requires_approval: true
                redact_response_fields: ["content.user.email"]
              listChannels:
                side_effects: read
                idempotent: true
                cacheable: true
                cache_ttl_seconds: 60
              "*":
                # default for any tool not specifically named above
                side_effects: unknown

          - name: github
            command: npx
            args: ["-y", "@modelcontextprotocol/server-github"]
            # Pulls overlays from overlays_dir/github.yml

After setup, all discovered tools are available under
``genie.tools.list()`` with namespaced identifiers
(``mcp_slack_postMessage``, ``mcp_github_createIssue``, etc.) and overlay
metadata applied for policy / approval / redaction.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from genie_tooling.core.types import Plugin
from genie_tooling.tools.impl.mcp_client_tool import (
    MCP_AVAILABLE,
    MCPClientToolPlugin,
    MCPRemoteTool,
)

logger = logging.getLogger(__name__)


class MCPCompositionPlugin(Plugin):
    plugin_id: str = "mcp_composition_v1"
    description: str = (
        "Multi-server MCP ingestion with side-effect overlay registry. "
        "The 'corporate MCP gateway' pattern: ingest N official MCP servers, "
        "apply HITL/audit overlays, expose a policy-controlled tool surface."
    )

    _client_plugins: List[MCPClientToolPlugin]
    _all_tools: List[MCPRemoteTool]

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._client_plugins = []
        self._all_tools = []

        if not MCP_AVAILABLE:
            logger.error(
                f"{self.plugin_id}: `mcp` SDK not installed. Install with the "
                f"'mcp' extra: `pip install genie-tooling[mcp]`."
            )
            return

        overlays_dir = cfg.get("overlays_dir")
        overlays_dir_path = Path(overlays_dir) if overlays_dir else None
        server_configs: List[Dict[str, Any]] = list(cfg.get("servers") or [])

        for server_cfg in server_configs:
            if not isinstance(server_cfg, dict):
                continue
            name = server_cfg.get("name", "default")
            # Compose the per-server config for MCPClientToolPlugin.
            client_cfg: Dict[str, Any] = {
                "name": name,
                "command": server_cfg.get("command"),
                "args": server_cfg.get("args", []),
                "env": server_cfg.get("env"),
            }
            # Resolve overlays: inline beats file
            inline_overlays = server_cfg.get("overlays")
            if inline_overlays is not None:
                client_cfg["overlays"] = inline_overlays
            elif overlays_dir_path is not None:
                loaded = _load_overlay_file(overlays_dir_path / f"{name}.yml")
                if loaded:
                    client_cfg["overlays"] = loaded

            client_plugin = MCPClientToolPlugin()
            try:
                await client_plugin.setup(client_cfg)
            except Exception as e:
                logger.error(
                    f"{self.plugin_id}: failed to set up MCP server {name!r}: {e}",
                    exc_info=True,
                )
                continue
            self._client_plugins.append(client_plugin)
            self._all_tools.extend(client_plugin.tools)

        logger.info(
            f"{self.plugin_id}: ingested {len(self._all_tools)} tool(s) across "
            f"{len(self._client_plugins)} MCP server(s)."
        )

    @property
    def tools(self) -> List[MCPRemoteTool]:
        return self._all_tools

    async def teardown(self) -> None:
        for cp in self._client_plugins:
            try:
                await cp.teardown()
            except Exception:
                logger.warning(f"{self.plugin_id}: error tearing down {cp._server_name}", exc_info=True)
        self._client_plugins.clear()
        self._all_tools.clear()


def _load_overlay_file(path: Path) -> Optional[Dict[str, Any]]:
    """Read a YAML overlay file. Returns None if missing or unreadable."""
    if not path.is_file():
        return None
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            logger.warning(f"MCP overlay file {path} is not a dict; skipping.")
            return None
        # Overlays can be nested under "overlays:" or be a flat dict
        return data.get("overlays", data)
    except Exception as e:
        logger.error(f"Failed to read MCP overlay file {path}: {e}", exc_info=True)
        return None
