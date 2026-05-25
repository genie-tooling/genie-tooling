"""E30 — Multi-server MCP ingestion with side-effect overlays (Phase 6B.1+6B.2).

Configures the MCP composition layer with two servers — filesystem (read/write)
and (optionally) Slack — applies bundled overlays so destructive operations
require approval, and routes everything through the Claude-Code permission
model.

Requires: poetry install --extras mcp.
"""
import asyncio
import logging
import os
from pathlib import Path

from genie_tooling.config.features import FeatureSettings
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.genie import Genie


async def main():
    logging.basicConfig(level=logging.INFO)

    overlays_dir = (
        Path(__file__).parent.parent
        / "src"
        / "genie_tooling"
        / "tools"
        / "impl"
        / "mcp_overlays"
    )

    cfg = MiddlewareConfig(
        environment="development",
        features=FeatureSettings(
            llm="ollama",
            llm_ollama_model_name="qwen3.6:35b",
            hitl_approver="claude_code_permissions",
        ),
        extension_configurations={
            "mcp_composition": {
                "overlays_dir": str(overlays_dir),
                "servers": [
                    {
                        "name": "filesystem",
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                    },
                    # Uncomment + set SLACK_BOT_TOKEN to add Slack:
                    # {
                    #     "name": "slack",
                    #     "command": "npx",
                    #     "args": ["-y", "@modelcontextprotocol/server-slack"],
                    #     "env": {"SLACK_BOT_TOKEN": os.environ.get("SLACK_BOT_TOKEN", "")},
                    # },
                ],
            },
        },
    )
    genie = await Genie.create(config=cfg)
    try:
        # The composition layer registers tools onto the framework; list them.
        tools = await genie.tools.list()
        print(f"\nMCP tools discovered ({len(tools)}):")
        for t in sorted(tools, key=lambda t: t.identifier):
            md = await t.get_metadata()
            print(f"  - {t.identifier}: side_effects={md.get('side_effects')}, "
                  f"requires_approval={md.get('requires_approval')}")
    finally:
        await genie.close()


if __name__ == "__main__":
    asyncio.run(main())
