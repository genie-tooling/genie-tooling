"""Phase 6C.10 — ``python -m genie_tooling.mcp_server`` console-script.

Bootstraps a Genie instance from a YAML config and runs the MCP server
bootstrap, exposing the configured tools to MCP clients (Claude Desktop,
IDE plugins, other agents) over stdio.

Usage::

    python -m genie_tooling.mcp_server --config mcp_serve.yml

Where ``mcp_serve.yml`` is a YAML serialization of
:class:`MiddlewareConfig` with ``extension_configurations.mcp_server.enabled``
set to True.

Minimal example config::

    environment: production
    auto_enable_registered_tools: false
    features:
      llm: anthropic
      llm_anthropic_model_name: claude-sonnet-4-6
    tool_configurations:
      calculator_tool: {}
      sandboxed_fs_tool_v1:
        sandbox_base_path: /var/genie/sandbox
    extension_configurations:
      mcp_server:
        enabled: true
        server_name: my-team-genie
        transport: stdio

You can also point Claude Desktop or any MCP-aware client at this script:

    {
      "mcpServers": {
        "my-team-genie": {
          "command": "python",
          "args": ["-m", "genie_tooling.mcp_server", "--config", "/etc/genie/mcp.yml"]
        }
      }
    }
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

logger = logging.getLogger("genie_tooling.mcp_server")


async def _serve(config_path: Path) -> int:
    try:
        import yaml
    except ImportError:
        print("Error: PyYAML is required to load a YAML config.", file=sys.stderr)
        return 2

    from genie_tooling.config.models import MiddlewareConfig
    from genie_tooling.genie import Genie

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        return 2
    except yaml.YAMLError as e:
        print(f"Error parsing YAML config: {e}", file=sys.stderr)
        return 2

    try:
        cfg = MiddlewareConfig.model_validate(raw)
    except Exception as e:
        print(f"Error validating MiddlewareConfig: {e}", file=sys.stderr)
        return 2

    # Force the MCP server bootstrap on if missing — the user might forget
    # but they ran this script for a reason.
    ext = cfg.extension_configurations.setdefault("mcp_server", {})
    if "enabled" not in ext:
        ext["enabled"] = True

    logger.info(f"Starting Genie + MCP server with config {config_path}...")
    genie = await Genie.create(config=cfg)

    # The MCP server bootstrap stashes its server task on the Genie facade.
    task = getattr(genie, "_mcp_server_task", None)
    if task is None:
        print(
            "Error: MCP server task did not start. Ensure mcp_server_bootstrap_v1 "
            "is registered and extension_configurations.mcp_server.enabled=True.",
            file=sys.stderr,
        )
        await genie.close()
        return 3

    # Forward SIGINT / SIGTERM cleanly.
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def _on_signal(*_: object) -> None:
        logger.info("Shutdown signal received; tearing down...")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _on_signal)
        except NotImplementedError:
            # Windows / restricted envs — fall back to no handler.
            pass

    try:
        # Run until either the server task completes or a shutdown signal fires.
        done, _pending = await asyncio.wait(
            [task, asyncio.create_task(shutdown_event.wait())],
            return_when=asyncio.FIRST_COMPLETED,
        )
        if task in done and not task.cancelled():
            exc = task.exception()
            if exc:
                logger.error(f"MCP server task ended with error: {exc}", exc_info=exc)
                await genie.close()
                return 1
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        await genie.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="genie_tooling.mcp_server",
        description="Run Genie tools as an MCP stdio server.",
    )
    parser.add_argument("--config", "-c", required=True, type=Path, help="Path to MiddlewareConfig YAML.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default INFO).")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,  # CRITICAL: stdout is reserved for the MCP stdio transport.
    )

    return asyncio.run(_serve(args.config))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
