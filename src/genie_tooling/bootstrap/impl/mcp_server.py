"""MCPServerBootstrapPlugin: exposes Genie's tools over the MCP protocol (M3).

Mirror image of MCPClientToolPlugin. When this bootstrap is enabled,
Genie spins up an MCP server (stdio transport) that surfaces every
registered Genie tool to external clients. Useful for letting Claude
Desktop, IDE plugins, or other agents discover and call your Genie
tools without you having to write a per-client adapter.

Configuration (under ``extension_configurations["mcp_server"]``)::

    enabled: bool (default False)
        Whether to actually start the server. Bootstrap is a no-op if false.
    transport: "stdio" (currently the only supported value)
    server_name: str (default "genie-tooling")
        Advertised to MCP clients as the server identity.

The server runs in the background (via asyncio.create_task) so the
bootstrap returns promptly. The task is attached to ``genie._mcp_server_task``
for teardown.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict

from genie_tooling.bootstrap.abc import BootstrapPlugin

if TYPE_CHECKING:
    from genie_tooling.genie import Genie

logger = logging.getLogger(__name__)

try:
    import mcp.types as mcp_types
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    MCP_AVAILABLE = True
except ImportError:  # pragma: no cover
    Server = None  # type: ignore
    stdio_server = None  # type: ignore
    mcp_types = None  # type: ignore
    MCP_AVAILABLE = False


class MCPServerBootstrapPlugin(BootstrapPlugin):
    plugin_id: str = "mcp_server_bootstrap_v1"
    description: str = (
        "Bootstrap plugin that starts an MCP server exposing Genie's "
        "registered tools to external MCP clients (Claude Desktop, IDEs, "
        "other agents)."
    )

    async def bootstrap(self, genie: "Genie") -> None:
        if not MCP_AVAILABLE:
            logger.warning(
                f"{self.plugin_id}: `mcp` SDK not installed; cannot start "
                "MCP server. Install the 'mcp' extra to enable."
            )
            return
        cfg = (genie._config.extension_configurations.get("mcp_server", {}) if hasattr(genie, "_config") else {}) or {}
        if not cfg.get("enabled", False):
            logger.debug(
                f"{self.plugin_id}: bootstrap disabled (set extension_configurations.mcp_server.enabled=True to enable)."
            )
            return
        server_name = cfg.get("server_name", "genie-tooling")
        transport = cfg.get("transport", "stdio")
        if transport != "stdio":
            logger.error(
                f"{self.plugin_id}: only the 'stdio' transport is currently supported; "
                f"got {transport!r}."
            )
            return

        server = Server(server_name)

        @server.list_tools()  # type: ignore[misc]
        async def _list_tools():
            """MCP -> Genie: surface every registered Genie tool."""
            out = []
            tools = await genie.tools.list(enabled_only=True)
            for tool in tools:
                try:
                    meta = await tool.get_metadata()
                except Exception:
                    continue
                out.append(
                    mcp_types.Tool(
                        name=meta.get("identifier", tool.identifier),
                        description=meta.get("description_llm")
                        or meta.get("description_human", "")
                        or "",
                        inputSchema=meta.get(
                            "input_schema", {"type": "object", "properties": {}}
                        ),
                    )
                )
            return out

        @server.call_tool()  # type: ignore[misc]
        async def _call_tool(name: str, arguments: Dict[str, Any]):
            """MCP -> Genie: route the tool call through genie.execute_tool
            with provenance metadata so audit can see the MCP origin."""
            try:
                result = await genie.execute_tool(
                    name,
                    context={"caller_chain": ["mcp_server_bootstrap_v1"]},
                    **(arguments or {}),
                )
                text = result if isinstance(result, str) else _safe_json(result)
                return [mcp_types.TextContent(type="text", text=text)]
            except Exception as e:
                logger.error(
                    f"{self.plugin_id}: error invoking tool {name!r}: {e}",
                    exc_info=True,
                )
                return [
                    mcp_types.TextContent(
                        type="text", text=f"Error: {type(e).__name__}: {e}"
                    )
                ]

        # Run the server as a background task. The stdio transport blocks
        # on stdin/stdout, so it's only useful when Genie is itself being
        # invoked as a subprocess (e.g. by Claude Desktop). When the host
        # process owns stdin/stdout for normal use, this transport is
        # incorrect — operators should use it deliberately.
        async def _run_server() -> None:
            async with stdio_server() as (read, write):
                await server.run(read, write, server.create_initialization_options())

        task = asyncio.create_task(_run_server())
        # Stash the task so the host process / tests can cancel it.
        genie._mcp_server_task = task
        logger.info(
            f"{self.plugin_id}: MCP server {server_name!r} started over stdio."
        )


def _safe_json(value: Any) -> str:
    import json
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)
