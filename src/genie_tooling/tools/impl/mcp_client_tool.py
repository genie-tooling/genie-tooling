"""MCPClientToolPlugin: dynamic tool plugin that proxies MCP server tools
into the Genie tool registry (M2).

Connects to a single MCP server, lists its tools, and exposes each as a
distinct Genie ``Tool`` instance. Existing agents (ReAct, ReWOO, etc.)
then see those tools alongside any locally-registered tools — no
per-tool wrappers needed.

The plugin is a **bootstrap-style** plugin in design: instead of a single
Tool plugin (which doesn't fit the "one MCP server = N tools" shape),
this is a ``Plugin`` that handles its own registration into a provided
``ToolManager`` at setup time. Typical use:

    config = MiddlewareConfig(
        extension_configurations={
            "mcp_client": {
                "servers": [
                    {
                        "name": "filesystem",
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                    },
                ],
            }
        }
    )

A companion bootstrap plugin (``MCPClientBootstrapPlugin``) reads
``extension_configurations["mcp_client"]`` and wires the connections.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from genie_tooling.core.types import Plugin
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tools.abc import Tool

logger = logging.getLogger(__name__)

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:  # pragma: no cover
    ClientSession = None  # type: ignore
    StdioServerParameters = None  # type: ignore
    stdio_client = None  # type: ignore
    MCP_AVAILABLE = False


class MCPRemoteTool(Tool):
    """A Genie Tool that proxies to a remote MCP server tool.

    Each instance corresponds to one tool exposed by an MCP server. When
    Genie calls ``execute()``, the tool forwards the call over the
    persistent MCP session and returns the result.

    Phase 6B.1: supports an optional ``overlay`` dict that injects
    side-effects classification, requires_approval, response redaction,
    and other framework-level metadata that the upstream MCP server
    doesn't know about.
    """

    def __init__(
        self,
        identifier: str,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        session: Any,
        remote_tool_name: str,
        overlay: Optional[Dict[str, Any]] = None,
    ):
        self._identifier = identifier
        self._name = name
        self._description = description
        self._input_schema = input_schema
        self._session = session
        self._remote_tool_name = remote_tool_name
        self._overlay = overlay or {}

    @property
    def identifier(self) -> str:  # type: ignore[override]
        return self._identifier

    @property
    def plugin_id(self) -> str:  # type: ignore[override]
        return self._identifier

    async def get_metadata(self) -> Dict[str, Any]:
        # Base metadata from MCP discovery
        md: Dict[str, Any] = {
            "identifier": self._identifier,
            "name": self._name,
            "description_human": self._description,
            "description_llm": self._description,
            "input_schema": self._input_schema,
            "output_schema": {"type": "object"},
            "key_requirements": [],
            "tags": list(self._overlay.get("tags") or ["mcp", "remote"]),
            "version": str(self._overlay.get("version") or "1.0"),
            "cacheable": bool(self._overlay.get("cacheable", False)),
            "cache_ttl_seconds": self._overlay.get("cache_ttl_seconds"),
            "side_effects": str(self._overlay.get("side_effects", "unknown")),
            "requires_approval": self._overlay.get("requires_approval"),
            "idempotent": bool(self._overlay.get("idempotent", False)),
            "source": "mcp",
        }
        return md

    async def execute(
        self,
        params: Dict[str, Any],
        key_provider: KeyProvider,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        try:
            result = await self._session.call_tool(
                name=self._remote_tool_name, arguments=params
            )
            content_blocks = []
            for block in getattr(result, "content", []) or []:
                btype = getattr(block, "type", None)
                if btype == "text":
                    content_blocks.append({"type": "text", "text": getattr(block, "text", "")})
                else:
                    content_blocks.append({"type": btype or "unknown", "data": str(block)})
            payload = {
                "is_error": bool(getattr(result, "isError", False)),
                "content": content_blocks,
            }
            # Phase 6B.1.3: response redaction by simple key-path. Each key
            # in `redact_response_fields` is a dotted path; the matching
            # value is replaced with `"[REDACTED]"` before the LLM sees it.
            redact = self._overlay.get("redact_response_fields") or []
            if redact:
                for keypath in redact:
                    _redact_keypath(payload, str(keypath))
            return payload
        except Exception as e:
            logger.error(
                f"MCP tool {self._identifier!r} failed: {e}", exc_info=True
            )
            return {"is_error": True, "content": [{"type": "text", "text": str(e)}]}

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        pass

    async def teardown(self) -> None:
        pass


def _redact_keypath(obj: Any, keypath: str) -> None:
    """In-place redaction of a dotted key path. Supports literal ``*`` to mean
    "redact every key at this level" but otherwise keeps it simple.

    Example: ``content.text`` redacts ``obj["content"]["text"]``.
    """
    parts = keypath.split(".")
    cur: Any = obj
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        if isinstance(cur, list):
            for item in cur:
                _redact_keypath(item, ".".join(parts[i:]))
            return
        if not isinstance(cur, dict):
            return
        if part == "*":
            if last:
                for k in list(cur.keys()):
                    cur[k] = "[REDACTED]"
                return
            for k in list(cur.keys()):
                _redact_keypath(cur[k], ".".join(parts[i + 1 :]))
            return
        if part not in cur:
            return
        if last:
            cur[part] = "[REDACTED]"
            return
        cur = cur[part]


class MCPClientToolPlugin(Plugin):
    """Manager-style plugin: connects to one MCP server and produces N
    ``MCPRemoteTool`` instances, one per discovered remote tool.

    Configuration::

        name: str (default "default")
            Used as a prefix for generated tool identifiers, so tools from
            different MCP servers don't collide. Resulting tool ids:
            ``mcp_<name>_<remote_tool_name>``.
        command: str (required for stdio transport)
            Subprocess command that runs the MCP server.
        args: list[str]
            Args to pass to the subprocess.
        env: dict[str, str] (optional)
            Environment variables for the subprocess.
    """

    plugin_id: str = "mcp_client_tool_v1"
    description: str = (
        "Connects to an MCP server via stdio and exposes its tools as Genie "
        "tools. Spawns one MCPRemoteTool per discovered server tool."
    )

    _exit_stack: Optional[AsyncExitStack] = None
    _session: Any = None
    _tools: List[MCPRemoteTool]
    _server_name: str = "default"

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        if not MCP_AVAILABLE:
            logger.error(
                f"{self.plugin_id}: `mcp` SDK not installed. Install with the "
                f"'mcp' extra: `pip install genie-tooling[mcp]`."
            )
            self._tools = []
            return
        cfg = config or {}
        self._server_name = cfg.get("name", "default")
        command = cfg.get("command")
        args = cfg.get("args", [])
        env = cfg.get("env")
        # Phase 6B.1.2: overlays map remote tool names → metadata patches
        # (side_effects, requires_approval, redact_response_fields, etc.)
        overlays: Dict[str, Dict[str, Any]] = dict(cfg.get("overlays") or {})
        if not command:
            logger.error(
                f"{self.plugin_id}: 'command' is required to launch the MCP server."
            )
            self._tools = []
            return

        params = StdioServerParameters(command=command, args=args, env=env)
        self._exit_stack = AsyncExitStack()
        try:
            stdio_transport = await self._exit_stack.enter_async_context(
                stdio_client(params)
            )
            read_stream, write_stream = stdio_transport
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self._session.initialize()
            list_result = await self._session.list_tools()
            self._tools = [
                MCPRemoteTool(
                    identifier=f"mcp_{self._server_name}_{t.name}",
                    name=t.name,
                    description=t.description or f"MCP tool {t.name}",
                    input_schema=t.inputSchema or {"type": "object", "properties": {}},
                    session=self._session,
                    remote_tool_name=t.name,
                    overlay=overlays.get(t.name) or overlays.get("*"),
                )
                for t in list_result.tools
            ]
            logger.info(
                f"{self.plugin_id}: connected to MCP server {self._server_name!r}, "
                f"discovered {len(self._tools)} tool(s); "
                f"{sum(1 for t in self._tools if t._overlay)} overlay match(es)."
            )
        except Exception as e:
            logger.error(
                f"{self.plugin_id}: failed to connect to MCP server "
                f"{self._server_name!r}: {e}",
                exc_info=True,
            )
            await self._safe_teardown()
            self._tools = []

    @property
    def tools(self) -> List[MCPRemoteTool]:
        """The list of remote tools discovered from the MCP server."""
        return self._tools

    async def _safe_teardown(self) -> None:
        if self._exit_stack is not None:
            try:
                await self._exit_stack.aclose()
            except Exception:
                logger.warning(
                    f"{self.plugin_id}: error closing MCP transport", exc_info=True
                )
            self._exit_stack = None
            self._session = None

    async def teardown(self) -> None:
        await self._safe_teardown()
