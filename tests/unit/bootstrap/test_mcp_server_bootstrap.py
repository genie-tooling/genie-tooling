"""Unit tests for MCPServerBootstrapPlugin (M3).

The MCP stdio server itself can't be unit-tested without subprocessing —
it owns stdin/stdout. These tests verify the bootstrap's configuration
gating: disabled-by-default, refuses unknown transports, no-op when the
SDK isn't installed.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from genie_tooling.bootstrap.impl.mcp_server import (
    MCP_AVAILABLE,
    MCPServerBootstrapPlugin,
)
from genie_tooling.config.models import MiddlewareConfig


class _GenieStub:
    """A minimal stub that doesn't auto-create attributes (unlike MagicMock).
    Lets us assert ``not hasattr(stub, 'x')`` reliably."""

    def __init__(self, extension_configurations=None):
        self._config = MiddlewareConfig(
            extension_configurations=extension_configurations or {}
        )


def _mock_genie(extension_configurations=None):
    return _GenieStub(extension_configurations)


@pytest.mark.asyncio
async def test_bootstrap_is_noop_when_disabled():
    """Default config (enabled=False) → bootstrap does nothing."""
    plugin = MCPServerBootstrapPlugin()
    genie = _mock_genie({"mcp_server": {"enabled": False}})
    await plugin.bootstrap(genie)
    assert not hasattr(genie, "_mcp_server_task")


@pytest.mark.asyncio
async def test_bootstrap_is_noop_when_no_config():
    """No extension_configurations entry at all → bootstrap does nothing."""
    plugin = MCPServerBootstrapPlugin()
    genie = _mock_genie({})
    await plugin.bootstrap(genie)
    assert not hasattr(genie, "_mcp_server_task")


@pytest.mark.asyncio
async def test_bootstrap_rejects_unknown_transport(caplog):
    """SSE / HTTP transports not yet implemented — bootstrap refuses
    rather than silently failing."""
    import logging
    caplog.set_level(logging.ERROR)
    plugin = MCPServerBootstrapPlugin()
    genie = _mock_genie({"mcp_server": {"enabled": True, "transport": "sse"}})
    await plugin.bootstrap(genie)
    assert any("only the 'stdio' transport" in r.message for r in caplog.records)


@pytest.mark.skipif(MCP_AVAILABLE, reason="needs absent MCP SDK")
@pytest.mark.asyncio
async def test_bootstrap_warns_when_mcp_sdk_missing(caplog):
    """Without the mcp extra, bootstrap should warn but not raise."""
    import logging
    caplog.set_level(logging.WARNING)
    plugin = MCPServerBootstrapPlugin()
    genie = _mock_genie({"mcp_server": {"enabled": True}})
    await plugin.bootstrap(genie)
    assert any("mcp` SDK not installed" in r.message for r in caplog.records)
