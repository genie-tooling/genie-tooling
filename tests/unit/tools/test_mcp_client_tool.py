"""Unit tests for the MCP client tool (M2).

Real MCP connectivity requires spawning a subprocess; here we test the
Genie-side translation layer: schema conversion, MCPRemoteTool metadata
shape, error handling on the call path.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from genie_tooling.tools.impl.mcp_client_tool import (
    MCP_AVAILABLE,
    MCPClientToolPlugin,
    MCPRemoteTool,
)

pytestmark = pytest.mark.skipif(not MCP_AVAILABLE, reason="mcp SDK not installed")


def _make_remote_tool(session_mock):
    return MCPRemoteTool(
        identifier="mcp_test_calculator",
        name="calculator",
        description="Does math.",
        input_schema={"type": "object", "properties": {"x": {"type": "number"}}},
        session=session_mock,
        remote_tool_name="calculator",
    )


@pytest.mark.asyncio
async def test_mcp_remote_tool_metadata_shape():
    tool = _make_remote_tool(MagicMock())
    meta = await tool.get_metadata()
    assert meta["identifier"] == "mcp_test_calculator"
    assert meta["name"] == "calculator"
    assert meta["description_llm"] == "Does math."
    assert meta["input_schema"] == {
        "type": "object",
        "properties": {"x": {"type": "number"}},
    }
    assert "mcp" in meta["tags"]
    assert "remote" in meta["tags"]
    assert meta["source"] == "mcp"


@pytest.mark.asyncio
async def test_mcp_remote_tool_execute_translates_call_result():
    """Test that the tool unwraps an MCP CallToolResult into a Genie-friendly
    dict shape with is_error + content blocks."""
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "the answer is 42"
    mock_result = MagicMock()
    mock_result.content = [text_block]
    mock_result.isError = False

    session = MagicMock()
    session.call_tool = AsyncMock(return_value=mock_result)

    tool = _make_remote_tool(session)
    result = await tool.execute(
        params={"x": 5}, key_provider=MagicMock(), context=None
    )
    assert result == {
        "is_error": False,
        "content": [{"type": "text", "text": "the answer is 42"}],
    }
    session.call_tool.assert_awaited_once_with(name="calculator", arguments={"x": 5})


@pytest.mark.asyncio
async def test_mcp_remote_tool_execute_returns_error_dict_on_exception():
    session = MagicMock()
    session.call_tool = AsyncMock(side_effect=RuntimeError("connection lost"))

    tool = _make_remote_tool(session)
    result = await tool.execute(
        params={}, key_provider=MagicMock(), context=None
    )
    assert result["is_error"] is True
    assert "connection lost" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_mcp_client_setup_without_command_fails_gracefully(caplog):
    import logging
    caplog.set_level(logging.ERROR)
    plugin = MCPClientToolPlugin()
    await plugin.setup(config={"name": "test"})
    assert plugin.tools == []
    assert any("command" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_mcp_client_setup_with_unreachable_server_returns_empty_tools():
    """If the MCP server subprocess fails to start, the plugin should log
    the error and surface zero tools rather than raising."""
    plugin = MCPClientToolPlugin()
    await plugin.setup(
        config={
            "name": "test",
            "command": "/does/not/exist",
            "args": [],
        }
    )
    assert plugin.tools == []
