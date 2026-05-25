"""Phase 6B.1 — MCP composition layer tests.

Tests overlay application + response redaction without needing the real
MCP SDK (we mock the session/client).
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from genie_tooling.tools.impl.mcp_client_tool import MCPRemoteTool, _redact_keypath


# ---- response redaction (pure) ----


def test_redact_keypath_simple_field():
    payload = {"content": {"text": "secret"}, "ok": True}
    _redact_keypath(payload, "content.text")
    assert payload == {"content": {"text": "[REDACTED]"}, "ok": True}


def test_redact_keypath_missing_path_is_noop():
    payload = {"content": {"text": "ok"}}
    _redact_keypath(payload, "nowhere.here")
    assert payload == {"content": {"text": "ok"}}


def test_redact_keypath_wildcard_at_terminal_level():
    payload = {"secrets": {"a": "x", "b": "y"}}
    _redact_keypath(payload, "secrets.*")
    assert payload == {"secrets": {"a": "[REDACTED]", "b": "[REDACTED]"}}


def test_redact_keypath_in_list_of_blocks():
    payload = {"content": [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]}
    _redact_keypath(payload, "content.text")
    assert payload["content"][0]["text"] == "[REDACTED]"
    assert payload["content"][1]["text"] == "[REDACTED]"


# ---- MCPRemoteTool overlay application ----


@pytest.mark.asyncio
async def test_overlay_sets_side_effects_and_requires_approval():
    """Overlay metadata flows through get_metadata so the policy plugin can read it."""
    session = MagicMock()
    tool = MCPRemoteTool(
        identifier="mcp_slack_postMessage",
        name="postMessage",
        description="post a message",
        input_schema={"type": "object"},
        session=session,
        remote_tool_name="postMessage",
        overlay={
            "side_effects": "write",
            "requires_approval": True,
            "idempotent": False,
            "tags": ["slack", "chat"],
        },
    )
    md = await tool.get_metadata()
    assert md["side_effects"] == "write"
    assert md["requires_approval"] is True
    assert md["idempotent"] is False
    assert "slack" in md["tags"]


@pytest.mark.asyncio
async def test_overlay_default_metadata_when_no_overlay():
    """Tool without overlay reports side_effects='unknown'."""
    session = MagicMock()
    tool = MCPRemoteTool(
        identifier="mcp_x_y",
        name="y",
        description="",
        input_schema={},
        session=session,
        remote_tool_name="y",
    )
    md = await tool.get_metadata()
    assert md["side_effects"] == "unknown"
    assert md["requires_approval"] is None


@pytest.mark.asyncio
async def test_execute_applies_redact_response_fields():
    """When overlay declares redact_response_fields, those keys are scrubbed."""
    # Mock MCP session call_tool returns a content block with a secret
    mock_block = SimpleNamespace(type="text", text="my secret 12345")
    mock_result = SimpleNamespace(content=[mock_block], isError=False)
    session = MagicMock()
    session.call_tool = AsyncMock(return_value=mock_result)

    tool = MCPRemoteTool(
        identifier="mcp_slack_postMessage",
        name="postMessage",
        description="",
        input_schema={},
        session=session,
        remote_tool_name="postMessage",
        overlay={"redact_response_fields": ["content.text"]},
    )
    result = await tool.execute({}, key_provider=MagicMock(), context={})
    assert result["content"][0]["text"] == "[REDACTED]"


@pytest.mark.asyncio
async def test_execute_no_overlay_no_redaction():
    """No overlay → response passes through untouched."""
    mock_block = SimpleNamespace(type="text", text="plain text")
    mock_result = SimpleNamespace(content=[mock_block], isError=False)
    session = MagicMock()
    session.call_tool = AsyncMock(return_value=mock_result)

    tool = MCPRemoteTool(
        identifier="mcp_x_y",
        name="y",
        description="",
        input_schema={},
        session=session,
        remote_tool_name="y",
    )
    result = await tool.execute({}, key_provider=MagicMock(), context={})
    assert result["content"][0]["text"] == "plain text"


@pytest.mark.asyncio
async def test_overlay_redact_multiple_paths():
    mock_block = SimpleNamespace(type="text", text="secret here")
    mock_result = SimpleNamespace(content=[mock_block], isError=False)
    session = MagicMock()
    session.call_tool = AsyncMock(return_value=mock_result)

    tool = MCPRemoteTool(
        identifier="mcp_x_y",
        name="y",
        description="",
        input_schema={},
        session=session,
        remote_tool_name="y",
        overlay={"redact_response_fields": ["content.text", "is_error"]},
    )
    result = await tool.execute({}, key_provider=MagicMock(), context={})
    assert result["content"][0]["text"] == "[REDACTED]"
    assert result["is_error"] == "[REDACTED]"


# ---- Overlay catalog loading ----


def test_load_overlay_file_returns_overlays_dict(tmp_path):
    from genie_tooling.tools.impl.mcp_composition import _load_overlay_file
    p = tmp_path / "slack.yml"
    p.write_text(
        """
overlays:
  postMessage:
    side_effects: write
    requires_approval: true
  listChannels:
    side_effects: read
    idempotent: true
"""
    )
    overlays = _load_overlay_file(p)
    assert overlays["postMessage"]["side_effects"] == "write"
    assert overlays["postMessage"]["requires_approval"] is True
    assert overlays["listChannels"]["idempotent"] is True


def test_load_overlay_file_returns_none_for_missing(tmp_path):
    from genie_tooling.tools.impl.mcp_composition import _load_overlay_file
    overlays = _load_overlay_file(tmp_path / "nope.yml")
    assert overlays is None


def test_load_overlay_file_accepts_flat_shape(tmp_path):
    """File can be a flat overlay map without the `overlays:` wrapper."""
    from genie_tooling.tools.impl.mcp_composition import _load_overlay_file
    p = tmp_path / "github.yml"
    p.write_text(
        """
createIssue:
  side_effects: write
  requires_approval: true
listIssues:
  side_effects: read
"""
    )
    overlays = _load_overlay_file(p)
    assert overlays["createIssue"]["side_effects"] == "write"
