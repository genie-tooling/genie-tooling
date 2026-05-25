"""Phase 6B.3.1 — Native Slack tool tests."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from genie_tooling.tools.impl.slack import (
    SlackAddReactionTool,
    SlackGetChannelHistoryTool,
    SlackGetUserProfileTool,
    SlackListChannelsTool,
    SlackPostMessageTool,
)


def _mock_response(json_body):
    r = MagicMock(spec=httpx.Response)
    r.content = b"ok"
    r.json = MagicMock(return_value=json_body)
    return r


def _kp(token: str = "xoxb-test"):
    kp = MagicMock()
    kp.get_key = AsyncMock(return_value=token)
    return kp


@pytest.mark.asyncio()
async def test_post_message_declares_write_side_effects():
    tool = SlackPostMessageTool()
    await tool.setup()
    md = await tool.get_metadata()
    assert md["side_effects"] == "write"
    assert md["requires_approval"] is None  # defer to policy
    assert "slack" in md["tags"]
    tool._client = None
    await tool.teardown()


@pytest.mark.asyncio()
async def test_post_message_audit_artifact_captures_request_body():
    tool = SlackPostMessageTool()
    await tool.setup()
    tool._client = MagicMock()
    tool._client.post = AsyncMock(return_value=_mock_response({"ok": True, "ts": "1.2345"}))

    result = await tool.execute(
        {"channel": "#engineering", "text": "hello"},
        key_provider=_kp(),
    )
    assert result.get("ok") is True
    artifact = result["audit_artifact"]
    assert artifact["endpoint"] == "chat.postMessage"
    assert artifact["request_body"] == {"channel": "#engineering", "text": "hello"}
    tool._client = None
    await tool.teardown()


@pytest.mark.asyncio()
async def test_post_message_with_thread_ts():
    tool = SlackPostMessageTool()
    await tool.setup()
    tool._client = MagicMock()
    tool._client.post = AsyncMock(return_value=_mock_response({"ok": True}))

    await tool.execute(
        {"channel": "C001", "text": "reply", "thread_ts": "1.234"},
        key_provider=_kp(),
    )
    body_arg = tool._client.post.await_args.kwargs["json"]
    assert body_arg["thread_ts"] == "1.234"
    tool._client = None
    await tool.teardown()


@pytest.mark.asyncio()
async def test_post_message_missing_token_returns_error():
    tool = SlackPostMessageTool()
    await tool.setup()
    tool._client = MagicMock()
    tool._client.post = AsyncMock()

    result = await tool.execute(
        {"channel": "C001", "text": "x"},
        key_provider=_kp(token=None),
    )
    assert "error" in result
    assert "not configured" in result["error"]
    tool._client.post.assert_not_called()
    tool._client = None
    await tool.teardown()


@pytest.mark.asyncio()
async def test_get_user_profile_is_read_idempotent_cacheable():
    tool = SlackGetUserProfileTool()
    await tool.setup()
    md = await tool.get_metadata()
    assert md["side_effects"] == "read"
    assert md["idempotent"] is True
    assert md["cacheable"] is True
    assert md["requires_approval"] is False
    tool._client = None
    await tool.teardown()


@pytest.mark.asyncio()
async def test_get_channel_history_calls_correct_endpoint():
    tool = SlackGetChannelHistoryTool()
    await tool.setup()
    tool._client = MagicMock()
    tool._client.get = AsyncMock(return_value=_mock_response({"ok": True, "messages": []}))

    await tool.execute(
        {"channel": "C001", "limit": 25, "oldest": "1700000000"},
        key_provider=_kp(),
    )
    url_arg = tool._client.get.await_args.args[0]
    assert url_arg.endswith("/conversations.history")
    params = tool._client.get.await_args.kwargs["params"]
    assert params == {"channel": "C001", "limit": 25, "oldest": "1700000000"}
    tool._client = None
    await tool.teardown()


@pytest.mark.asyncio()
async def test_list_channels_pagination_with_cursor():
    tool = SlackListChannelsTool()
    await tool.setup()
    tool._client = MagicMock()
    tool._client.get = AsyncMock(return_value=_mock_response({"ok": True, "channels": []}))

    await tool.execute({"limit": 100, "cursor": "abc"}, key_provider=_kp())
    params = tool._client.get.await_args.kwargs["params"]
    assert params["cursor"] == "abc"
    tool._client = None
    await tool.teardown()


@pytest.mark.asyncio()
async def test_slack_error_response_surfaces_in_result():
    tool = SlackPostMessageTool()
    await tool.setup()
    tool._client = MagicMock()
    tool._client.post = AsyncMock(return_value=_mock_response({"ok": False, "error": "channel_not_found"}))

    result = await tool.execute({"channel": "BAD", "text": "x"}, key_provider=_kp())
    assert result.get("error") == "channel_not_found"
    tool._client = None
    await tool.teardown()


@pytest.mark.asyncio()
async def test_add_reaction_idempotent_and_write():
    tool = SlackAddReactionTool()
    await tool.setup()
    md = await tool.get_metadata()
    assert md["side_effects"] == "write"
    assert md["idempotent"] is True
    tool._client = None
    await tool.teardown()
