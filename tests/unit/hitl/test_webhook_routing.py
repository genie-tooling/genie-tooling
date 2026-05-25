"""Phase 6A.5 — WebhookApprovalPlugin per-request routing."""
from __future__ import annotations

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from genie_tooling.hitl.impl.webhook_approval import WebhookApprovalPlugin


def _make_response(json_body: Dict[str, Any]):
    r = MagicMock(spec=httpx.Response)
    r.status_code = 200
    r.raise_for_status = MagicMock()
    r.json = MagicMock(return_value=json_body)
    return r


@pytest.mark.asyncio()
async def test_routes_by_side_effects_destructive_to_pagerduty():
    plugin = WebhookApprovalPlugin()
    await plugin.setup(
        {
            "routes": [
                {
                    "match": {"side_effects_in": ["destructive"]},
                    "url": "https://pagerduty.example/approve",
                },
                {
                    "match": {"tool_id_in": ["github_*"]},
                    "url": "https://github.example/approve",
                },
            ],
            "default_url": "https://general.example/approve",
        }
    )
    plugin._client = MagicMock()
    plugin._client.post = AsyncMock(return_value=_make_response({"status": "approved", "approver_id": "human"}))

    await plugin.request_approval(
        {
            "request_id": "r1",
            "prompt": "?",
            "data_to_approve": {
                "tool_id": "kubectl_delete",
                "tool_metadata": {"side_effects": "destructive"},
                "params": {},
            },
            "context": {},
            "timeout_seconds": 60,
        }
    )
    call = plugin._client.post.await_args
    assert call.args[0] == "https://pagerduty.example/approve"


@pytest.mark.asyncio()
async def test_routes_by_tool_id_glob_to_code_reviewers():
    plugin = WebhookApprovalPlugin()
    await plugin.setup(
        {
            "routes": [
                {
                    "match": {"side_effects_in": ["destructive"]},
                    "url": "https://pagerduty.example/approve",
                },
                {
                    "match": {"tool_id_in": ["github_*", "gitlab_*"]},
                    "url": "https://reviewers.example/approve",
                },
            ],
            "default_url": "https://general.example/approve",
        }
    )
    plugin._client = MagicMock()
    plugin._client.post = AsyncMock(return_value=_make_response({"status": "approved", "approver_id": "human"}))

    await plugin.request_approval(
        {
            "request_id": "r1",
            "prompt": "?",
            "data_to_approve": {
                "tool_id": "github_create_pr",
                "tool_metadata": {"side_effects": "write"},
                "params": {"repo": "team/foo"},
            },
            "context": {},
            "timeout_seconds": 60,
        }
    )
    call = plugin._client.post.await_args
    assert call.args[0] == "https://reviewers.example/approve"


@pytest.mark.asyncio()
async def test_falls_through_to_default_url_on_no_route_match():
    plugin = WebhookApprovalPlugin()
    await plugin.setup(
        {
            "routes": [{"match": {"tool_id": "very_specific_tool"}, "url": "https://specific.example"}],
            "default_url": "https://general.example/approve",
        }
    )
    plugin._client = MagicMock()
    plugin._client.post = AsyncMock(return_value=_make_response({"status": "approved", "approver_id": "human"}))

    await plugin.request_approval(
        {
            "request_id": "r1",
            "prompt": "?",
            "data_to_approve": {"tool_id": "something_else", "params": {}},
            "context": {},
            "timeout_seconds": 60,
        }
    )
    call = plugin._client.post.await_args
    assert call.args[0] == "https://general.example/approve"


@pytest.mark.asyncio()
async def test_routes_by_params_match_glob():
    """params_match supports glob for string values."""
    plugin = WebhookApprovalPlugin()
    await plugin.setup(
        {
            "routes": [
                {
                    "match": {"tool_id": "kubectl_apply", "params_match": {"namespace": "prod*"}},
                    "url": "https://prod-approvers.example/approve",
                },
            ],
            "default_url": "https://default.example/approve",
        }
    )
    plugin._client = MagicMock()
    plugin._client.post = AsyncMock(return_value=_make_response({"status": "approved", "approver_id": "human"}))

    await plugin.request_approval(
        {
            "request_id": "r1",
            "prompt": "?",
            "data_to_approve": {"tool_id": "kubectl_apply", "params": {"namespace": "prod-us-east-1"}},
            "context": {},
            "timeout_seconds": 60,
        }
    )
    assert plugin._client.post.await_args.args[0] == "https://prod-approvers.example/approve"


@pytest.mark.asyncio()
async def test_legacy_single_url_still_works():
    """Old config shape (url=...) continues to function as the default."""
    plugin = WebhookApprovalPlugin()
    await plugin.setup({"url": "https://legacy.example/approve"})
    plugin._client = MagicMock()
    plugin._client.post = AsyncMock(return_value=_make_response({"status": "denied", "approver_id": "x", "reason": "no"}))

    resp = await plugin.request_approval(
        {
            "request_id": "r1",
            "prompt": "?",
            "data_to_approve": {"tool_id": "anything", "params": {}},
            "context": {},
            "timeout_seconds": 60,
        }
    )
    assert plugin._client.post.await_args.args[0] == "https://legacy.example/approve"
    assert resp["status"] == "denied"


@pytest.mark.asyncio()
async def test_route_specific_headers_override_default():
    plugin = WebhookApprovalPlugin()
    await plugin.setup(
        {
            "routes": [
                {
                    "match": {"side_effects_in": ["destructive"]},
                    "url": "https://pd.example",
                    "headers": {"X-PD-Token": "secret"},
                }
            ],
            "default_url": "https://default.example",
            "default_headers": {"X-General-Token": "general"},
        }
    )
    plugin._client = MagicMock()
    plugin._client.post = AsyncMock(return_value=_make_response({"status": "approved", "approver_id": "x"}))

    await plugin.request_approval(
        {
            "request_id": "r1",
            "prompt": "?",
            "data_to_approve": {
                "tool_id": "x",
                "tool_metadata": {"side_effects": "destructive"},
                "params": {},
            },
            "context": {},
            "timeout_seconds": 60,
        }
    )
    headers_used = plugin._client.post.await_args.kwargs["headers"]
    assert headers_used == {"X-PD-Token": "secret"}
