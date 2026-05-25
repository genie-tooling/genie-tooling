"""Phase 6A.5b — ClaudeCodePermissionsPlugin tests.

Three-tier allow/ask/deny with glob match on tool_id AND params, side-effects
defaults, session-level always-allow.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from genie_tooling.hitl.impl.claude_code_permissions import (
    ClaudeCodePermissionsPlugin,
)
from genie_tooling.hitl.manager import HITLManager


# ---------------------------------------------------------------------------
# Direct plugin behaviour
# ---------------------------------------------------------------------------


async def _build_plugin(policy_inline=None, **extra):
    p = ClaudeCodePermissionsPlugin()
    await p.setup({"policy_inline": policy_inline, **extra})
    return p


async def _req(p, tool_id, params=None, side_effects=None, requires_approval=None, **ctx):
    return await p.request_approval(
        {
            "request_id": "r1",
            "prompt": "approve?",
            "data_to_approve": {
                "tool_id": tool_id,
                "params": params or {},
                "tool_metadata": {
                    "side_effects": side_effects,
                    "requires_approval": requires_approval,
                },
            },
            "context": ctx or {},
            "timeout_seconds": 60,
        }
    )


@pytest.mark.asyncio
async def test_explicit_allow_rule_by_tool_id_and_params():
    """kubectl get pods → allowed by an exact rule."""
    policy = {
        "rules": [
            {
                "id": "ALLOW_KUBECTL_READS",
                "match": {"tool_id": "kubectl_tool_v1", "params_match": {"operation": "get"}},
                "decision": "allow",
                "reason": "reads are safe",
            }
        ]
    }
    p = await _build_plugin(policy_inline=policy)
    resp = await _req(p, "kubectl_tool_v1", params={"operation": "get", "resource": "pods"})
    assert resp["status"] == "approved"
    assert "ALLOW_KUBECTL_READS" in resp["approver_id"]


@pytest.mark.asyncio
async def test_explicit_deny_rule_for_destructive_params():
    """kubectl delete namespace * → denied."""
    policy = {
        "rules": [
            {
                "id": "DENY_NS_DELETE",
                "match": {
                    "tool_id": "kubectl_tool_v1",
                    "params_match": {"operation": "delete", "resource": "namespace"},
                },
                "decision": "deny",
                "reason": "namespace deletion is forbidden",
            }
        ]
    }
    p = await _build_plugin(policy_inline=policy)
    resp = await _req(p, "kubectl_tool_v1", params={"operation": "delete", "resource": "namespace", "name": "prod"})
    assert resp["status"] == "denied"
    assert "namespace deletion" in resp["reason"]


@pytest.mark.asyncio
async def test_glob_match_on_tool_id_in():
    """`slack:*` matches any tool starting with slack:."""
    policy = {
        "rules": [
            {
                "id": "ALLOW_SLACK_LIST_ONLY",
                "match": {"tool_id_in": ["slack:list*"]},
                "decision": "allow",
            }
        ]
    }
    p = await _build_plugin(policy_inline=policy)
    assert (await _req(p, "slack:listChannels"))["status"] == "approved"
    # slack:postMessage doesn't match the glob → no rule → defaults to ask
    assert (await _req(p, "slack:postMessage", side_effects="write"))["status"] == "ask_human"


@pytest.mark.asyncio
async def test_glob_match_on_param_values():
    """Channel value '#engineering*' glob matches '#engineering-platform'."""
    policy = {
        "rules": [
            {
                "id": "SLACK_POST_TO_ENG",
                "match": {
                    "tool_id": "slack_tool_v1",
                    "params_match": {"channel": "#engineering*"},
                },
                "decision": "allow",
            }
        ]
    }
    p = await _build_plugin(policy_inline=policy)
    r_eng = await _req(p, "slack_tool_v1", params={"channel": "#engineering-platform"})
    assert r_eng["status"] == "approved"
    r_other = await _req(p, "slack_tool_v1", params={"channel": "#general"}, side_effects="write")
    assert r_other["status"] == "ask_human"


@pytest.mark.asyncio
async def test_side_effects_in_match():
    """Match by side_effects_in list."""
    policy = {
        "rules": [
            {
                "id": "ASK_ALL_WRITES",
                "match": {"side_effects_in": ["write", "destructive"]},
                "decision": "ask",
            }
        ]
    }
    p = await _build_plugin(policy_inline=policy)
    assert (await _req(p, "any_tool", side_effects="write"))["status"] == "ask_human"
    assert (await _req(p, "any_tool", side_effects="destructive"))["status"] == "ask_human"
    # Read tool with no matching rule → side_effects_default → allow
    assert (await _req(p, "any_tool", side_effects="read"))["status"] == "approved"


@pytest.mark.asyncio
async def test_side_effects_defaults_apply_when_no_rule_matches():
    """No rule matches; falls through to side_effects_defaults (read→allow, write→ask, destructive→ask, unknown→ask)."""
    p = await _build_plugin(policy_inline={"rules": []})
    assert (await _req(p, "x", side_effects="none"))["status"] == "approved"
    assert (await _req(p, "x", side_effects="read"))["status"] == "approved"
    assert (await _req(p, "x", side_effects="write"))["status"] == "ask_human"
    assert (await _req(p, "x", side_effects="destructive"))["status"] == "ask_human"
    assert (await _req(p, "x", side_effects="unknown"))["status"] == "ask_human"


@pytest.mark.asyncio
async def test_tool_metadata_requires_approval_true_forces_ask():
    """A tool declaring requires_approval=True is asked even with read side_effects."""
    p = await _build_plugin(policy_inline={"rules": []})
    resp = await _req(p, "sensitive_read_tool", side_effects="read", requires_approval=True)
    assert resp["status"] == "ask_human"


@pytest.mark.asyncio
async def test_tool_metadata_requires_approval_false_short_circuits_to_allow():
    """A tool declaring requires_approval=False is auto-allowed even with destructive side_effects."""
    p = await _build_plugin(policy_inline={"rules": []})
    resp = await _req(p, "tool_x", side_effects="destructive", requires_approval=False)
    assert resp["status"] == "approved"


@pytest.mark.asyncio
async def test_session_allow_overrides_policy():
    """Caller can add a session-scoped always-allow."""
    p = await _build_plugin(policy_inline={"rules": []})
    p.add_session_allow("s-42", tool_id="slack_tool_v1", params_match={"channel": "#engineering*"})
    r = await _req(p, "slack_tool_v1", params={"channel": "#engineering"}, session_id="s-42")
    assert r["status"] == "approved"
    assert "session" in r["approver_id"]
    # Different session → no override
    r2 = await _req(p, "slack_tool_v1", params={"channel": "#engineering"}, session_id="s-43", side_effects="write")
    assert r2["status"] == "ask_human"


@pytest.mark.asyncio
async def test_first_match_wins():
    """First matching rule wins; later rules ignored."""
    policy = {
        "rules": [
            {"id": "ALLOW_FIRST", "match": {"tool_id": "x"}, "decision": "allow"},
            {"id": "DENY_LATER", "match": {"tool_id": "x"}, "decision": "deny"},
        ]
    }
    p = await _build_plugin(policy_inline=policy)
    r = await _req(p, "x")
    assert r["status"] == "approved"
    assert "ALLOW_FIRST" in r["approver_id"]


@pytest.mark.asyncio
async def test_user_identity_match():
    policy = {
        "rules": [
            {
                "id": "ADMIN_ALLOWED_WRITES",
                "match": {"side_effects_in": ["write"], "user_identity": {"role_in": ["admin"]}},
                "decision": "allow",
            }
        ]
    }
    p = await _build_plugin(policy_inline=policy)
    r_admin = await _req(p, "x", side_effects="write", user_identity={"role": "admin"})
    assert r_admin["status"] == "approved"
    r_user = await _req(p, "x", side_effects="write", user_identity={"role": "user"})
    assert r_user["status"] == "ask_human"


# ---------------------------------------------------------------------------
# HITLManager chain integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hitl_manager_chain_walks_on_ask_human():
    """Chain [permissions, webhook]: 'ask_human' from permissions delegates to webhook."""
    permissions = ClaudeCodePermissionsPlugin()
    await permissions.setup({"policy_inline": {"rules": []}})  # everything falls to defaults

    # Use a real class that satisfies HumanApprovalRequestPlugin runtime_checkable protocol.
    class _StubWebhook:
        plugin_id = "webhook_approval_v1"
        description = "stub"

        async def setup(self, config=None):
            pass

        async def teardown(self):
            pass

        async def request_approval(self, request):
            return {
                "request_id": request.get("request_id", "r?"),
                "status": "approved",
                "approver_id": "webhook:human1",
                "reason": "human approved",
                "timestamp": 1.0,
            }

    webhook = _StubWebhook()

    pm = MagicMock()

    async def _load(plugin_id, config):
        if plugin_id == "claude_code_permissions_v1":
            return permissions
        if plugin_id == "webhook_approval_v1":
            return webhook
        return None

    pm.get_plugin_instance = AsyncMock(side_effect=_load)

    mgr = HITLManager(
        pm,
        default_approver_id=None,
        approver_configurations={},
        default_approver_chain=["claude_code_permissions_v1", "webhook_approval_v1"],
    )

    # A "write" tool that the policy will ask_human about
    resp = await mgr.request_approval(
        {
            "request_id": "r1",
            "prompt": "do it?",
            "data_to_approve": {
                "tool_id": "any_write_tool",
                "params": {},
                "tool_metadata": {"side_effects": "write"},
            },
            "context": {},
            "timeout_seconds": 60,
        }
    )

    # Permissions said ask_human → webhook (the human approver) was called → approved.
    assert resp["status"] == "approved"
    assert resp["approver_id"] == "webhook:human1"


@pytest.mark.asyncio
async def test_hitl_manager_chain_decisive_first_link_short_circuits():
    """When permissions plugin decides outright, webhook is not invoked."""
    permissions = ClaudeCodePermissionsPlugin()
    await permissions.setup(
        {
            "policy_inline": {
                "rules": [
                    {
                        "id": "ALLOW_READS",
                        "match": {"side_effects_in": ["read"]},
                        "decision": "allow",
                    }
                ]
            }
        }
    )

    webhook_called = []

    class _StubWebhook:
        plugin_id = "webhook_approval_v1"
        description = "stub"

        async def setup(self, config=None):
            pass

        async def teardown(self):
            pass

        async def request_approval(self, request):
            webhook_called.append(request)
            return {"request_id": "x", "status": "approved", "approver_id": "webhook"}

    webhook = _StubWebhook()

    pm = MagicMock()

    async def _load(plugin_id, config):
        return permissions if plugin_id == "claude_code_permissions_v1" else webhook

    pm.get_plugin_instance = AsyncMock(side_effect=_load)

    mgr = HITLManager(
        pm,
        default_approver_id=None,
        approver_configurations={},
        default_approver_chain=["claude_code_permissions_v1", "webhook_approval_v1"],
    )

    resp = await mgr.request_approval(
        {
            "request_id": "r1",
            "prompt": "do it?",
            "data_to_approve": {
                "tool_id": "x",
                "params": {},
                "tool_metadata": {"side_effects": "read"},
            },
            "context": {},
            "timeout_seconds": 60,
        }
    )
    assert resp["status"] == "approved"
    assert "ALLOW_READS" in resp["approver_id"]
    # Webhook never reached
    assert webhook_called == []


@pytest.mark.asyncio
async def test_hitl_manager_chain_exhausted_with_ask_human_becomes_denial():
    """If the chain is exhausted while still 'ask_human', the manager returns deny + explanation."""
    permissions = ClaudeCodePermissionsPlugin()
    await permissions.setup({"policy_inline": {"rules": []}})  # always ask_human

    pm = MagicMock()
    pm.get_plugin_instance = AsyncMock(return_value=permissions)

    mgr = HITLManager(
        pm,
        default_approver_id=None,
        approver_configurations={},
        default_approver_chain=["claude_code_permissions_v1"],
    )

    resp = await mgr.request_approval(
        {
            "request_id": "r1",
            "prompt": "do it?",
            "data_to_approve": {
                "tool_id": "y",
                "params": {},
                "tool_metadata": {"side_effects": "destructive"},
            },
            "context": {},
            "timeout_seconds": 60,
        }
    )
    assert resp["status"] == "denied"
    assert "chain exhausted" in (resp.get("reason") or "").lower()
