"""Unit tests for the two production-grade HITL plugins (Phase 4 — B1, B2)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from genie_tooling.hitl.impl.policy_approval import PolicyAutoApproveHITLPlugin
from genie_tooling.hitl.impl.webhook_approval import WebhookApprovalPlugin


# ---------------------------------------------------------------------------
# WebhookApprovalPlugin tests — real local HTTP server
# ---------------------------------------------------------------------------


class _ApprovalServer:
    """Tiny HTTP server fixture that responds to POSTs with a configured
    JSON body (or simulates timeouts / errors). Built on aiohttp-like
    asyncio.start_server primitives so it has no external dep beyond
    Python stdlib + httpx (already a project dep)."""

    def __init__(self, response: Dict[str, Any], delay: float = 0.0,
                 status_code: int = 200):
        self.response = response
        self.delay = delay
        self.status_code = status_code
        self.received: list[Dict[str, Any]] = []
        self._server = None
        self.port = 0

    async def __aenter__(self):
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *args):
        self._server.close()
        await self._server.wait_closed()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/approve"

    async def _handle(self, reader, writer):
        # Read full HTTP request
        request_bytes = b""
        try:
            while b"\r\n\r\n" not in request_bytes:
                chunk = await reader.read(4096)
                if not chunk:
                    break
                request_bytes += chunk
            head, _, body = request_bytes.partition(b"\r\n\r\n")
            # Parse Content-Length if present and read remaining body bytes
            content_length = 0
            for line in head.split(b"\r\n"):
                if line.lower().startswith(b"content-length:"):
                    content_length = int(line.split(b":", 1)[1].strip())
            while len(body) < content_length:
                chunk = await reader.read(content_length - len(body))
                if not chunk:
                    break
                body += chunk
            try:
                self.received.append(json.loads(body.decode("utf-8")))
            except Exception:
                self.received.append({"raw": body.decode("utf-8", errors="replace")})

            if self.delay > 0:
                await asyncio.sleep(self.delay)

            body_bytes = json.dumps(self.response).encode("utf-8")
            status_line = {
                200: "HTTP/1.1 200 OK",
                500: "HTTP/1.1 500 Internal Server Error",
            }.get(self.status_code, f"HTTP/1.1 {self.status_code} ERR")
            writer.write(
                (
                    status_line + "\r\n"
                    "Content-Type: application/json\r\n"
                    f"Content-Length: {len(body_bytes)}\r\n"
                    "Connection: close\r\n\r\n"
                ).encode("utf-8")
                + body_bytes
            )
            await writer.drain()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass


@pytest.mark.asyncio
async def test_webhook_approves_when_endpoint_returns_approved():
    async with _ApprovalServer(
        {"status": "approved", "approver_id": "test_human", "reason": "looks fine"}
    ) as server:
        plugin = WebhookApprovalPlugin()
        await plugin.setup(config={"url": server.url, "timeout_seconds": 5.0})
        try:
            resp = await plugin.request_approval(
                {
                    "request_id": "req-1",
                    "prompt": "approve this tool call?",
                    "data_to_approve": {"tool_id": "calculator_tool", "params": {"x": 1}},
                }
            )
        finally:
            await plugin.teardown()

    assert resp["status"] == "approved"
    assert resp["approver_id"] == "test_human"
    assert resp["reason"] == "looks fine"
    # The webhook received the full request body
    assert len(server.received) == 1
    received = server.received[0]
    assert received["request_id"] == "req-1"
    assert received["data_to_approve"]["tool_id"] == "calculator_tool"


@pytest.mark.asyncio
async def test_webhook_denies_when_endpoint_returns_denied():
    async with _ApprovalServer(
        {"status": "denied", "approver_id": "approver", "reason": "policy violation"}
    ) as server:
        plugin = WebhookApprovalPlugin()
        await plugin.setup(config={"url": server.url, "timeout_seconds": 5.0})
        try:
            resp = await plugin.request_approval({"request_id": "r"})
        finally:
            await plugin.teardown()
    assert resp["status"] == "denied"
    assert resp["reason"] == "policy violation"


@pytest.mark.asyncio
async def test_webhook_timeout_denies_by_default():
    """Safety default: if the webhook is slow/unreachable, deny rather than
    silently approve."""
    async with _ApprovalServer({"status": "approved"}, delay=2.0) as server:
        plugin = WebhookApprovalPlugin()
        await plugin.setup(
            config={
                "url": server.url,
                "timeout_seconds": 0.3,
                "deny_on_error": True,
            }
        )
        try:
            resp = await plugin.request_approval({"request_id": "r"})
        finally:
            await plugin.teardown()
    assert resp["status"] == "denied"
    assert "timed out" in (resp.get("reason") or "").lower()


@pytest.mark.asyncio
async def test_webhook_returns_status_error_when_deny_on_error_false():
    """deny_on_error=False propagates errors as status=error so the caller
    can distinguish 'genuinely denied' from 'I couldn't decide'."""
    plugin = WebhookApprovalPlugin()
    await plugin.setup(
        config={
            "url": "http://127.0.0.1:1/never-listens",  # unreachable
            "timeout_seconds": 0.3,
            "deny_on_error": False,
        }
    )
    try:
        resp = await plugin.request_approval({"request_id": "r"})
    finally:
        await plugin.teardown()
    assert resp["status"] == "error"


@pytest.mark.asyncio
async def test_webhook_http_5xx_denies_safely():
    async with _ApprovalServer({"error": "boom"}, status_code=500) as server:
        plugin = WebhookApprovalPlugin()
        await plugin.setup(config={"url": server.url, "timeout_seconds": 5.0})
        try:
            resp = await plugin.request_approval({"request_id": "r"})
        finally:
            await plugin.teardown()
    assert resp["status"] == "denied"
    assert "500" in (resp.get("reason") or "")


@pytest.mark.asyncio
async def test_webhook_missing_url_denies_at_request_time():
    plugin = WebhookApprovalPlugin()
    await plugin.setup(config={})
    resp = await plugin.request_approval({"request_id": "r"})
    await plugin.teardown()
    assert resp["status"] == "denied"
    assert "no configured URL" in (resp.get("reason") or "")


# ---------------------------------------------------------------------------
# PolicyAutoApproveHITLPlugin tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_policy_inline_simple_allow():
    plugin = PolicyAutoApproveHITLPlugin()
    await plugin.setup(
        config={
            "policies": [
                {
                    "id": "ALLOW_CALCULATOR",
                    "match": {"tool_id": "calculator_tool"},
                    "decision": "approve",
                    "reason": "read-only math",
                },
            ]
        }
    )
    resp = await plugin.request_approval(
        {"request_id": "r1", "data_to_approve": {"tool_id": "calculator_tool"}}
    )
    assert resp["status"] == "approved"
    assert "ALLOW_CALCULATOR" in resp["approver_id"]
    assert resp["reason"] == "read-only math"


@pytest.mark.asyncio
async def test_policy_no_match_defaults_to_deny():
    plugin = PolicyAutoApproveHITLPlugin()
    await plugin.setup(
        config={
            "policies": [
                {
                    "id": "ALLOW_X",
                    "match": {"tool_id": "x"},
                    "decision": "approve",
                }
            ]
        }
    )
    resp = await plugin.request_approval(
        {"request_id": "r", "data_to_approve": {"tool_id": "y"}}
    )
    assert resp["status"] == "denied"
    assert "no policy matched" in (resp["reason"] or "")
    assert resp["approver_id"].endswith(":no_match")


@pytest.mark.asyncio
async def test_policy_glob_tool_id_match():
    plugin = PolicyAutoApproveHITLPlugin()
    await plugin.setup(
        config={
            "policies": [
                {
                    "id": "ALLOW_LOOKUPS",
                    "match": {"tool_id_in": ["lookup_*"]},
                    "decision": "approve",
                }
            ]
        }
    )
    resp_match = await plugin.request_approval(
        {"request_id": "r", "data_to_approve": {"tool_id": "lookup_physics_constant"}}
    )
    assert resp_match["status"] == "approved"

    resp_no_match = await plugin.request_approval(
        {"request_id": "r", "data_to_approve": {"tool_id": "write_database"}}
    )
    assert resp_no_match["status"] == "denied"


@pytest.mark.asyncio
async def test_policy_role_gated_match():
    plugin = PolicyAutoApproveHITLPlugin()
    await plugin.setup(
        config={
            "policies": [
                {
                    "id": "ALLOW_ADMIN_WRITES",
                    "match": {
                        "tool_id_in": ["sandboxed_fs_tool_v1"],
                        "user_identity": {"role_in": ["admin"]},
                    },
                    "decision": "approve",
                },
                {
                    "id": "DEFAULT_DENY",
                    "match": {},
                    "decision": "deny",
                    "reason": "default deny",
                },
            ]
        }
    )
    admin_resp = await plugin.request_approval(
        {
            "request_id": "r",
            "data_to_approve": {"tool_id": "sandboxed_fs_tool_v1"},
            "context": {"user_identity": {"role": "admin"}},
        }
    )
    assert admin_resp["status"] == "approved"
    assert "ALLOW_ADMIN_WRITES" in admin_resp["approver_id"]

    intern_resp = await plugin.request_approval(
        {
            "request_id": "r",
            "data_to_approve": {"tool_id": "sandboxed_fs_tool_v1"},
            "context": {"user_identity": {"role": "intern"}},
        }
    )
    assert intern_resp["status"] == "denied"
    assert "DEFAULT_DENY" in intern_resp["approver_id"]


@pytest.mark.asyncio
async def test_policy_first_match_wins(tmp_path: Path):
    """Order matters: an early ALLOW for the calculator must take
    precedence over a later DENY_ALL."""
    plugin = PolicyAutoApproveHITLPlugin()
    await plugin.setup(
        config={
            "policies": [
                {
                    "id": "ALLOW_CALC",
                    "match": {"tool_id": "calculator_tool"},
                    "decision": "approve",
                },
                {
                    "id": "DENY_EVERYTHING",
                    "match": {},
                    "decision": "deny",
                },
            ]
        }
    )
    resp = await plugin.request_approval(
        {"request_id": "r", "data_to_approve": {"tool_id": "calculator_tool"}}
    )
    assert resp["status"] == "approved"
    assert "ALLOW_CALC" in resp["approver_id"]


@pytest.mark.asyncio
async def test_policy_loads_from_yaml_file(tmp_path: Path):
    policy_file = tmp_path / "policy.yml"
    policy_file.write_text(
        yaml.dump(
            [
                {
                    "id": "ALLOW_CALC",
                    "match": {"tool_id": "calculator_tool"},
                    "decision": "approve",
                    "reason": "from file",
                }
            ]
        )
    )
    plugin = PolicyAutoApproveHITLPlugin()
    await plugin.setup(config={"policy_path": str(policy_file)})
    resp = await plugin.request_approval(
        {"request_id": "r", "data_to_approve": {"tool_id": "calculator_tool"}}
    )
    assert resp["status"] == "approved"
    assert resp["reason"] == "from file"


@pytest.mark.asyncio
async def test_policy_missing_file_defaults_to_deny_all():
    plugin = PolicyAutoApproveHITLPlugin()
    await plugin.setup(config={"policy_path": "/nonexistent/path/policy.yml"})
    resp = await plugin.request_approval(
        {"request_id": "r", "data_to_approve": {"tool_id": "calculator_tool"}}
    )
    assert resp["status"] == "denied"
