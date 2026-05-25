"""WebhookApprovalPlugin: HITL approver that delegates to an HTTP endpoint.

The webhook receives the approval request as a JSON POST body and is
expected to return a JSON response with at minimum a ``status`` field
(``"approved"`` or ``"denied"``). Suitable for integrating with Slack
interactive messages, Microsoft Teams adaptive cards, JIRA/ServiceNow
ticket workflows, or any custom approval UI — they all collapse to
"my endpoint returns JSON when a human clicks the button."

## Configuration

Two configuration shapes are supported.

**Single-endpoint (legacy)** — backwards compatible::

    url: str (required)
    headers: Dict[str, str] (optional)
    timeout_seconds: float (default 60.0)
    deny_on_error: bool (default True)

**Multi-endpoint routing (Phase 6A.5)** — pick a webhook per request::

    routes:
      - match:
          side_effects_in: ["destructive"]
        url: "https://hooks.example.com/destructive-approvers"
        headers: {Authorization: "Bearer ..."}
      - match:
          tool_id_in: ["github_*", "gitlab_*", "github:*"]
        url: "https://hooks.example.com/code-reviewers"
      - match:
          tool_id_in: ["kubectl_*"]
        url: "https://hooks.example.com/pagerduty-oncall"
    default_url: "https://hooks.example.com/general"
    default_headers: {Authorization: "Bearer ..."}
    timeout_seconds: 60.0
    deny_on_error: true

Match keys mirror ``claude_code_permissions_v1``: ``tool_id``,
``tool_id_in`` (fnmatch globs), ``side_effects``, ``side_effects_in``,
``params_match`` (dict; glob for string values), ``user_identity``.

If a request matches no route and ``default_url`` is unset, the plugin
denies with reason ``"no webhook route matched"``.

## Common config keys

  timeout_seconds: float (default 60.0)
      How long to wait for the approver to respond. On timeout the
      plugin returns status="error" so the calling agent can fail loud
      rather than treating the timeout as approval.
  deny_on_error: bool (default True)
      If the webhook is unreachable or returns a non-2xx response, deny
      by default. False would auto-approve on error — UNSAFE in
      production; only useful for development.
"""
from __future__ import annotations

import fnmatch
import logging
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

import httpx

from genie_tooling.hitl.abc import HumanApprovalRequestPlugin
from genie_tooling.hitl.types import ApprovalRequest, ApprovalResponse

logger = logging.getLogger(__name__)


class WebhookApprovalPlugin(HumanApprovalRequestPlugin):
    plugin_id: str = "webhook_approval_v1"
    description: str = (
        "HITL approver that POSTs each approval request to a configured URL "
        "and parses the JSON response. Supports per-request routing by tool "
        "ID, side-effects level, parameter patterns, and user identity."
    )

    _routes: List[Dict[str, Any]]
    _default_url: Optional[str] = None
    _default_headers: Dict[str, str]
    _timeout_seconds: float = 60.0
    _deny_on_error: bool = True
    _client: Optional[httpx.AsyncClient] = None

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        # Legacy single-endpoint config
        legacy_url = cfg.get("url")
        legacy_headers = dict(cfg.get("headers", {}))
        # Multi-endpoint routing
        self._routes = list(cfg.get("routes") or [])
        self._default_url = cfg.get("default_url") or legacy_url
        self._default_headers = dict(cfg.get("default_headers") or legacy_headers)
        self._timeout_seconds = float(cfg.get("timeout_seconds", 60.0))
        self._deny_on_error = bool(cfg.get("deny_on_error", True))
        self._client = httpx.AsyncClient(timeout=self._timeout_seconds)

        if not self._routes and not self._default_url:
            logger.error(
                "%s: no routes and no default_url/url configured. This plugin "
                "will deny every approval request because it has nowhere to ask.",
                self.plugin_id,
            )
        logger.info(
            f"{self.plugin_id}: Initialized. routes={len(self._routes)} "
            f"default_url={self._default_url!r} timeout={self._timeout_seconds}s "
            f"deny_on_error={self._deny_on_error}"
        )

    def _select_route(self, request: ApprovalRequest) -> Tuple[Optional[str], Dict[str, Any]]:
        """Phase 6A.5 — pick the URL + headers for this request, or fall back
        to defaults. Returns (None, _) when no route matches and no default
        URL is set."""
        data = request.get("data_to_approve") or {}
        ctx = request.get("context") or {}
        tool_id = data.get("tool_id", "")
        params = data.get("params") or {}
        tool_metadata = data.get("tool_metadata") or {}
        side_effects = str(tool_metadata.get("side_effects") or "unknown")
        user_identity = ctx.get("user_identity") if isinstance(ctx, dict) else None

        for route in self._routes:
            if not isinstance(route, dict):
                continue
            match = route.get("match") or {}
            if self._route_matches(match, tool_id, params, side_effects, user_identity):
                return route.get("url"), dict(route.get("headers") or self._default_headers)
        return self._default_url, dict(self._default_headers)

    @staticmethod
    def _route_matches(
        match: Mapping[str, Any],
        tool_id: str,
        params: Mapping[str, Any],
        side_effects: str,
        user_identity: Optional[Mapping[str, Any]],
    ) -> bool:
        if not match:
            return True
        if "tool_id" in match and tool_id != match["tool_id"]:
            return False
        if "tool_id_in" in match:
            allowed = match["tool_id_in"] or []
            if not any(fnmatch.fnmatchcase(tool_id, pat) for pat in allowed):
                return False
        if "side_effects" in match and side_effects != match["side_effects"]:
            return False
        if "side_effects_in" in match:
            allowed_se = match["side_effects_in"] or []
            if side_effects not in allowed_se:
                return False
        if "params_match" in match:
            spec = match["params_match"] or {}
            for key, expected in spec.items():
                if key not in params:
                    return False
                actual_val = params[key]
                if isinstance(expected, str) and any(c in expected for c in "*?["):
                    if not isinstance(actual_val, str) or not fnmatch.fnmatchcase(actual_val, expected):
                        return False
                else:
                    if actual_val != expected:
                        return False
        if "user_identity" in match:
            ui_match = match["user_identity"] or {}
            ui = user_identity or {}
            if "role" in ui_match and ui.get("role") != ui_match["role"]:
                return False
            if "role_in" in ui_match and ui.get("role") not in (ui_match["role_in"] or []):
                return False
        return True

    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        request_id = request.get("request_id", "")
        now = time.time()

        if not self._client:
            return self._error_response(
                request_id, "WebhookApprovalPlugin not initialized", now
            )

        url, headers = self._select_route(request)
        if not url:
            # Two flavours of "nowhere to send this":
            if not self._routes and not self._default_url:
                reason = "WebhookApprovalPlugin has no configured URL"
            else:
                reason = "no webhook route matched and no default_url configured"
            return self._error_response(request_id, reason, now)

        # Marshal the ApprovalRequest into JSON. ApprovalRequest may carry
        # non-JSON-friendly values inside data_to_approve; coerce gently.
        try:
            payload = self._jsonable(dict(request))
        except Exception as e:  # pragma: no cover - defensive
            return self._error_response(
                request_id, f"failed to serialize approval request: {e}", now
            )

        try:
            response = await self._client.post(
                url, json=payload, headers=headers
            )
            response.raise_for_status()
            body = response.json()
        except httpx.TimeoutException:
            return self._error_response(
                request_id,
                f"webhook timed out after {self._timeout_seconds}s",
                now,
            )
        except httpx.HTTPStatusError as e:
            return self._error_response(
                request_id,
                f"webhook returned HTTP {e.response.status_code}: {e.response.text[:200]}",
                now,
            )
        except Exception as e:
            return self._error_response(
                request_id, f"webhook call failed: {e!s}", now
            )

        # Body must be a JSON object with a 'status' field.
        if not isinstance(body, dict):
            return self._error_response(
                request_id,
                f"webhook returned non-object JSON: {type(body).__name__}",
                now,
            )

        status = str(body.get("status", "")).lower()
        if status not in ("approved", "denied"):
            return self._error_response(
                request_id,
                f"webhook returned invalid status {status!r}; expected 'approved' or 'denied'",
                now,
            )

        return ApprovalResponse(
            request_id=request_id,
            status="approved" if status == "approved" else "denied",
            approver_id=str(body.get("approver_id", "webhook")),
            reason=body.get("reason"),
            timestamp=now,
            data_approved=request.get("data_to_approve")
            if status == "approved"
            else None,
        )

    def _error_response(
        self, request_id: str, reason: str, ts: float
    ) -> ApprovalResponse:
        """Build a response when the webhook can't be consulted. Honors
        deny_on_error — defaults to denying, which is the safe-by-default
        corporate stance."""
        logger.warning(f"{self.plugin_id}: {reason}")
        if self._deny_on_error:
            return ApprovalResponse(
                request_id=request_id,
                status="denied",
                approver_id=f"{self.plugin_id}:fallback",
                reason=reason,
                timestamp=ts,
            )
        return ApprovalResponse(
            request_id=request_id,
            status="error",
            approver_id=f"{self.plugin_id}:fallback",
            reason=reason,
            timestamp=ts,
        )

    @staticmethod
    def _jsonable(obj: Any) -> Any:
        """Best-effort JSON-friendly conversion."""
        import json

        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            if isinstance(obj, dict):
                return {k: WebhookApprovalPlugin._jsonable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [WebhookApprovalPlugin._jsonable(v) for v in obj]
            return repr(obj)

    async def teardown(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
