"""WebhookApprovalPlugin: HITL approver that delegates to an HTTP endpoint.

The webhook receives the approval request as a JSON POST body and is
expected to return a JSON response with at minimum a ``status`` field
(``"approved"`` or ``"denied"``). Suitable for integrating with Slack
interactive messages, Microsoft Teams adaptive cards, JIRA/ServiceNow
ticket workflows, or any custom approval UI — they all collapse to
"my endpoint returns JSON when a human clicks the button."

Configuration (passed via ``hitl_approver_configurations``)::

    url: str (required)
        The endpoint the plugin POSTs the approval request to.
    headers: Dict[str, str] (optional)
        Extra HTTP headers (e.g. an authorization token).
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

import logging
import time
from typing import Any, Dict, Optional

import httpx

from genie_tooling.hitl.abc import HumanApprovalRequestPlugin
from genie_tooling.hitl.types import ApprovalRequest, ApprovalResponse

logger = logging.getLogger(__name__)


class WebhookApprovalPlugin(HumanApprovalRequestPlugin):
    plugin_id: str = "webhook_approval_v1"
    description: str = (
        "HITL approver that POSTs each approval request to a configured URL "
        "and parses the JSON response. Suitable for Slack/Teams/JIRA "
        "integrations or any custom approval UI."
    )

    _url: Optional[str] = None
    _headers: Dict[str, str]
    _timeout_seconds: float = 60.0
    _deny_on_error: bool = True
    _client: Optional[httpx.AsyncClient] = None

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._url = cfg.get("url")
        if not self._url:
            logger.error(
                "%s: required config key 'url' is missing. This plugin will "
                "deny every approval request because it has nowhere to ask.",
                self.plugin_id,
            )
        self._headers = dict(cfg.get("headers", {}))
        self._timeout_seconds = float(cfg.get("timeout_seconds", 60.0))
        self._deny_on_error = bool(cfg.get("deny_on_error", True))
        self._client = httpx.AsyncClient(timeout=self._timeout_seconds)
        logger.info(
            f"{self.plugin_id}: Initialized. url={self._url!r} "
            f"timeout={self._timeout_seconds}s deny_on_error={self._deny_on_error}"
        )

    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        request_id = request.get("request_id", "")
        now = time.time()

        if not self._url or not self._client:
            return self._error_response(
                request_id, "WebhookApprovalPlugin has no configured URL", now
            )

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
                self._url, json=payload, headers=self._headers
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
