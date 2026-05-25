"""PolicyAutoApproveHITLPlugin: rule-based auto-approval for HITL requests.

For corporate deployments, a useful middle ground between the
``dev_auto_approve_hitl_v1`` (approves everything — unsafe in prod) and a
human-in-the-loop webhook (slow, requires a person) is a *deterministic
policy*: a YAML file that names which tool calls auto-approve and which
fall through to denial. Decisions are logged with the rule that matched
so the audit trail tells the story.

Policy YAML structure (default policy path: ``./hitl_policy.yml``)::

    # Policies are evaluated top-to-bottom; first match wins.
    # Each policy has match conditions and a decision (approve/deny).
    - id: ALLOW_READ_ONLY_TOOLS
      match:
        tool_id_in: ["calculator_tool", "lookup_*"]
      decision: approve
      reason: "read-only tools are unconditionally safe"

    - id: ALLOW_ADMINS_TO_WRITE
      match:
        tool_id_in: ["sandboxed_fs_tool_v1"]
        user_identity:
          role_in: ["admin"]
      decision: approve
      reason: "admin role permits sandboxed write"

    - id: DEFAULT_DENY
      match: {}        # empty match means "always"
      decision: deny
      reason: "no allow policy matched — default deny"

Match keys supported on the top-level ``match`` dict:
  * ``tool_id``: str — exact match against ``data_to_approve['tool_id']``
  * ``tool_id_in``: list[str] — membership test, supports ``"prefix_*"`` glob
  * ``user_identity``: dict — sub-conditions against ``request.context['user_identity']``
    (any dict of identity fields, e.g. role / tenant_id)

Match keys on ``user_identity`` sub-dict:
  * ``role``: str — exact match
  * ``role_in``: list[str] — membership

If no policy matches, the plugin returns status=denied with a "no policy
matched" reason. This is a safer default than "no policy matched -> approve".
"""
from __future__ import annotations

import asyncio
import fnmatch
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from genie_tooling.hitl.abc import HumanApprovalRequestPlugin
from genie_tooling.hitl.types import ApprovalRequest, ApprovalResponse

logger = logging.getLogger(__name__)


class PolicyAutoApproveHITLPlugin(HumanApprovalRequestPlugin):
    plugin_id: str = "policy_auto_approve_hitl_v1"
    description: str = (
        "Policy-based auto-approver. Approves/denies HITL requests based on "
        "a YAML policy file; each decision logged with the matching rule "
        "ID for audit."
    )

    _policy_path: Path
    _policies: List[Dict[str, Any]]

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        path = cfg.get("policy_path", "./hitl_policy.yml")
        self._policy_path = Path(path)
        # Allow inline policy specification for tests / programmatic use.
        inline_policies = cfg.get("policies")
        if inline_policies is not None:
            self._policies = list(inline_policies)
        else:
            self._policies = await asyncio.to_thread(self._load_policy_file)
        logger.info(
            f"{self.plugin_id}: Initialized with {len(self._policies)} policy rule(s) "
            f"from {self._policy_path if inline_policies is None else 'inline config'}."
        )

    def _load_policy_file(self) -> List[Dict[str, Any]]:
        if not self._policy_path.is_file():
            logger.warning(
                f"{self.plugin_id}: policy file {self._policy_path} not found. "
                "All approval requests will fall through to default deny."
            )
            return []
        try:
            with open(self._policy_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, list):
                logger.error(
                    f"{self.plugin_id}: policy file must be a YAML list; "
                    f"got {type(data).__name__}. No policies loaded."
                )
                return []
            return data
        except Exception as e:
            logger.error(
                f"{self.plugin_id}: failed to load policy file {self._policy_path}: {e}",
                exc_info=True,
            )
            return []

    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        request_id = request.get("request_id", "")
        now = time.time()

        data = request.get("data_to_approve") or {}
        context = request.get("context") or {}
        user_identity = context.get("user_identity") if isinstance(context, dict) else None

        for policy in self._policies:
            if not isinstance(policy, dict):
                continue
            match = policy.get("match", {})
            if self._policy_matches(match, data, user_identity):
                decision = str(policy.get("decision", "")).lower()
                rule_id = str(policy.get("id", "<unnamed>"))
                reason = policy.get("reason") or f"policy {rule_id} matched"
                if decision == "approve":
                    return ApprovalResponse(
                        request_id=request_id,
                        status="approved",
                        approver_id=f"{self.plugin_id}:{rule_id}",
                        reason=reason,
                        timestamp=now,
                        data_approved=data,
                    )
                if decision == "deny":
                    return ApprovalResponse(
                        request_id=request_id,
                        status="denied",
                        approver_id=f"{self.plugin_id}:{rule_id}",
                        reason=reason,
                        timestamp=now,
                    )
                # Malformed decision — skip to next policy
                logger.warning(
                    f"{self.plugin_id}: policy {rule_id} has invalid "
                    f"decision={decision!r}; expected 'approve' or 'deny'. "
                    "Skipping."
                )

        # No policy matched — default deny, safe-by-default.
        return ApprovalResponse(
            request_id=request_id,
            status="denied",
            approver_id=f"{self.plugin_id}:no_match",
            reason="no policy matched; default deny",
            timestamp=now,
        )

    @staticmethod
    def _policy_matches(
        match: Dict[str, Any],
        data: Dict[str, Any],
        user_identity: Optional[Dict[str, Any]],
    ) -> bool:
        if not match:
            return True

        # tool_id exact match
        if "tool_id" in match:
            if data.get("tool_id") != match["tool_id"]:
                return False

        # tool_id_in: list with glob support
        if "tool_id_in" in match:
            tid = data.get("tool_id", "")
            allowed = match["tool_id_in"] or []
            if not any(fnmatch.fnmatchcase(tid, pat) for pat in allowed):
                return False

        # user_identity sub-match
        if "user_identity" in match:
            ui_match = match["user_identity"] or {}
            ui = user_identity or {}
            if "role" in ui_match and ui.get("role") != ui_match["role"]:
                return False
            if "role_in" in ui_match:
                if ui.get("role") not in (ui_match["role_in"] or []):
                    return False

        return True

    async def teardown(self) -> None:
        pass
