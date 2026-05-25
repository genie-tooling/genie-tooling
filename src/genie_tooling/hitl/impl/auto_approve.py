"""DevAutoApproveHITLPlugin: HITL approver that always approves.

**Intended for dev/test only.** PlanAndExecuteAgent (and any other component
that calls ``genie.human_in_loop.request_approval``) unconditionally invokes
the HITL flow — without an approver, those calls return "denied" and the
agent fails. This plugin lets developers and CI test suites bypass that
gate by registering a non-interactive approver that always returns
"approved".

For corporate / production use, use one of the policy- or webhook-based
HITL plugins instead so approval decisions are auditable. The setup() of
this plugin emits a loud warning if MiddlewareConfig.environment is
"production".

Plugin IDs:
  * Primary: ``dev_auto_approve_hitl_v1``
  * Deprecated alias: ``auto_approve_hitl_v1`` (kept one cycle for
    migration; emits a deprecation warning when resolved by that name).
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from genie_tooling.hitl.abc import HumanApprovalRequestPlugin
from genie_tooling.hitl.types import ApprovalRequest, ApprovalResponse

logger = logging.getLogger(__name__)


class DevAutoApproveHITLPlugin(HumanApprovalRequestPlugin):
    """HITL approver that auto-approves every request.

    Two pre-set production-safety hatches:
      1. Setup emits a WARNING-level log every time it initializes,
         making it visible in CI/dev logs that approvals are non-interactive.
      2. If MiddlewareConfig.environment == "production", setup() emits an
         ERROR-level log naming the plugin and telling operators to swap
         to a real approver.
    """

    plugin_id: str = "dev_auto_approve_hitl_v1"
    description: str = (
        "HITL approver that auto-approves every request. DEV/TEST ONLY — "
        "emits a loud warning if MiddlewareConfig.environment is 'production'."
    )

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        # The PluginManager injects `plugin_manager` into config; the
        # MiddlewareConfig lives on the Genie instance. We don't have a
        # direct handle here, but the resolver wires `environment` into
        # this plugin's config block via hitl_approver_configurations.
        environment = cfg.get("environment", "development")
        if environment == "production":
            logger.error(
                "%s: configured as the HITL approver in a 'production' "
                "environment. Every approval request will auto-approve. "
                "Swap to a real approver (cli_approval_plugin_v1, or a "
                "webhook/policy-based one).",
                self.plugin_id,
            )
        else:
            logger.warning(
                "%s: dev-mode HITL approver active — all approval requests "
                "will auto-approve. Safe in dev/test; do NOT ship to "
                "production with this approver.",
                self.plugin_id,
            )

    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        return ApprovalResponse(
            request_id=request.get("request_id", ""),
            status="approved",
            approver_id=self.plugin_id,
            reason="auto-approved by DevAutoApproveHITLPlugin",
            timestamp=time.time(),
            data_approved=request.get("data_to_approve"),
        )

    async def teardown(self) -> None:
        pass


class _DeprecatedAutoApproveHITLAlias(DevAutoApproveHITLPlugin):
    """Backward-compat shim. Same behavior as DevAutoApproveHITLPlugin but
    warns at setup that the plugin_id was renamed. Will be removed in a
    future minor release."""

    plugin_id: str = "auto_approve_hitl_v1"
    description: str = "DEPRECATED alias for dev_auto_approve_hitl_v1."

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        logger.warning(
            "%s is a deprecated alias for 'dev_auto_approve_hitl_v1'. "
            "Update your config (e.g. MiddlewareConfig.default_hitl_approver_id "
            "or FeatureSettings.hitl_approver) to use the new ID.",
            self.plugin_id,
        )
        await super().setup(config)


# Back-compat: keep the AutoApproveHITLPlugin name exported so any
# import-by-name continues to work for one cycle.
AutoApproveHITLPlugin = DevAutoApproveHITLPlugin
