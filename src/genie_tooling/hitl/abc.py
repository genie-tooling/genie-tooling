"""Abstract Base Class/Protocol for HumanApprovalRequest Plugins."""
import logging
from typing import Protocol, runtime_checkable

from genie_tooling.core.types import Plugin

from .types import ApprovalRequest, ApprovalResponse

logger = logging.getLogger(__name__)

@runtime_checkable
class HumanApprovalRequestPlugin(Plugin, Protocol):
    """Protocol for a plugin that requests human approval for an action."""
    plugin_id: str # From Plugin

    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        """
        Requests human approval for a given action or data.
        Implementations will vary (e.g., CLI prompt, web UI notification, API call).
        """
        logger.warning(f"HumanApprovalRequestPlugin '{self.plugin_id}' request_approval method not fully implemented.")
        # Default to denied if not implemented
        return ApprovalResponse(
            request_id=request.request_id,
            status="denied",
            approver_id="system_default",
            reason="Plugin not implemented."
        )
