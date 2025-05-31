"""Human-in-the-Loop (HITL) Abstractions and Implementations."""

from .abc import HumanApprovalRequestPlugin
from .manager import HITLManager
from .types import ApprovalRequest, ApprovalResponse, ApprovalStatus

# Concrete implementations will be registered via entry points
# from .impl.cli_approval import CliApprovalPlugin

__all__ = [
    "HumanApprovalRequestPlugin",
    "HITLManager",
    "ApprovalRequest",
    "ApprovalResponse",
    "ApprovalStatus",
    # "CliApprovalPlugin",
]
