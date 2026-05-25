"""Types for Human-in-the-Loop (HITL) components."""
from typing import Any, Dict, Literal, Optional, TypedDict

# "ask_human" (Phase 6A.5b) is emitted by deterministic-policy approvers
# (e.g. `claude_code_permissions_v1`) to signal "policy did not decide;
# delegate to the next approver in the chain." The HITLManager walks the
# configured chain and stops on the first non-"ask_human" response.
ApprovalStatus = Literal[
    "pending",
    "approved",
    "denied",
    "timeout",
    "error",
    "ask_human",
]

class ApprovalRequest(TypedDict):
    """Represents a request for human approval."""
    request_id: str # Unique ID for this request, can be auto-generated
    prompt: str # Message shown to the human approver
    data_to_approve: Dict[str, Any] # The data or action needing approval
    context: Optional[Dict[str, Any]] # Additional context for the approver
    timeout_seconds: Optional[int] # How long to wait for approval

class ApprovalResponse(TypedDict):
    """Represents the response from a human approval request."""
    request_id: str
    status: ApprovalStatus
    approver_id: Optional[str] # Identifier of the human who approved/denied
    reason: Optional[str] # Reason for denial or additional comments
    timestamp: Optional[float] # When the decision was made
