"""Types for Human-in-the-Loop (HITL) components."""
from typing import Any, Dict, Literal, Optional, TypedDict

ApprovalStatus = Literal["pending", "approved", "denied", "timeout", "error"]

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
