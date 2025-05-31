"""Types for Guardrail components."""
from typing import Any, Dict, Literal, Optional, TypedDict

GuardrailAction = Literal["allow", "block", "warn"]

class GuardrailViolation(TypedDict):
    """Represents the outcome of a guardrail check."""
    action: GuardrailAction
    reason: Optional[str] # Explanation for the action, especially for block/warn
    guardrail_id: Optional[str] # ID of the guardrail that triggered this
    details: Optional[Dict[str, Any]] # Additional context about the violation
