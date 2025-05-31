"""Types for Observability components."""
from typing import Any, Dict, Optional, TypedDict


class TraceEvent(TypedDict):
    """Represents a single event to be traced."""
    event_name: str
    data: Dict[str, Any]
    timestamp: float # Typically time.time() or loop.time()
    component: Optional[str] # e.g., "LLMProviderManager", "ToolInvoker:calculator_tool"
    correlation_id: Optional[str] # For linking related events
    # Potentially add: user_id, session_id, severity, etc.
