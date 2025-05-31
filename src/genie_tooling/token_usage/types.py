"""Types for Token Usage Tracking components."""
from typing import Optional, TypedDict


class TokenUsageRecord(TypedDict, total=False):
    """Represents a single LLM token usage event."""
    provider_id: str
    model_name: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    timestamp: float # time.time()
    call_type: Optional[str] # "chat", "generate"
    user_id: Optional[str]
    session_id: Optional[str]
    custom_tags: Optional[dict]
