# src/genie_tooling/conversation/types.py
"""Types for Conversation State Management components."""
from typing import Any, Dict, List, Optional, TypedDict

from genie_tooling.llm_providers.types import ChatMessage

class ConversationState(TypedDict):
    session_id: str
    history: List[ChatMessage]
    metadata: Optional[Dict[str, Any]] # For user-specific data, timestamps, etc.
    # Potentially: user_id, last_accessed, summary, active_tool_calls