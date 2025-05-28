"""Data structures for Tool Lookup results."""
from typing import Any, Dict, Optional


class RankedToolResult:
    """Represents a tool found by a lookup provider."""
    def __init__(self, tool_identifier: str, score: float, matched_tool_data: Optional[Dict[str, Any]] = None, description_snippet: Optional[str] = None):
        self.tool_identifier: str = tool_identifier
        self.score: float = score # Relevance score (higher is better)

        # The data that the provider used for matching (e.g., formatted description, keywords).
        # This helps the LLM or orchestrator understand why this tool was chosen.
        self.matched_tool_data: Optional[Dict[str, Any]] = matched_tool_data or {}

        # A short snippet or reason for the match, if provided by the lookup provider.
        self.description_snippet: Optional[str] = description_snippet

    def __repr__(self) -> str:
        return f"RankedToolResult(id='{self.tool_identifier}', score={self.score:.4f})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_identifier": self.tool_identifier,
            "score": self.score,
            "matched_tool_data": self.matched_tool_data,
            "description_snippet": self.description_snippet,
        }
