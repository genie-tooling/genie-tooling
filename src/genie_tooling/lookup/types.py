"""Data structures for Tool Lookup results."""
from typing import Any, Dict, List, Optional


class RankedToolResult:
    """Represents a tool found by a lookup provider, with diagnostic info."""
    def __init__(
        self,
        tool_identifier: str,
        score: float,
        matched_tool_data: Optional[Dict[str, Any]] = None,
        description_snippet: Optional[str] = None,
        matched_keywords: Optional[List[str]] = None,
        similarity_score_details: Optional[Dict[str, float]] = None,
    ):
        self.tool_identifier: str = tool_identifier
        self.score: float = score # Relevance score (higher is better)
        self.matched_tool_data: Optional[Dict[str, Any]] = matched_tool_data or {}
        self.description_snippet: Optional[str] = description_snippet
        self.matched_keywords: Optional[List[str]] = matched_keywords
        self.similarity_score_details: Optional[Dict[str, float]] = similarity_score_details

    def __repr__(self) -> str:
        return f"RankedToolResult(id='{self.tool_identifier}', score={self.score:.4f})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_identifier": self.tool_identifier,
            "score": self.score,
            "matched_tool_data": self.matched_tool_data,
            "description_snippet": self.description_snippet,
            "matched_keywords": self.matched_keywords,
            "similarity_score_details": self.similarity_score_details,
        }
