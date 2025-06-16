# src/genie_tooling/prompts/types.py
"""Types for Prompt Management components."""
from typing import Any, Dict, List, Optional, TypedDict, Union

from genie_tooling.llm_providers.types import ChatMessage

PromptData = Dict[str, Any]
FormattedPrompt = Union[str, List[ChatMessage]]

class PromptIdentifier(TypedDict):
    name: str
    version: Optional[str]
    description: Optional[str]
