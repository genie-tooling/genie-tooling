# src/genie_tooling/command_processors/types.py
from typing import Any, Dict, Optional, TypedDict


class CommandProcessorResponse(TypedDict, total=False):
    """
    Standardized response from a CommandProcessorPlugin.
    """
    chosen_tool_id: Optional[str]       # Identifier of the tool selected by the processor.
    extracted_params: Optional[Dict[str, Any]] # Parameters extracted for the chosen tool.
    llm_thought_process: Optional[str]  # Explanation or reasoning from the processor (e.g., LLM's chain of thought).
    error: Optional[str]                # Error message if processing failed.
    raw_response: Optional[Any]         # Original response from an LLM if one was used.
