# src/genie_tooling/llm_output_parsers/abc.py
"""Abstract Base Class for LLMOutputParser Plugins."""
import json
import logging
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from genie_tooling.core.types import Plugin

from .types import ParsedOutput

logger = logging.getLogger(__name__)

@runtime_checkable
class LLMOutputParserPlugin(Plugin, Protocol):
    """Protocol for a plugin that parses the text output of an LLM."""
    plugin_id: str

    def parse(self, text_output: str, schema: Optional[Any] = None) -> ParsedOutput:
        """
        Parses the LLM's text output.
        Args:
            text_output: The raw text string from the LLM.
            schema: Optional schema (e.g., Pydantic model, JSON schema) to guide parsing.
        Returns:
            The parsed data, which could be a dict, list, Pydantic model instance, etc.
        Raises:
            ValueError or a custom parsing exception if parsing fails.
        """
        logger.warning(f"LLMOutputParserPlugin '{self.plugin_id}' parse method not implemented.")
        # Basic fallback: return as is, or attempt JSON if it looks like it
        if text_output.strip().startswith("{") and text_output.strip().endswith("}"):
            try:
                return json.loads(text_output) # type: ignore
            except:
                pass
        return text_output # type: ignore