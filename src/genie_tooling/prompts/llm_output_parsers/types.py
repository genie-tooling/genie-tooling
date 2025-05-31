# src/genie_tooling/llm_output_parsers/types.py
"""Types for LLM Output Parser components."""
from typing import Any

ParsedOutput = Any # The output can be of any type after parsing (dict, list, Pydantic model, etc.)
