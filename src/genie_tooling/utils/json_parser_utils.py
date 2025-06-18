# src/genie_tooling/utils/json_parser_utils.py
"""
Utility functions for parsing and extracting JSON from text.
"""
import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


def extract_json_block(text: str) -> Optional[str]:
    """
    Finds and extracts the first valid JSON object or array from a string.

    This function is designed to be robust against common LLM outputs, where the
    JSON might be embedded within markdown code blocks or surrounded by other text.

    The search order is:
    1. A markdown code block explicitly marked as 'json'.
    2. Any markdown code block (```...```).
    3. The first parsable JSON object (`{...}`) or array (`[...]`) found in the text.

    If a higher-priority block is found but is malformed, the function will
    continue to search for lower-priority, valid blocks.

    Args:
        text: The input string to search for a JSON block.

    Returns:
        The extracted JSON string if a valid block is found, otherwise None.
    """
    if not text or not isinstance(text, str):
        return None

    # --- PRIORITY 1: Markdown 'json' block ---
    code_block_match_json = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.DOTALL)
    if code_block_match_json:
        potential_json = code_block_match_json.group(1).strip()
        try:
            json.loads(potential_json)  # Validate
            logger.debug("Extracted valid JSON from ```json ... ``` block.")
            return potential_json
        except json.JSONDecodeError:
            logger.debug(
                f"Found ```json``` block, but content was invalid. Will continue searching. Content: {potential_json[:100]}..."
            )
            # Do NOT return here. Fall through to other methods on the full string.

    # --- PRIORITY 2: Generic markdown block ---
    code_block_match_generic = re.search(r"```\s*([\s\S]*?)\s*```", text, re.DOTALL)
    if code_block_match_generic:
        potential_json = code_block_match_generic.group(1).strip()
        if potential_json.startswith(("{", "[")):  # Heuristic
            try:
                json.loads(potential_json)  # Validate
                logger.debug("Extracted valid JSON from generic ``` ... ``` block.")
                return potential_json
            except json.JSONDecodeError:
                logger.debug(
                    f"Found generic ``` ``` block, but content was invalid. Will continue searching. Content: {potential_json[:100]}..."
                )
                # Do NOT return here. Fall through.

    # --- PRIORITY 3: Raw string scan ---
    stripped_text = text.strip()
    decoder = json.JSONDecoder()

    # Find all possible start indices for objects or arrays
    potential_starts = [
        m.start() for m in re.finditer(r"\{|\[", stripped_text)
    ]

    for start_idx in potential_starts:
        try:
            # Attempt to decode from this starting point
            obj, end_idx = decoder.raw_decode(stripped_text[start_idx:])
            found_json_str = stripped_text[start_idx : start_idx + end_idx]
            logger.debug(f"Extracted valid JSON by raw_decode: {found_json_str[:100]}...")
            return found_json_str
        except json.JSONDecodeError:
            # This start index was not the beginning of a valid JSON object/array.
            # Continue to the next potential start index.
            continue

    logger.debug(f"Could not extract any valid JSON block from text after trying all methods: {text[:200]}...")
    return None
