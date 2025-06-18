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

    Args:
        text: The input string to search for a JSON block.

    Returns:
        The extracted JSON string if a valid block is found, otherwise None.
    """
    if not text or not isinstance(text, str):
        return None

    # 1. Try to find JSON within ```json ... ```
    code_block_match_json = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.DOTALL)
    if code_block_match_json:
        potential_json = code_block_match_json.group(1).strip()
        try:
            json.loads(potential_json)  # Validate
            logger.debug("Extracted JSON from ```json ... ``` block.")
            return potential_json
        except json.JSONDecodeError:
            logger.debug(
                f"Found ```json``` block, but content is not valid JSON: {potential_json[:100]}..."
            )

    # 2. Try to find JSON within generic ``` ... ```
    code_block_match_generic = re.search(r"```\s*([\s\S]*?)\s*```", text, re.DOTALL)
    if code_block_match_generic:
        potential_json = code_block_match_generic.group(1).strip()
        if potential_json.startswith(("{", "[")):  # Heuristic
            try:
                json.loads(potential_json)  # Validate
                logger.debug("Extracted JSON from generic ``` ... ``` block.")
                return potential_json
            except json.JSONDecodeError:
                logger.debug(
                    f"Found generic ``` ``` block, but content is not valid JSON: {potential_json[:100]}..."
                )

    # 3. If no code block, try to find the first JSON object or array in the possibly "dirty" string
    stripped_text = text.strip()
    decoder = json.JSONDecoder()

    first_obj_idx = stripped_text.find("{")
    first_arr_idx = stripped_text.find("[")

    start_indices = []
    if first_obj_idx != -1:
        start_indices.append(first_obj_idx)
    if first_arr_idx != -1:
        start_indices.append(first_arr_idx)

    if not start_indices:
        logger.debug("No '{' or '[' found in stripped text for general extraction.")
        return None

    start_indices.sort()

    for start_idx in start_indices:
        try:
            # raw_decode will parse one complete JSON object/array and return its end position.
            _obj, end_idx = decoder.raw_decode(stripped_text[start_idx:])
            found_json_str = stripped_text[start_idx : start_idx + end_idx]
            logger.debug(
                f"Extracted JSON by raw_decode: {found_json_str[:100]}..."
            )
            return found_json_str
        except json.JSONDecodeError:
            # This start index was not the beginning of a valid JSON structure, so we continue
            # to the next potential start index (if any).
            logger.debug(
                f"No valid JSON found by raw_decode starting at index {start_idx} of stripped text."
            )
            continue

    logger.debug(f"Could not extract any valid JSON block from text: {text[:200]}...")
    return None
