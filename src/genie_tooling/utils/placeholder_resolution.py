# src/genie_tooling/utils/placeholder_resolution.py
import logging
import json
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

def _get_value_from_path(current_val: Any, path_parts: List[str], full_path_for_error: str) -> Any:
    """
    Recursively retrieves a value from a nested structure using a list of path parts.
    Supports dictionary key access and list indexing.
    """
    if not path_parts:
        return current_val

    key_part = path_parts[0]
    remaining_parts = path_parts[1:]

    if isinstance(current_val, dict):
        if key_part not in current_val:
            # If the key is missing, it could mean an optional extraction failed.
            # Return None in this case, and let schema validation handle if it was required.
            logger.debug(f"Key '{key_part}' not found in dictionary for path '{full_path_for_error}'. Returning None for this part.")
            return None
        return _get_value_from_path(current_val[key_part], remaining_parts, full_path_for_error)
    elif isinstance(current_val, list):
        try:
            idx = int(key_part)
            if not (-(len(current_val)) <= idx < len(current_val)):
                logger.debug(f"List index {idx} out of bounds (length {len(current_val)}) for path '{full_path_for_error}'. Returning None for this part.")
                return None # Index out of bounds
            return _get_value_from_path(current_val[idx], remaining_parts, full_path_for_error)
        except (ValueError, TypeError):
            logger.warning(f"Invalid list index '{key_part}' (not an integer) for path '{full_path_for_error}'.")
            raise ValueError(f"Invalid list index '{key_part}' (not an integer). Full path for error context: '{full_path_for_error}'")
    else:
        # If current_val is not a dict or list, but there are more path_parts, it's an error.
        # If no remaining_parts, it means we've reached the end, and current_val is the leaf value.
        if not remaining_parts:
             return current_val # This is the resolved leaf value
        logger.warning(f"Cannot access key/index '{key_part}' on non-dict/list type '{type(current_val).__name__}' for path '{full_path_for_error}'.")
        raise TypeError(f"Cannot access key/index '{key_part}' on non-dict/list type '{type(current_val).__name__}'. Full path for error context: '{full_path_for_error}'")

def resolve_placeholders(
    input_structure: Any,
    scratchpad: Dict[str, Any]
) -> Any:
    """
    Recursively resolves placeholders like '{outputs.variable_name.path.to.value}'
    or '{{outputs.variable_name.path.to.value}}' (supporting single or double braces)
    within a given data structure (string, list, or dict) using values from the scratchpad.

    This function now robustly handles placeholders embedded within larger strings.
    """
    placeholder_pattern = re.compile(r"\{{1,2}outputs\.([\w.-]+(?:\.[\w.-]+)*)\}{1,2}")

    if isinstance(input_structure, str):
        # Check for a full match first to handle cases where the entire string is a placeholder.
        # This allows returning complex types (dicts, lists) directly.
        full_match = placeholder_pattern.fullmatch(input_structure.strip())
        if full_match: # The entire string is a placeholder
            path_expression = full_match.group(1)
            path_parts = path_expression.split('.')
            if "outputs" not in scratchpad:
                raise ValueError("Placeholder resolution failed: 'outputs' key not found in scratchpad.")
            return _get_value_from_path(scratchpad["outputs"], path_parts, path_expression)

        # If not a full match, use re.sub to replace embedded placeholders.
        def replacer(match: re.Match) -> str:
            path_expression = match.group(1)
            path_parts = path_expression.split('.')
            if "outputs" not in scratchpad:
                raise ValueError(f"Placeholder resolution failed: 'outputs' key not found in scratchpad for placeholder '{match.group(0)}'.")
            resolved_value = _get_value_from_path(scratchpad["outputs"], path_parts, path_expression)

            try:
                # For string substitution, the resolved value must be converted to a string.
                # If it's a dict or list, JSON dump it.
                if isinstance(resolved_value, (dict, list)):
                    return json.dumps(resolved_value)
                return str(resolved_value)
            except (TypeError, ValueError) as e: # Catch errors from _get_value_from_path if path is malformed for the data type
                logger.warning(f"Error during path traversal for placeholder '{input_structure}': {e}")
                raise ValueError(f"Error resolving placeholder '{input_structure}': {e}") from e

        return placeholder_pattern.sub(replacer, input_structure)

    elif isinstance(input_structure, list):
        return [resolve_placeholders(item, scratchpad) for item in input_structure]
    elif isinstance(input_structure, dict):
        return {k: resolve_placeholders(v, scratchpad) for k, v in input_structure.items()}

    return input_structure