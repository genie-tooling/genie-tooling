# src/genie_tooling/utils/placeholder_resolution.py
import json
import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _get_value_from_path(current_val: Any, path_parts: List[str], full_path_for_error: str) -> Any:
    """
    Recursively retrieves a value from a nested structure using a list of path parts.
    Supports dictionary key access and list indexing.
    Raises KeyError or IndexError on invalid paths.
    """
    # Base case: no more path parts to traverse, return the current value.
    if not path_parts:
        return current_val

    key_part = path_parts[0]
    remaining_parts = path_parts[1:]

    if isinstance(current_val, dict):
        if key_part not in current_val:
            logger.warning(f"Key '{key_part}' not found in dictionary for path '{full_path_for_error}'.")
            raise KeyError(f"Key '{key_part}' not found in scratchpad for path '{full_path_for_error}'.")
        return _get_value_from_path(current_val[key_part], remaining_parts, full_path_for_error)

    elif isinstance(current_val, list):
        try:
            idx = int(key_part)
            # Let the list indexing itself raise the IndexError
            return _get_value_from_path(current_val[idx], remaining_parts, full_path_for_error)
        except IndexError:
            logger.warning(f"List index {key_part} out of bounds for path '{full_path_for_error}'.")
            raise IndexError(f"List index {key_part} out of bounds for path '{full_path_for_error}'.")
        except (ValueError, TypeError):
            logger.warning(f"Invalid list index '{key_part}' (not an integer) for path '{full_path_for_error}'.")
            raise ValueError(f"Invalid list index '{key_part}' (not an integer). Full path for error context: '{full_path_for_error}'")

    # If it's not a dict or list, but there are still path parts, it's an error.
    else:
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

            # This can now raise KeyError, IndexError, TypeError which will propagate up
            resolved_value = _get_value_from_path(scratchpad["outputs"], path_parts, path_expression)

            # For string substitution, the resolved value must be converted to a string.
            # If it's a dict or list, JSON dump it.
            if isinstance(resolved_value, (dict, list)):
                return json.dumps(resolved_value)
            return str(resolved_value)

        try:
            return placeholder_pattern.sub(replacer, input_structure)
        except (KeyError, IndexError, TypeError, ValueError) as e:
            logger.warning(f"Error during placeholder replacement for string '{input_structure}': {e}")
            raise ValueError(f"Error resolving placeholder in string '{input_structure}': {e}") from e

    elif isinstance(input_structure, list):
        return [resolve_placeholders(item, scratchpad) for item in input_structure]
    elif isinstance(input_structure, dict):
        return {k: resolve_placeholders(v, scratchpad) for k, v in input_structure.items()}

    return input_structure