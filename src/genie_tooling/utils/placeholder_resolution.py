# src/genie_tooling/utils/placeholder_resolution.py
import json
import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _get_value_from_path(current_val: Any, path_parts: List[str], full_path_for_error: str) -> Any:
    """
    Recursively retrieves a value from a nested structure using a list of path parts.
    Supports dictionary key access and list indexing (e.g., 'results[0]').
    """
    if not path_parts:
        return current_val

    key_part = path_parts[0]
    remaining_parts = path_parts[1:]

    # Check for list indexing like 'results[0]'
    index_match = re.match(r"(.+)\[(\d+)\]", key_part)
    if index_match:
        dict_key, index_str = index_match.groups()
        index = int(index_str)
        if not isinstance(current_val, dict) or dict_key not in current_val:
            raise KeyError(f"Key '{dict_key}' not found before indexing in path '{full_path_for_error}'.")
        target_list = current_val[dict_key]
        if not isinstance(target_list, list):
            raise TypeError(f"Attempted to index a non-list type for key '{dict_key}' in path '{full_path_for_error}'.")
        if index >= len(target_list):
            raise IndexError(f"List index {index} out of bounds for list at '{dict_key}' in path '{full_path_for_error}'.")
        return _get_value_from_path(target_list[index], remaining_parts, full_path_for_error)

    # Handle simple dictionary key access
    if isinstance(current_val, dict):
        if key_part not in current_val:
            raise KeyError(f"Key '{key_part}' not found in scratchpad for path '{full_path_for_error}'.")
        return _get_value_from_path(current_val[key_part], remaining_parts, full_path_for_error)
    elif isinstance(current_val, list):
        try:
            idx = int(key_part)
            if idx >= len(current_val):
                raise IndexError(f"List index {idx} out of bounds for path '{full_path_for_error}'.")
            return _get_value_from_path(current_val[idx], remaining_parts, full_path_for_error)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid list index '{key_part}' (not an integer). Full path: '{full_path_for_error}'") from None
    else:
        raise TypeError(f"Cannot access key/index '{key_part}' on non-dict/list type '{type(current_val).__name__}'.")


def resolve_placeholders(
    input_structure: Any,
    scratchpad: Dict[str, Any]
) -> Any:
    """
    Recursively resolves placeholders like '{outputs.variable.path[0].value}'
    within a given data structure using values from the scratchpad.
    """
    placeholder_pattern = re.compile(r"\{{1,2}\s*(?:outputs\.)?([\w.-]+(?:\[\d+\])?(?:\.[\w.-]+(?:\[\d+\])?)*)\s*\}{1,2}")

    if isinstance(input_structure, str):
        full_match = placeholder_pattern.fullmatch(input_structure.strip())
        if full_match:
            path_expression = full_match.group(1)
            path_parts = re.split(r"\.(?![^\[]*\])", path_expression)
            if "outputs" not in scratchpad:
                raise ValueError("Placeholder resolution failed: 'outputs' key not found in scratchpad.")
            # FIX: Do NOT wrap the error here. Let the original KeyError/IndexError propagate
            # as this is the expected behavior for a full-string replacement failure.
            return _get_value_from_path(scratchpad["outputs"], path_parts, path_expression)

        def replacer(match: re.Match) -> str:
            path_expression = match.group(1)
            path_parts = re.split(r"\.(?![^\[]*\])", path_expression)
            if "outputs" not in scratchpad:
                raise ValueError(f"Placeholder resolution failed: 'outputs' key not found for '{match.group(0)}'.")
            try:
                resolved_value = _get_value_from_path(scratchpad["outputs"], path_parts, path_expression)
                return str(resolved_value) if not isinstance(resolved_value, (dict, list)) else json.dumps(resolved_value)
            except (KeyError, IndexError, TypeError) as e:
                # FIX: For embedded placeholders, wrapping in a ValueError is appropriate
                # because the substitution itself is failing within a larger string context.
                logger.warning(f"Could not resolve placeholder '{match.group(0)}': {e}")
                raise ValueError(f"Error resolving placeholder '{match.group(0)}'") from e

        return placeholder_pattern.sub(replacer, input_structure)

    elif isinstance(input_structure, list):
        return [resolve_placeholders(item, scratchpad) for item in input_structure]
    elif isinstance(input_structure, dict):
        return {k: resolve_placeholders(v, scratchpad) for k, v in input_structure.items()}

    return input_structure