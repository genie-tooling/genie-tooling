# src/genie_tooling/tools/impl/custom_text_parameter_extractor.py
import logging
import re
from typing import Any, Dict, List, Optional

from genie_tooling.decorators import tool

logger = logging.getLogger(__name__)

@tool
async def custom_text_parameter_extractor(
    text_content: str,
    parameter_names: List[str],
    regex_patterns: List[str],
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    DataExtractor: Extracts specific named pieces of information (e.g., scores, numbers, names, dates)
    from a larger block of text using regular expressions. Use this *after* obtaining full text content.
    `parameter_names` and `regex_patterns` are parallel lists; each `regex_patterns[i]` is used to find the value for `parameter_names[i]`.
    Output: A dictionary where keys are from `parameter_names`, and values are the extracted strings/numbers, or None if not found/error.
    Args:
        text_content (str): The block of text to extract parameters from.
        parameter_names (List[str]): A list of output keys for the extracted values.
        regex_patterns (List[str]): A list of Python regex patterns. Each pattern must have one capturing group `(...)` for the value.
        context (Optional[Dict[str, Any]]): Invocation context.
    """
    if len(parameter_names) != len(regex_patterns):
        err_msg = f"Mismatched argument lengths: {len(parameter_names)} names provided, but {len(regex_patterns)} patterns. They must be parallel lists."
        logger.warning(f"{custom_text_parameter_extractor._tool_metadata_['identifier']}: {err_msg}")
        return {"error": err_msg}

    extracted_values: Dict[str, Any] = {}
    if not text_content or not isinstance(text_content, str):
        logger.warning(f"{custom_text_parameter_extractor._tool_metadata_['identifier']}: Input 'text_content' is empty or not a string.")
        for name in parameter_names:
            extracted_values[name] = None
        return extracted_values

    for name, pattern_str in zip(parameter_names, regex_patterns, strict=False):
        if not name or not pattern_str:
            unnamed_key = f"extraction_error_param_{len(extracted_values)}"
            extracted_values[name or unnamed_key] = None
            logger.warning(f"Invalid parameter specification. Name: '{name}', Pattern: '{pattern_str}'.")
            continue

        try:
            match = re.search(pattern_str, text_content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if match:
                value_str = match.group(1) if match.groups() else match.group(0)
                value_str = value_str.strip()
                if value_str.replace(".", "", 1).replace("-","",1).isdigit():
                    if "." in value_str:
                        try:
                            extracted_values[name] = float(value_str)
                        except ValueError:
                            extracted_values[name] = value_str
                    else:
                        try:
                            extracted_values[name] = int(value_str)
                        except ValueError:
                            extracted_values[name] = value_str
                else:
                    extracted_values[name] = value_str
            else:
                extracted_values[name] = None
        except re.error as e_re:
            logger.error(f"{custom_text_parameter_extractor._tool_metadata_['identifier']}: Regex error for parameter '{name}' with pattern '{pattern_str}': {e_re}", exc_info=False)
            extracted_values[name] = None
        except Exception as e:
            logger.error(f"{custom_text_parameter_extractor._tool_metadata_['identifier']}: Unexpected error extracting parameter '{name}': {e}", exc_info=True)
            extracted_values[name] = None

    logger.debug(f"{custom_text_parameter_extractor._tool_metadata_['identifier']}: Extracted values: {extracted_values}")
    return extracted_values
