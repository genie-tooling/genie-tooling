### src/genie_tooling/logging_monitoring/redaction/schema_aware.py
"""Schema-aware data redaction logic for basic sensitive field masking."""
import logging
import re
from typing import Any, Dict, Optional

from genie_tooling.redactors.abc import Redactor

logger = logging.getLogger(__name__)

SENSITIVE_FIELD_MARKER_KEY = "x-sensitive" # Custom JSON Schema extension key
SENSITIVE_FORMAT_VALUES = ["password", "secret", "api_key", "token", "credit_card", "ssn", "credentials", "auth_token", "private_key"]
REDACTION_PLACEHOLDER_VALUE = "[REDACTED]"

# Regex to find sensitive keywords within a key name, case-insensitive
COMMON_SENSITIVE_KEY_NAMES_REGEX = re.compile(
    r"(api_*key|secret|token|password|passwd|credentials|auth|private_*key)", re.IGNORECASE
)


def sanitize_data_with_schema_based_rules(
    data: Any,
    schema: Optional[Dict[str, Any]] = None,
    redact_matching_key_names: bool = True
) -> Any:
    """
    Recursively sanitizes data.
    1. If a schema is provided, it uses 'x-sensitive: true' or 'format' hints in the schema,
       respecting the schema's 'type' if specified.
    2. If `redact_matching_key_names` is True, it also redacts dictionary values if their
       keys contain common sensitive patterns, even without schema hints.
    """
    if schema:
        schema_type = schema.get("type")
        data_type_matches_schema_type = True # Default to true if schema has no type or we don't check it strictly

        if schema_type: # Only check type if schema specifies one
            current_data_type_ok = False
            schema_types_list = [schema_type] if isinstance(schema_type, str) else (schema_type if isinstance(schema_type, list) else [])

            for s_type in schema_types_list:
                if s_type == "string" and isinstance(data, str):
                    current_data_type_ok = True
                    break
                elif s_type == "integer" and isinstance(data, int):
                    current_data_type_ok = True
                    break
                elif s_type == "number" and isinstance(data, (int, float)):
                    current_data_type_ok = True
                    break
                elif s_type == "boolean" and isinstance(data, bool):
                    current_data_type_ok = True
                    break
                elif s_type == "object" and isinstance(data, dict):
                    current_data_type_ok = True
                    break
                elif s_type == "array" and isinstance(data, list):
                    current_data_type_ok = True
                    break
                elif s_type == "null" and data is None:
                    current_data_type_ok = True
                    break
            data_type_matches_schema_type = current_data_type_ok

        # Apply schema-based redaction only if data type is compatible (or schema doesn't specify type)
        if data_type_matches_schema_type:
            if schema.get(SENSITIVE_FIELD_MARKER_KEY, False) is True:
                logger.debug(f"Redacting item (schema type compatible) due to '{SENSITIVE_FIELD_MARKER_KEY}: true'.")
                return REDACTION_PLACEHOLDER_VALUE
            if schema.get("format", "").lower() in SENSITIVE_FORMAT_VALUES:
                logger.debug(f"Redacting item (schema type compatible) due to sensitive format '{schema.get('format')}'.")
                return REDACTION_PLACEHOLDER_VALUE

    if isinstance(data, dict):
        sanitized_dict: Dict[str, Any] = {}
        properties_schema = schema.get("properties", {}) if schema else {}

        for key, value in data.items():
            field_schema = properties_schema.get(key)

            # Priority 1: Schema-based redaction for the field (will respect type if field_schema has type)
            # The recursive call handles the direct redaction of 'value' based on 'field_schema'

            # Priority 2: Key name-based redaction (if enabled)
            # This applies if schema-based redaction on the value itself didn't already redact it
            # based on field_schema's direct sensitivity.
            should_redact_by_key_name = False
            if redact_matching_key_names and isinstance(key, str) and COMMON_SENSITIVE_KEY_NAMES_REGEX.search(key): # Changed to search
                # Check if schema for this specific field already marked it sensitive
                is_field_schema_sensitive = False
                if field_schema:
                    field_schema_type = field_schema.get("type")
                    field_data_type_matches = True # Assume true if field_schema has no type
                    if field_schema_type:
                        # Simplified check for this specific path
                        if isinstance(field_schema_type, str):
                            if field_schema_type == "string" and not isinstance(value, str):
                                field_data_type_matches = False
                            # Add other type checks as above if needed for more precision here
                        # else if list of types, more complex

                    if field_data_type_matches and (field_schema.get(SENSITIVE_FIELD_MARKER_KEY, False) is True or \
                       field_schema.get("format", "").lower() in SENSITIVE_FORMAT_VALUES):
                        is_field_schema_sensitive = True

                if not is_field_schema_sensitive:
                    should_redact_by_key_name = True

            if should_redact_by_key_name:
                sanitized_dict[key] = REDACTION_PLACEHOLDER_VALUE
                logger.debug(f"Redacted dict field '{key}' due to matching common sensitive key name pattern.")
                continue

            # If not redacted by key name, recurse for the value
            sanitized_dict[key] = sanitize_data_with_schema_based_rules(
                value, field_schema, redact_matching_key_names=redact_matching_key_names
            )
        return sanitized_dict

    elif isinstance(data, list):
        items_schema = schema.get("items") if schema and isinstance(schema.get("items"), dict) else None
        return [sanitize_data_with_schema_based_rules(
            item, items_schema, redact_matching_key_names=redact_matching_key_names
        ) for item in data]

    return data

class SchemaAwareRedactor(Redactor):
    plugin_id: str = "schema_aware_redactor_v1"
    description: str = "Redacts data based on JSON schema hints ('x-sensitive', 'format') and common sensitive key name patterns."

    _redact_key_names: bool = True

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._redact_key_names = bool(cfg.get("redact_matching_key_names", True))
        logger.debug(f"{self.plugin_id}: Initialized. Redact matching key names: {self._redact_key_names}")

    def sanitize(self, data: Any, schema_hints: Optional[Dict[str, Any]] = None) -> Any:
        logger.debug(f"{self.plugin_id}: Sanitizing data. Schema hints provided: {schema_hints is not None}")
        return sanitize_data_with_schema_based_rules(
            data,
            schema=schema_hints,
            redact_matching_key_names=self._redact_key_names
        )

    async def teardown(self) -> None:
        logger.debug(f"{self.plugin_id}: Teardown complete.")
