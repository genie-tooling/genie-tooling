### tests/unit/logging_monitoring/redaction/test_schema_aware.py
import logging
from typing import Any, Dict

import pytest
from genie_tooling.redactors.impl.schema_aware import (
    REDACTION_PLACEHOLDER_VALUE,
    SENSITIVE_FIELD_MARKER_KEY,
    sanitize_data_with_schema_based_rules,
)

# Test Data and Schemas

@pytest.fixture
def simple_data() -> Dict[str, Any]:
    return {
        "username": "john_doe",
        "password": "secure_password123",
        "email": "john.doe@example.com",
        "token": "abcdef123456",
        "api_key": "xyz789-api-key"
    }

@pytest.fixture
def simple_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "username": {"type": "string"},
            "password": {"type": "string", SENSITIVE_FIELD_MARKER_KEY: True},
            "email": {"type": "string", "format": "email"},
            "token": {"type": "string", "format": "token"}, # "token" is in SENSITIVE_FORMAT_VALUES
            "api_key": {"type": "string"} # No specific schema hint here, relies on key name
        }
    }

@pytest.fixture
def nested_data() -> Dict[str, Any]:
    return {
        "user_info": {
            "id": 123,
            "details": {
                "full_name": "Jane Doe",
                "credentials": { # This key will match the sensitive key name regex
                    "type": "oauth",
                    "secret_token": "super_secret_oauth_token" # This field also has x-sensitive
                }
            }
        },
        "configuration": {
            "settings": ["a", "b"],
            "private_key_pem": "-----BEGIN RSA PRIVATE KEY-----\n MII...\n-----END RSA PRIVATE KEY-----"
        },
        "history": [
            {"action": "login", "timestamp": "2023-01-01T10:00:00Z"},
            {"action": "view_profile", "auth_details": {"session_id": "sess1", "access_token": "user_access_token"}}
        ]
    }

@pytest.fixture
def nested_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "user_info": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "details": {
                        "type": "object",
                        "properties": {
                            "full_name": {"type": "string"},
                            "credentials": { # Schema for the 'credentials' object
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string"},
                                    "secret_token": {"type": "string", SENSITIVE_FIELD_MARKER_KEY: True}
                                }
                            }
                        }
                    }
                }
            },
            "configuration": {
                "type": "object",
                "properties": {
                    "settings": {"type": "array", "items": {"type": "string"}},
                    "private_key_pem": {"type": "string", "format": "private_key"} # sensitive format
                }
            },
            "history": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "timestamp": {"type": "string", "format": "date-time"},
                        "auth_details": { # Schema for the 'auth_details' object
                            "type": "object",
                            "properties": {
                                "session_id": {"type": "string"},
                                "access_token": {"type": "string", "format": "auth_token"}
                            },
                            SENSITIVE_FIELD_MARKER_KEY: True # Mark entire auth_details object as sensitive
                        }
                    }
                }
            }
        }
    }

# Test Functions

def test_redact_with_x_sensitive_marker(simple_data: Dict[str, Any], simple_schema: Dict[str, Any]):
    sanitized = sanitize_data_with_schema_based_rules(simple_data, simple_schema)
    assert sanitized["username"] == "john_doe"
    assert sanitized["password"] == REDACTION_PLACEHOLDER_VALUE
    assert sanitized["email"] == "john.doe@example.com"

def test_redact_with_sensitive_format(simple_data: Dict[str, Any], simple_schema: Dict[str, Any]):
    sanitized = sanitize_data_with_schema_based_rules(simple_data, simple_schema)
    assert sanitized["token"] == REDACTION_PLACEHOLDER_VALUE

def test_redact_with_sensitive_key_name(simple_data: Dict[str, Any], simple_schema: Dict[str, Any]):
    sanitized = sanitize_data_with_schema_based_rules(simple_data, simple_schema, redact_matching_key_names=True)
    assert sanitized["api_key"] == REDACTION_PLACEHOLDER_VALUE

    data_case = {"API_KEY": "testkey", "MyPassword": "pw", "auth_method": "basic"}
    sanitized_case = sanitize_data_with_schema_based_rules(data_case, None, redact_matching_key_names=True)
    assert sanitized_case["API_KEY"] == REDACTION_PLACEHOLDER_VALUE
    assert sanitized_case["MyPassword"] == REDACTION_PLACEHOLDER_VALUE
    assert sanitized_case["auth_method"] == REDACTION_PLACEHOLDER_VALUE # "auth" is in regex

def test_no_redact_if_key_name_redaction_disabled(simple_data: Dict[str, Any]):
    schema_no_hint_for_api_key = {
        "type": "object",
        "properties": {
            "api_key": {"type": "string"}
        }
    }
    data = {"api_key": "xyz789-api-key"}
    sanitized = sanitize_data_with_schema_based_rules(data, schema_no_hint_for_api_key, redact_matching_key_names=False)
    assert sanitized["api_key"] == "xyz789-api-key"

def test_no_redaction_if_no_hints_or_sensitive_keys(simple_data: Dict[str, Any]):
    data = {"field1": "value1", "another_field": 123}
    schema = {
        "type": "object",
        "properties": {
            "field1": {"type": "string"},
            "another_field": {"type": "integer"}
        }
    }
    sanitized = sanitize_data_with_schema_based_rules(data, schema, redact_matching_key_names=True)
    assert sanitized == data

    sanitized_no_schema = sanitize_data_with_schema_based_rules(data.copy(), None, redact_matching_key_names=True)
    assert sanitized_no_schema == data

def test_recursive_redaction_dict_in_dict(nested_data: Dict[str, Any], nested_schema: Dict[str, Any]):
    sanitized = sanitize_data_with_schema_based_rules(nested_data, nested_schema)
    # "credentials" key itself will cause the object to be redacted by key name
    assert sanitized["user_info"]["details"]["credentials"] == REDACTION_PLACEHOLDER_VALUE
    assert sanitized["user_info"]["details"]["full_name"] == "Jane Doe"
    assert sanitized["configuration"]["private_key_pem"] == REDACTION_PLACEHOLDER_VALUE

def test_recursive_redaction_list_of_dicts(nested_data: Dict[str, Any], nested_schema: Dict[str, Any]):
    sanitized = sanitize_data_with_schema_based_rules(nested_data, nested_schema)
    assert sanitized["history"][0]["action"] == "login"
    # auth_details object itself is marked sensitive by its schema
    assert sanitized["history"][1]["auth_details"] == REDACTION_PLACEHOLDER_VALUE

def test_redact_entire_object_marked_sensitive(nested_data: Dict[str, Any], nested_schema: Dict[str, Any]):
    sanitized = sanitize_data_with_schema_based_rules(nested_data, nested_schema)
    assert sanitized["history"][1]["auth_details"] == REDACTION_PLACEHOLDER_VALUE
    with pytest.raises(TypeError):
        _ = sanitized["history"][1]["auth_details"]["access_token"] # type: ignore

def test_redact_item_directly_if_schema_applies_to_it():
    data_string = "a_password_string"
    schema_string = {"type": "string", "format": "password"}
    sanitized_string = sanitize_data_with_schema_based_rules(data_string, schema_string)
    assert sanitized_string == REDACTION_PLACEHOLDER_VALUE

    data_list = ["normal", "api_key_value", "public", 123]
    list_item_schema = {"type": "string", "format": "api_key"}
    list_schema = {"type": "array", "items": list_item_schema}
    sanitized_list = sanitize_data_with_schema_based_rules(data_list, list_schema)

    assert sanitized_list[0] == REDACTION_PLACEHOLDER_VALUE
    assert sanitized_list[1] == REDACTION_PLACEHOLDER_VALUE
    assert sanitized_list[2] == REDACTION_PLACEHOLDER_VALUE
    assert sanitized_list[3] == 123

def test_no_schema_provided_key_name_redaction(simple_data: Dict[str, Any]):
    sanitized = sanitize_data_with_schema_based_rules(simple_data.copy(), None, redact_matching_key_names=True)
    assert sanitized["username"] == "john_doe"
    assert sanitized["password"] == REDACTION_PLACEHOLDER_VALUE
    assert sanitized["email"] == "john.doe@example.com"
    assert sanitized["token"] == REDACTION_PLACEHOLDER_VALUE
    assert sanitized["api_key"] == REDACTION_PLACEHOLDER_VALUE

def test_no_schema_provided_no_key_name_redaction(simple_data: Dict[str, Any]):
    sanitized = sanitize_data_with_schema_based_rules(simple_data.copy(), None, redact_matching_key_names=False)
    assert sanitized == simple_data

def test_logging_of_redaction_actions(simple_data: Dict[str, Any], simple_schema: Dict[str, Any], caplog: pytest.LogCaptureFixture):
    target_logger_name = "genie_tooling.redactors.impl.schema_aware"
    caplog.set_level(logging.DEBUG, logger=target_logger_name)

    sanitize_data_with_schema_based_rules(simple_data, simple_schema, redact_matching_key_names=True)

    # Check for password redaction - this happens at the item level due to "x-sensitive" on its schema
    assert "Redacting item (schema type compatible) due to 'x-sensitive: true'." in caplog.text
    # Check for token redaction - this happens at item level due to "format: token" on its schema
    assert "Redacting item (schema type compatible) due to sensitive format 'token'." in caplog.text
    # Check for api_key redaction by key_name (since its field_schema has no direct sensitive markers)
    assert "Redacted dict field 'api_key' due to matching common sensitive key name pattern." in caplog.text

def test_non_string_keys_not_redacted_by_name():
    data = {123: "sensitive_value_for_int_key", "name": "Test"}
    sanitized = sanitize_data_with_schema_based_rules(data, None, redact_matching_key_names=True)
    assert sanitized[123] == "sensitive_value_for_int_key"
    assert sanitized["name"] == "Test"

def test_partial_schema_coverage(simple_data: Dict[str, Any]):
    partial_schema = {
        "type": "object",
        "properties": {
            "username": {"type": "string"}
        }
    }
    sanitized = sanitize_data_with_schema_based_rules(simple_data.copy(), partial_schema, redact_matching_key_names=True)
    assert sanitized["username"] == "john_doe"
    assert sanitized["password"] == REDACTION_PLACEHOLDER_VALUE
    assert sanitized["api_key"] == REDACTION_PLACEHOLDER_VALUE
    assert sanitized["token"] == REDACTION_PLACEHOLDER_VALUE

def test_sensitive_format_values_are_lowercase_in_check():
    data = {"my_secret": "secret_data"}
    schema = {
        "type": "object",
        "properties": {
            "my_secret": {"type": "string", "format": "API_KEY"}
        }
    }
    sanitized = sanitize_data_with_schema_based_rules(data, schema)
    assert sanitized["my_secret"] == REDACTION_PLACEHOLDER_VALUE
