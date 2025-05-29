### tests/unit/invocation/test_validation.py
"""Unit tests for input validation components."""
from unittest.mock import patch

import pytest

# Updated import paths for InputValidationException and JSONSchemaInputValidator
from genie_tooling.input_validators import (
    InputValidationException,
    JSONSchemaInputValidator,
)

# --- Fixtures for JSONSchemaInputValidator ---

@pytest.fixture
def schema_validator() -> JSONSchemaInputValidator:
    """Provides a JSONSchemaInputValidator instance."""
    return JSONSchemaInputValidator()

# --- Test Cases for JSONSchemaInputValidator ---

def test_jsonschema_validator_success(schema_validator: JSONSchemaInputValidator):
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0}
        },
        "required": ["name"]
    }
    params = {"name": "Alice", "age": 30}

    validated_params = schema_validator.validate(params, schema)
    assert validated_params == params

def test_jsonschema_validator_failure_missing_required(schema_validator: JSONSchemaInputValidator):
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"]
    }
    params = {"age": 30}

    with pytest.raises(InputValidationException) as excinfo:
        schema_validator.validate(params, schema)

    # Check the primary message passed to the exception constructor
    assert str(excinfo.value) == "Input validation failed."
    assert excinfo.value.params == params
    assert len(excinfo.value.errors) == 1
    specific_error_details = excinfo.value.errors[0] # This is now a dict
    assert "'name' is a required property" in specific_error_details["message"]
    assert specific_error_details["validator"] == "required"


def test_jsonschema_validator_failure_wrong_type(schema_validator: JSONSchemaInputValidator):
    schema = {
        "type": "object",
        "properties": {"age": {"type": "integer"}}
    }
    params = {"age": "thirty"}

    with pytest.raises(InputValidationException) as excinfo:
        schema_validator.validate(params, schema)

    assert str(excinfo.value) == "Input validation failed."
    assert len(excinfo.value.errors) == 1
    specific_error_details = excinfo.value.errors[0]
    assert "'thirty' is not of type 'integer'" in specific_error_details["message"]
    assert specific_error_details["path"] == ["age"]
    assert specific_error_details["instance_failed"] == "thirty"


def test_jsonschema_validator_failure_minimum_constraint(schema_validator: JSONSchemaInputValidator):
    schema = {
        "type": "object",
        "properties": {"age": {"type": "integer", "minimum": 0}}
    }
    params = {"age": -5}

    with pytest.raises(InputValidationException) as excinfo:
        schema_validator.validate(params, schema)

    assert str(excinfo.value) == "Input validation failed."
    assert len(excinfo.value.errors) == 1
    specific_error_details = excinfo.value.errors[0]
    assert "-5 is less than the minimum of 0" in specific_error_details["message"]
    assert specific_error_details["validator"] == "minimum"
    assert specific_error_details["instance_failed"] == -5


def test_jsonschema_validator_with_defaults_in_schema(schema_validator: JSONSchemaInputValidator):
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "city": {"type": "string", "default": "Unknown"}
        },
        "required": ["name"]
    }
    params = {"name": "Bob"}

    validated_params = schema_validator.validate(params, schema)
    assert validated_params == params


@patch("genie_tooling.input_validators.impl.jsonschema_validator.jsonschema", None)
@patch("genie_tooling.input_validators.impl.jsonschema_validator.validators", None)
@patch("genie_tooling.input_validators.impl.jsonschema_validator.JSONSchemaValidationError", None)
@patch("genie_tooling.input_validators.impl.jsonschema_validator.JSONSchemaSchemaError", None)
def test_jsonschema_validator_jsonschema_not_installed():
    validator_no_lib = JSONSchemaInputValidator()
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    params = {"name": 123}

    validated_params = validator_no_lib.validate(params, schema)
    assert validated_params == params


def test_jsonschema_validator_invalid_schema(schema_validator: JSONSchemaInputValidator):
    invalid_schema = {
        "type": "object",
        "properties": {"name": {"type": "nonexistent_type"}}
    }
    params = {"name": "Test"}

    with pytest.raises(InputValidationException) as excinfo:
        schema_validator.validate(params, invalid_schema)

    exception_message = str(excinfo.value)
    assert "Invalid schema configuration" in exception_message
    assert "Unknown type 'nonexistent_type'" in exception_message
    assert len(excinfo.value.errors) == 1
    specific_error_details = excinfo.value.errors[0] # This is now a dict
    assert "Unknown type 'nonexistent_type'" in specific_error_details["message"]


def test_jsonschema_validator_handles_multiple_errors(schema_validator: JSONSchemaInputValidator):
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 3},
            "age": {"type": "integer", "minimum": 18},
            "email": {"type": "string", "format": "email"}
        },
        "required": ["name", "age", "email"]
    }
    params = {
        "name": "Al",
        "age": 16,
    }

    with pytest.raises(InputValidationException) as excinfo:
        schema_validator.validate(params, schema)

    assert str(excinfo.value) == "Input validation failed."
    assert len(excinfo.value.errors) == 3

    error_messages_from_details = [err["message"] for err in excinfo.value.errors]
    assert any("'Al' is too short" in msg for msg in error_messages_from_details)
    assert any("16 is less than the minimum of 18" in msg for msg in error_messages_from_details)
    assert any("'email' is a required property" in msg for msg in error_messages_from_details)
