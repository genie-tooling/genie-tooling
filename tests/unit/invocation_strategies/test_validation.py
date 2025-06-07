"""Unit tests for input validation components."""
from unittest.mock import patch

import pytest
from genie_tooling.input_validators import (
    InputValidationException,
    JSONSchemaInputValidator,
)


@pytest.fixture
def schema_validator() -> JSONSchemaInputValidator:
    return JSONSchemaInputValidator()

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
    assert str(excinfo.value) == "Input validation failed."
    assert len(excinfo.value.errors) == 1
    assert "'name' is a required property" in excinfo.value.errors[0]["message"]

def test_jsonschema_validator_failure_wrong_type(schema_validator: JSONSchemaInputValidator):
    schema = {"type": "object", "properties": {"age": {"type": "integer"}}, "required": ["age"]}
    params = {"age": "thirty"}
    with pytest.raises(InputValidationException) as excinfo:
        schema_validator.validate(params, schema)
    assert str(excinfo.value) == "Input validation failed."
    assert len(excinfo.value.errors) == 1
    assert "'thirty' is not of type 'integer'" in excinfo.value.errors[0]["message"]

def test_jsonschema_validator_failure_minimum_constraint(schema_validator: JSONSchemaInputValidator):
    schema = {"type": "object", "properties": {"age": {"type": "integer", "minimum": 0}}, "required": ["age"]}
    params = {"age": -5}
    with pytest.raises(InputValidationException) as excinfo:
        schema_validator.validate(params, schema)
    assert str(excinfo.value) == "Input validation failed."
    assert len(excinfo.value.errors) == 1
    assert "-5 is less than the minimum of 0" in excinfo.value.errors[0]["message"]

def test_jsonschema_validator_with_defaults_in_schema(schema_validator: JSONSchemaInputValidator):
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "city": {"type": "string", "default": "Unknown"},
            "country": {"type": "string", "default": "USA"}
        },
        "required": ["name"]
    }
    params = {"name": "Bob", "country": "Canada"}
    expected_params_with_defaults = {"name": "Bob", "city": "Unknown", "country": "Canada"}

    validated_params = schema_validator.validate(params, schema)

    assert validated_params == expected_params_with_defaults
    assert params == {"name": "Bob", "country": "Canada"}

def test_jsonschema_validator_defaults_not_overwritten(schema_validator: JSONSchemaInputValidator):
    schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string", "default": "Unknown"},
        }
    }
    params = {"city": "Metropolis"}
    expected_params = {"city": "Metropolis"}
    validated_params = schema_validator.validate(params, schema)
    assert validated_params == expected_params

def test_jsonschema_validator_nested_defaults(schema_validator: JSONSchemaInputValidator):
    schema = {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string", "default": "guest"}
                },
                "required": ["name"],
                "default": {}
            }
        }
    }
    params1 = {"user": {"name": "Alice"}}
    expected1 = {"user": {"name": "Alice", "role": "guest"}}
    validated1 = schema_validator.validate(params1, schema)
    assert validated1 == expected1

    params2 = {}
    with pytest.raises(InputValidationException) as excinfo:
        schema_validator.validate(params2, schema)
    assert "'name' is a required property" in excinfo.value.errors[0]["message"]
    assert list(excinfo.value.errors[0]["path"]) == ["user"]

    schema_user_name_not_required = {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string", "default": "guest"}
                },
                "default": {}
            }
        }
    }
    params3 = {}
    expected3 = {"user": {"role": "guest"}}
    validated3 = schema_validator.validate(params3, schema_user_name_not_required)
    assert validated3 == expected3

@patch("genie_tooling.input_validators.impl.jsonschema_validator.JSONSCHEMA_AVAILABLE", False)
def test_jsonschema_validator_jsonschema_not_installed():
    with patch("genie_tooling.input_validators.impl.jsonschema_validator.validators", None), \
         patch("genie_tooling.input_validators.impl.jsonschema_validator.jsonschema", None):
        validator_no_lib = JSONSchemaInputValidator()

    assert validator_no_lib._jsonschema_available is False
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    params = {"name": 123}
    validated_params = validator_no_lib.validate(params, schema)
    assert validated_params == params

def test_jsonschema_validator_invalid_schema(schema_validator: JSONSchemaInputValidator):
    invalid_schema = {"type": "object", "properties": {"name": {"type": "nonexistent_type"}}}
    params = {"name": "Test"}
    with pytest.raises(InputValidationException) as excinfo:
        schema_validator.validate(params, invalid_schema)

    assert "Invalid schema configuration: " in str(excinfo.value)
    assert "'nonexistent_type' is not valid under any of the given schemas" in excinfo.value.errors[0]["message"]

# REMOVED @pytest.mark.xfail
def test_jsonschema_validator_handles_multiple_independent_errors(schema_validator: JSONSchemaInputValidator):
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 10},
            "age": {"type": "integer", "minimum": 18},
            "status": {"type": "string", "enum": ["active", "inactive"]}
        }
    }
    params = { "name": "Al", "age": 16, "status": "pending" }
    with pytest.raises(InputValidationException) as excinfo:
        schema_validator.validate(params, schema)
    assert str(excinfo.value) == "Input validation failed."
    # UPDATED ASSERTION: Expect 3 errors now
    assert len(excinfo.value.errors) == 3
    error_messages = [e['message'] for e in excinfo.value.errors]
    assert "'Al' is too short" in error_messages
    assert "16 is less than the minimum of 18" in error_messages
    assert "'pending' is not one of ['active', 'inactive']" in error_messages


# REMOVED @pytest.mark.xfail
def test_jsonschema_validator_handles_required_and_other_errors(schema_validator: JSONSchemaInputValidator):
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 10},
            "age": {"type": "integer", "minimum": 18},
            "email": {"type": "string"}
        },
        "required": ["name", "age", "email"]
    }
    params = { "name": "Al", "age": 16 } # Missing email, name too short, age too low
    with pytest.raises(InputValidationException) as excinfo:
        schema_validator.validate(params, schema)
    assert str(excinfo.value) == "Input validation failed."
    # UPDATED ASSERTION: Expect 3 errors now
    assert len(excinfo.value.errors) == 3
    error_messages = [e['message'] for e in excinfo.value.errors]
    assert "'email' is a required property" in error_messages
    assert "'Al' is too short" in error_messages
    assert "16 is less than the minimum of 18" in error_messages
