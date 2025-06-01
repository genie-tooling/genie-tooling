### tests/unit/output_transformers/impl/test_json_transformer.py
import json
import logging
from typing import Any, Dict

import pytest

from genie_tooling.output_transformers.abc import OutputTransformationException
from genie_tooling.output_transformers.impl.json_transformer import (
    JSONOutputTransformer,
)

# Logger name for the module under test
TRANSFORMER_LOGGER_NAME = "genie_tooling.output_transformers.impl.json_transformer"

@pytest.fixture
def json_transformer_dict_output() -> JSONOutputTransformer:
    """Fixture for JSONOutputTransformer configured to output dict/list."""
    return JSONOutputTransformer(output_format="dict")


@pytest.fixture
def json_transformer_string_output() -> JSONOutputTransformer:
    """Fixture for JSONOutputTransformer configured to output JSON string."""
    return JSONOutputTransformer(output_format="string")


@pytest.fixture
def sample_schema() -> Dict[str, Any]:
    """A sample schema, though not heavily used by this transformer."""
    return {"type": "object", "properties": {"message": {"type": "string"}}}


# Test Initialization
def test_json_transformer_init_default_is_dict():
    transformer = JSONOutputTransformer()
    assert transformer.output_format == "dict"


def test_json_transformer_init_string():
    transformer = JSONOutputTransformer(output_format="string")
    assert transformer.output_format == "string"


def test_json_transformer_init_invalid_format():
    with pytest.raises(ValueError, match="output_format must be 'dict' or 'string'"):
        JSONOutputTransformer(output_format="xml") # type: ignore


# Test transform method with output_format="dict"
def test_transform_dict_output_already_dict(
    json_transformer_dict_output: JSONOutputTransformer, sample_schema: Dict[str, Any]
):
    input_data = {"key": "value", "number": 123}
    transformed = json_transformer_dict_output.transform(input_data, sample_schema)
    assert transformed == input_data
    assert isinstance(transformed, dict)


def test_transform_dict_output_already_list(
    json_transformer_dict_output: JSONOutputTransformer, sample_schema: Dict[str, Any]
):
    input_data = [1, "two", {"three": 3}]
    transformed = json_transformer_dict_output.transform(input_data, sample_schema)
    assert transformed == input_data
    assert isinstance(transformed, list)


def test_transform_dict_output_from_json_string(
    json_transformer_dict_output: JSONOutputTransformer, sample_schema: Dict[str, Any]
):
    input_json_string = '{"message": "hello", "count": 5}'
    expected_dict = {"message": "hello", "count": 5}
    transformed = json_transformer_dict_output.transform(
        input_json_string, sample_schema
    )
    assert transformed == expected_dict
    assert isinstance(transformed, dict)


def test_transform_dict_output_from_json_array_string(
    json_transformer_dict_output: JSONOutputTransformer, sample_schema: Dict[str, Any]
):
    input_json_string = '[true, null, "item"]'
    expected_list = [True, None, "item"]
    transformed = json_transformer_dict_output.transform(
        input_json_string, sample_schema
    )
    assert transformed == expected_list
    assert isinstance(transformed, list)


def test_transform_dict_output_non_serializable_input(
    json_transformer_dict_output: JSONOutputTransformer, sample_schema: Dict[str, Any]
):
    class NonSerializable:
        pass

    input_data = {"data": NonSerializable()}
    with pytest.raises(OutputTransformationException) as excinfo:
        json_transformer_dict_output.transform(input_data, sample_schema)
    # Check for the specific message when input is already dict/list but contains non-serializable
    assert "Input dict is not JSON serializable: Object of type NonSerializable is not JSON serializable" in str(excinfo.value)
    assert excinfo.value.original_output == input_data


def test_transform_dict_output_json_string_not_dict_or_list(
    json_transformer_dict_output: JSONOutputTransformer, sample_schema: Dict[str, Any]
):
    input_json_string_true = "true" # This is a valid JSON value, but not a dict/list
    with pytest.raises(OutputTransformationException) as excinfo_true:
        json_transformer_dict_output.transform(input_json_string_true, sample_schema)

    assert "Input string parsed to JSON type 'bool', but expected dict/list" in str(
        excinfo_true.value
    )

    input_json_string_number = "123.45"
    with pytest.raises(OutputTransformationException) as excinfo_num:
        json_transformer_dict_output.transform(input_json_string_number, sample_schema)

    assert "Input string parsed to JSON type 'float', but expected dict/list" in str(
        excinfo_num.value
    )

    input_json_string_invalid = "not a json string"
    with pytest.raises(OutputTransformationException) as excinfo_invalid:
        json_transformer_dict_output.transform(input_json_string_invalid, sample_schema)
    assert "Input string is not valid JSON" in str(excinfo_invalid.value)


# Test transform method with output_format="string"
def test_transform_string_output_from_dict(
    json_transformer_string_output: JSONOutputTransformer, sample_schema: Dict[str, Any]
):
    input_data = {"message": "world", "value": 7}
    expected_json_string = '{"message": "world", "value": 7}'
    transformed = json_transformer_string_output.transform(input_data, sample_schema)
    assert json.loads(transformed) == json.loads(expected_json_string) # Compare parsed
    assert isinstance(transformed, str)


def test_transform_string_output_from_list(
    json_transformer_string_output: JSONOutputTransformer, sample_schema: Dict[str, Any]
):
    input_data = ["a", "b", 1]
    expected_json_string = '["a", "b", 1]'
    transformed = json_transformer_string_output.transform(input_data, sample_schema)
    assert json.loads(transformed) == json.loads(expected_json_string)
    assert isinstance(transformed, str)


def test_transform_string_output_from_json_string(
    json_transformer_string_output: JSONOutputTransformer, sample_schema: Dict[str, Any]
):
    # If input is already a string, it should be dumped again to ensure it's valid JSON string output
    input_json_string = '{"already": "a string"}'
    # json.dumps will escape the inner quotes
    expected_output_string = json.dumps(input_json_string)

    transformed = json_transformer_string_output.transform(
        input_json_string, sample_schema
    )
    # The transformer should treat the input string as a Python string to be JSON-encoded,
    # not as already-encoded JSON.
    # So, '{"already": "a string"}' becomes '"{\\"already\\": \\"a string\\"}"'
    assert transformed == expected_output_string
    assert isinstance(transformed, str)

    # Test with a simple string that is not JSON
    input_simple_string = "just a simple string"
    expected_simple_string_output = '"just a simple string"' # JSON representation of a string
    transformed_simple = json_transformer_string_output.transform(input_simple_string, sample_schema)
    assert transformed_simple == expected_simple_string_output


def test_transform_string_output_non_serializable_input(
    json_transformer_string_output: JSONOutputTransformer,
    sample_schema: Dict[str, Any],
):
    class NonSerializableForString:
        pass

    input_data = {"complex": NonSerializableForString()}
    with pytest.raises(OutputTransformationException) as excinfo:
        json_transformer_string_output.transform(input_data, sample_schema)
    assert "Output is not JSON serializable: Object of type NonSerializableForString is not JSON serializable" in str(excinfo.value)


# Test logging (basic check for unexpected errors during logging)
def test_transform_logging_robustness(
    json_transformer_dict_output: JSONOutputTransformer,
    json_transformer_string_output: JSONOutputTransformer,
    sample_schema: Dict[str, Any],
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.DEBUG, logger=TRANSFORMER_LOGGER_NAME) # Target specific logger
    input_data = {"log_test": "data"}

    # Test dict output logging
    json_transformer_dict_output.transform(input_data, sample_schema)
    assert "JSONOutputTransformer: Input was already dict/list and is serializable." in caplog.text
    caplog.clear()

    # Test string output logging
    json_transformer_string_output.transform(input_data, sample_schema)
    assert "JSONOutputTransformer: Output transformed to JSON string." in caplog.text
    caplog.clear()

    # Test error logging for non-serializable
    class NonSerializableLog: pass
    with pytest.raises(OutputTransformationException):
        json_transformer_dict_output.transform({"bad": NonSerializableLog()}, sample_schema)
    # Check for the specific error message logged by the transformer
    assert "JSONOutputTransformer: Input dict is not JSON serializable: Object of type NonSerializableLog is not JSON serializable" in caplog.text
    caplog.clear()


# Test setup and teardown (default pass behavior)
@pytest.mark.asyncio
async def test_json_transformer_setup_teardown(
    json_transformer_dict_output: JSONOutputTransformer,
):
    # Default setup and teardown are no-ops
    await json_transformer_dict_output.setup()
    await json_transformer_dict_output.teardown()
    # No assertions needed, just checking they don't raise errors
    assert True
