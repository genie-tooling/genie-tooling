### tests/unit/prompts/llm_output_parsers/impl/test_pydantic_output_parser.py
import logging
from typing import List, Optional
from unittest.mock import patch

import pytest
from genie_tooling.prompts.llm_output_parsers.impl.pydantic_output_parser import (
    PYDANTIC_AVAILABLE,
    PydanticOutputParserPlugin,
)

# Mock Pydantic BaseModel if not available, or use real one
if PYDANTIC_AVAILABLE:
    from pydantic import BaseModel, Field, ValidationError
else:
    # Minimal mock for BaseModel if pydantic is not installed
    class BaseModel: # type: ignore
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        @classmethod
        def model_validate_json(cls, json_str: str): # Mock for Pydantic v2
            import json
            data = json.loads(json_str)
            return cls(**data)
        def model_dump(self): # Mock for Pydantic v2
            return self.__dict__

    class Field: # type: ignore
        def __init__(self, default=None, **kwargs): pass

    class ValidationError(ValueError): # type: ignore
        def __init__(self, errors):
            super().__init__(str(errors))
            self._errors = errors
        def errors(self): return self._errors


PARSER_LOGGER_NAME = "genie_tooling.prompts.llm_output_parsers.impl.pydantic_output_parser"

class MyTestModel(BaseModel):
    name: str
    age: int
    tags: Optional[List[str]] = Field(default_factory=list)

class NestedModelForPydantic(BaseModel):
    item_id: int
    description: Optional[str] = None

class ModelWithNested(BaseModel):
    main_id: str
    nested_obj: Optional[NestedModelForPydantic] = None
    nested_list: List[NestedModelForPydantic] = Field(default_factory=list)


@pytest.fixture()
async def pydantic_parser() -> PydanticOutputParserPlugin:
    parser = PydanticOutputParserPlugin()
    await parser.setup()
    return parser

@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic library not installed")
@pytest.mark.asyncio()
async def test_parse_valid_json_to_pydantic_model(pydantic_parser: PydanticOutputParserPlugin):
    parser = await pydantic_parser
    text = '```json\n{"name": "Alice", "age": 30, "tags": ["test", "user"]}\n```'
    parsed_model = parser.parse(text, schema=MyTestModel)
    assert isinstance(parsed_model, MyTestModel)
    assert parsed_model.name == "Alice"
    assert parsed_model.age == 30
    assert parsed_model.tags == ["test", "user"]

@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic library not installed")
@pytest.mark.asyncio()
async def test_parse_pydantic_validation_error(pydantic_parser: PydanticOutputParserPlugin, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING, logger=PARSER_LOGGER_NAME)
    parser = await pydantic_parser
    text = '{"name": "Bob", "age": "thirty"}' # Age is string, not int
    with pytest.raises(ValueError, match="Pydantic validation failed"):
        parser.parse(text, schema=MyTestModel)
    assert "Pydantic validation failed for model 'MyTestModel'" in caplog.text

@pytest.mark.asyncio()
async def test_parse_no_pydantic_schema_provided(pydantic_parser: PydanticOutputParserPlugin):
    parser = await pydantic_parser
    if not PYDANTIC_AVAILABLE: pytest.skip("Pydantic not available")
    text = '{"data": "any"}'
    with pytest.raises(ValueError, match="A Pydantic model class must be provided as the 'schema' argument."):
        parser.parse(text, schema=None) # type: ignore
    with pytest.raises(ValueError, match="A Pydantic model class must be provided as the 'schema' argument."):
        parser.parse(text, schema=dict) # type: ignore

@pytest.mark.asyncio()
async def test_parse_no_json_block_found(pydantic_parser: PydanticOutputParserPlugin, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING, logger=PARSER_LOGGER_NAME)
    parser = await pydantic_parser
    if not PYDANTIC_AVAILABLE: pytest.skip("Pydantic not available")
    text = "This is plain text."
    with pytest.raises(ValueError, match="No parsable JSON block found in the input text for Pydantic parsing."):
        parser.parse(text, schema=MyTestModel)
    assert "No valid JSON block found in text_output" in caplog.text

@pytest.mark.asyncio()
async def test_pydantic_not_available_runtime_error(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=PARSER_LOGGER_NAME)
    with patch("genie_tooling.prompts.llm_output_parsers.impl.pydantic_output_parser.PYDANTIC_AVAILABLE", False):
        parser_no_pydantic = PydanticOutputParserPlugin()
        await parser_no_pydantic.setup() # Logs error during setup
        assert "Pydantic library not installed. This plugin will not function." in caplog.text

        with pytest.raises(RuntimeError, match="Pydantic library not available at runtime."):
            parser_no_pydantic.parse("text", schema=MyTestModel) # type: ignore

@pytest.mark.asyncio()
async def test_teardown(pydantic_parser: PydanticOutputParserPlugin, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG, logger=PARSER_LOGGER_NAME)
    parser = await pydantic_parser
    await parser.teardown()
    assert f"{parser.plugin_id}: Teardown complete." in caplog.text

# --- New/Enhanced Tests for PydanticOutputParserPlugin ---

@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic library not installed")
@pytest.mark.asyncio()
@pytest.mark.parametrize(
    "text_input, expected_json_str",
    [
        ('Some text before {"key": "value"} and after.', '{"key": "value"}'),
        ('{"only_json": true}', '{"only_json": true}'),
        ('```json\n{"code_block_json": "data"}\n```', '{"code_block_json": "data"}'),
        ('```\n{"generic_code_block": true}\n```', '{"generic_code_block": true}'),
        ('Text with array: [1, 2, {"key": "val"}] trailing.', '[1, 2, {"key": "val"}]'),
        ("No JSON here.", None),
        ('Malformed {json: "block",', None),
        ('Text with { "inner": { "nested": "value" } } block.', '{ "inner": { "nested": "value" } }'),
        ('{"a":1} some text {"b":2}', '{"a":1}'),
        ('Thought: ... \n```json\n{"thought": "User wants to calculate.", "tool_id": "tool_calc", "params": {"num1": 5, "num2": 3}}\n```',
         '{"thought": "User wants to calculate.", "tool_id": "tool_calc", "params": {"num1": 5, "num2": 3}}'),
        ('```\n{"generic_code_block": true}\n```', '{"generic_code_block": true}'),
        ('Text with multiple JSON blocks: {"first": 1} and then {"second": 2}.', '{"first": 1}'),
        ('Text with nested JSON in text: "outer text {\\"inner_json\\": true}"', None),
        ('Text with array: [1, 2, {"key": "val"}] trailing.', '[1, 2, {"key": "val"}]'),
        ('```json\n[\n  {"item": 1},\n  {"item": 2}\n]\n```', '[\n  {"item": 1},\n  {"item": 2}\n]'),
    ],
)
async def test_extract_json_block_various_inputs(
    pydantic_parser: PydanticOutputParserPlugin, text_input: str, expected_json_str: Optional[str]
):
    """Test the internal _extract_json_block method with various inputs."""
    parser = await pydantic_parser
    assert parser._extract_json_block(text_input) == expected_json_str

@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic library not installed")
@pytest.mark.asyncio()
async def test_parse_with_nested_pydantic_models(pydantic_parser: PydanticOutputParserPlugin):
    parser = await pydantic_parser
    text_input = """
    Some introductory text.
    ```json
    {
        "main_id": "main123",
        "nested_obj": {
            "item_id": 1,
            "description": "Nested Item One"
        },
        "nested_list": [
            {"item_id": 2, "description": "Item A"},
            {"item_id": 3}
        ]
    }
    ```
    Some trailing text.
    """
    parsed_model = parser.parse(text_input, schema=ModelWithNested)
    assert isinstance(parsed_model, ModelWithNested)
    assert parsed_model.main_id == "main123"
    assert parsed_model.nested_obj is not None
    assert parsed_model.nested_obj.item_id == 1
    assert parsed_model.nested_obj.description == "Nested Item One"
    assert len(parsed_model.nested_list) == 2
    assert parsed_model.nested_list[0].item_id == 2
    assert parsed_model.nested_list[0].description == "Item A"
    assert parsed_model.nested_list[1].item_id == 3
    assert parsed_model.nested_list[1].description is None

@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic library not installed")
@pytest.mark.asyncio()
async def test_parse_empty_input_string_raises_value_error(pydantic_parser: PydanticOutputParserPlugin):
    parser = await pydantic_parser
    with pytest.raises(ValueError, match="Input text_output is empty or whitespace."):
        parser.parse("", schema=MyTestModel)
    with pytest.raises(ValueError, match="Input text_output is empty or whitespace."):
        parser.parse("   \n \t ", schema=MyTestModel)

@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic library not installed")
@pytest.mark.asyncio()
async def test_parse_json_loads_error_after_extraction(pydantic_parser: PydanticOutputParserPlugin, caplog: pytest.LogCaptureFixture):
    """Test scenario where _extract_json_block finds something that then fails Pydantic validation (or json.loads if Pydantic v1)."""
    parser = await pydantic_parser
    caplog.set_level(logging.WARNING, logger=PARSER_LOGGER_NAME)

    malformed_json_string = '{"key": "value", "malformed": }' # Invalid JSON

    with patch.object(parser, "_extract_json_block", return_value=malformed_json_string) as mock_extract:
        # Pydantic v2's model_validate_json will raise ValidationError if the string is not valid JSON.
        # The message within ValidationError will indicate a JSON parsing problem.
        with pytest.raises(ValueError, match="Pydantic validation failed: "):
            parser.parse("Some text containing a malformed JSON block.", schema=MyTestModel)

    mock_extract.assert_called_once()
    # Check the log message from the PydanticOutputParserPlugin's parse method's ValidationError block
    expected_log_prefix = f"{parser.plugin_id}: Pydantic validation failed for model 'MyTestModel'."
    # Check for common JSON parsing error messages within the Pydantic error details
    json_error_indicators = ["json_invalid", "expecting value", "decodeerror"]

    log_found = False
    for record in caplog.records:
        if record.name == PARSER_LOGGER_NAME and record.levelno == logging.WARNING:
            if expected_log_prefix in record.message:
                if any(indicator in record.message.lower() for indicator in json_error_indicators):
                    log_found = True
                    break
    assert log_found, f"Expected warning log for Pydantic validation failure due to invalid JSON not found or message incorrect. Logs: {caplog.text}"


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic library not installed")
@pytest.mark.asyncio()
async def test_parse_unexpected_error_during_pydantic_validation(pydantic_parser: PydanticOutputParserPlugin, caplog: pytest.LogCaptureFixture):
    """Test handling of unexpected errors from Pydantic's validation, not just ValidationError."""
    parser = await pydantic_parser
    caplog.set_level(logging.ERROR, logger=PARSER_LOGGER_NAME)
    text_input = '{"name": "Test", "age": 30}' # Valid JSON for MyTestModel

    # Mock Pydantic's model_validate_json to raise an unexpected error
    with patch.object(MyTestModel, "model_validate_json", side_effect=TypeError("Unexpected Pydantic internal error")):
        with pytest.raises(ValueError, match="Unexpected Pydantic parsing error: Unexpected Pydantic internal error"):
            parser.parse(text_input, schema=MyTestModel)

    assert "Unexpected error during Pydantic parsing: Unexpected Pydantic internal error" in caplog.text

@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic library not installed")
@pytest.mark.asyncio()
async def test_parse_malformed_json_in_code_block_pydantic(pydantic_parser: PydanticOutputParserPlugin, caplog: pytest.LogCaptureFixture):
    """Test malformed JSON within a code block for Pydantic parser."""
    parser = await pydantic_parser
    text_input = "```json\n{key: 'malformed, missing quotes'}\n```" # Malformed JSON

    with caplog.at_level(logging.DEBUG, logger=PARSER_LOGGER_NAME): # Capture DEBUG for _extract_json_block
        with pytest.raises(ValueError, match="No parsable JSON block found in the input text for Pydantic parsing."):
            parser.parse(text_input, schema=MyTestModel)

    # Check that _extract_json_block logged its attempt and failure
    expected_debug_log_fragment = f"{parser.plugin_id}: Found ```json``` block, but content is not valid JSON:"
    assert any(expected_debug_log_fragment in record.message for record in caplog.records if record.levelno == logging.DEBUG), \
        f"Expected DEBUG log for malformed JSON in code block not found. Logs: {caplog.text}"

    # Check the final warning from the parse method itself (which is raised as an error)
    assert any("No valid JSON block found in text_output" in record.message for record in caplog.records if record.levelno == logging.WARNING), \
        f"Expected WARNING log for no valid JSON block found not present. Logs: {caplog.text}"
