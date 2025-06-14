### tests/unit/prompts/llm_output_parsers/impl/test_json_output_parser.py
import logging

import pytest
from genie_tooling.prompts.llm_output_parsers.impl.json_output_parser import (
    JSONOutputParserPlugin,
)

PARSER_LOGGER_NAME = "genie_tooling.prompts.llm_output_parsers.impl.json_output_parser"
# Get the actual logger instance that the module uses
module_logger_instance = logging.getLogger(PARSER_LOGGER_NAME)


@pytest.fixture()
async def json_parser_strict() -> JSONOutputParserPlugin:
    parser = JSONOutputParserPlugin()
    await parser.setup(config={"strict_parsing": True})
    return parser

@pytest.fixture()
async def json_parser_non_strict() -> JSONOutputParserPlugin:
    parser = JSONOutputParserPlugin()
    await parser.setup(config={"strict_parsing": False}) # Default
    return parser

@pytest.mark.asyncio()
async def test_parse_strict_valid_json(json_parser_strict: JSONOutputParserPlugin):
    parser = await json_parser_strict
    text = '{"key": "value", "number": 123}'
    parsed = parser.parse(text)
    assert parsed == {"key": "value", "number": 123}

@pytest.mark.asyncio()
async def test_parse_strict_invalid_json(json_parser_strict: JSONOutputParserPlugin, caplog: pytest.LogCaptureFixture):
    # caplog.set_level(logging.WARNING) # Not needed if using caplog.at_level
    parser = await json_parser_strict
    text = '{"key": "value", "number": 123} trailing text'
    with caplog.at_level(logging.WARNING, logger=PARSER_LOGGER_NAME):
        with pytest.raises(ValueError, match="Strict JSON parsing failed: Extra data"):
            parser.parse(text)

    assert any(
        rec.name == PARSER_LOGGER_NAME
        and rec.levelno == logging.WARNING
        and "Strict parsing failed. Invalid JSON: Extra data" in rec.message
        for rec in caplog.get_records(when="call")
    )

@pytest.mark.asyncio()
async def test_parse_non_strict_extract_from_markdown_code_block(json_parser_non_strict: JSONOutputParserPlugin):
    parser = await json_parser_non_strict
    text = 'Some preamble...\n```json\n{"data": "inside code block"}\n```\nSome epilogue.'
    parsed = parser.parse(text)
    assert parsed == {"data": "inside code block"}

    text_generic_block = '```\n{"generic": true}\n```'
    parsed_generic = parser.parse(text_generic_block)
    assert parsed_generic == {"generic": True}

@pytest.mark.asyncio()
async def test_parse_non_strict_extract_direct_json_object(json_parser_non_strict: JSONOutputParserPlugin):
    parser = await json_parser_non_strict
    text = 'Thought: I should output JSON. {"result": "success", "items": [1, 2]}'
    parsed = parser.parse(text)
    assert parsed == {"result": "success", "items": [1, 2]}

@pytest.mark.asyncio()
async def test_parse_non_strict_extract_direct_json_array(json_parser_non_strict: JSONOutputParserPlugin):
    parser = await json_parser_non_strict
    text = 'The LLM decided to return an array: ["first", {"second": null}, true]'
    parsed = parser.parse(text)
    assert parsed == ["first", {"second": None}, True]

@pytest.mark.asyncio()
async def test_parse_non_strict_no_json_found(json_parser_non_strict: JSONOutputParserPlugin, caplog: pytest.LogCaptureFixture):
    # caplog.set_level(logging.WARNING) # Not needed if using caplog.at_level
    parser = await json_parser_non_strict
    text = "This is just plain text without any JSON structure."
    with caplog.at_level(logging.WARNING, logger=PARSER_LOGGER_NAME):
        with pytest.raises(ValueError, match="No parsable JSON block found in the input text."):
            parser.parse(text)
    assert any(
        rec.name == PARSER_LOGGER_NAME
        and rec.levelno == logging.WARNING
        and "No JSON block found in text_output" in rec.message
        for rec in caplog.get_records(when="call")
    )

@pytest.mark.asyncio()
async def test_parse_non_strict_malformed_json_in_code_block(json_parser_non_strict: JSONOutputParserPlugin, caplog: pytest.LogCaptureFixture):
    parser = await json_parser_non_strict
    text_input = "```json\n{key: 'malformed, missing quotes'}\n```"

    with caplog.at_level(logging.DEBUG, logger=PARSER_LOGGER_NAME):
        with pytest.raises(ValueError, match="No parsable JSON block found in the input text."):
            parser.parse(text_input)

    expected_debug_log_fragment = f"{parser.plugin_id}: Found ```json``` block, but content is not valid JSON:"

    debug_log_found = False
    for record in caplog.get_records(when="call"):
        if record.name == PARSER_LOGGER_NAME and record.levelno == logging.DEBUG:
            if expected_debug_log_fragment in record.message:
                debug_log_found = True
                break

    assert debug_log_found, \
        f"Expected DEBUG log containing '{expected_debug_log_fragment}' not found. Captured logs: {caplog.text}"


@pytest.mark.asyncio()
async def test_parse_empty_input_raises_value_error(json_parser_non_strict: JSONOutputParserPlugin):
    parser = await json_parser_non_strict
    with pytest.raises(ValueError, match="Input text_output is empty or whitespace."):
        parser.parse("")
    with pytest.raises(ValueError, match="Input text_output is empty or whitespace."):
        parser.parse("   ")

@pytest.mark.asyncio()
async def test_teardown(json_parser_non_strict: JSONOutputParserPlugin, caplog: pytest.LogCaptureFixture):
    parser = await json_parser_non_strict

    with caplog.at_level(logging.DEBUG, logger=PARSER_LOGGER_NAME):
        await parser.teardown()

    found_log = False
    expected_message_part = f"{parser.plugin_id}: Teardown complete."
    # Check records specifically from the 'call' phase of the test item
    # which corresponds to when parser.teardown() was executed.
    for rec in caplog.get_records(when="call"):
        if rec.name == PARSER_LOGGER_NAME and rec.levelno == logging.DEBUG:
            if expected_message_part in rec.message:
                found_log = True
                break

    assert found_log, \
        f"Expected DEBUG log containing '{expected_message_part}' not found. Captured logs during call: {[r.message for r in caplog.get_records(when='call') if r.name == PARSER_LOGGER_NAME]}. Full caplog.text: {caplog.text}"
