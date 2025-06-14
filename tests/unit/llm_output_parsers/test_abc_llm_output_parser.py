### tests/unit/prompts/llm_output_parsers/test_abc_llm_output_parser.py
import json
import logging
from typing import Any, Optional

import pytest
from genie_tooling.core.types import Plugin
from genie_tooling.prompts.llm_output_parsers.abc import LLMOutputParserPlugin


class DefaultImplParser(LLMOutputParserPlugin, Plugin):
    """Minimal concrete implementation of the parser protocol for testing defaults."""

    plugin_id: str = "default_impl_parser_v1"
    description: str = "A parser using only default ABC implementations."

    async def setup(self, config: Optional[dict[str, Any]] = None) -> None:
        pass

    async def teardown(self) -> None:
        pass


@pytest.fixture
def default_parser() -> DefaultImplParser:
    return DefaultImplParser()


def test_default_parse_logs_warning(default_parser: DefaultImplParser, caplog):
    """Test that the default parse method logs a warning."""
    with caplog.at_level(logging.WARNING):
        default_parser.parse("some text")
    assert f"LLMOutputParserPlugin '{default_parser.plugin_id}' parse method not implemented." in caplog.text


def test_default_parse_returns_input_for_non_json(default_parser: DefaultImplParser):
    """Test default parse returns the input string if it's not JSON-like."""
    input_text = "This is just a regular string."
    result = default_parser.parse(input_text)
    assert result == input_text


def test_default_parse_attempts_json_load_for_json_like_string(default_parser: DefaultImplParser):
    """Test default parse attempts to decode a string that looks like JSON."""
    json_string = '{"key": "value", "items": [1, 2]}'
    expected_dict = {"key": "value", "items": [1, 2]}
    result = default_parser.parse(json_string)
    assert result == expected_dict


def test_default_parse_returns_string_if_json_decode_fails(default_parser: DefaultImplParser):
    """Test default parse returns the original string if JSON decoding fails."""
    malformed_json = '{"key": "value", "items": [1, 2'  # Missing closing brace and bracket
    result = default_parser.parse(malformed_json)
    assert result == malformed_json