### tests/unit/utils/test_json_parser_utils.py
"""
Unit tests for the `extract_json_block` utility function.
This test suite aims for 100% coverage of all extraction paths and edge cases,
testing each extraction method and their priority.
"""
import json
from typing import Optional

import pytest
from genie_tooling.utils.json_parser_utils import extract_json_block


def assert_json_equal(result: Optional[str], expected: Optional[str], test_id: str):
    """
    Compares two JSON strings by loading them into Python objects,
    ignoring whitespace and key order differences.
    """
    if expected is None:
        assert result is None, f"Test '{test_id}': Expected None, but got a result."
        return
    assert result is not None, f"Test '{test_id}': Expected a result, but got None."

    try:
        result_obj = json.loads(result)
        expected_obj = json.loads(expected)
        assert result_obj == expected_obj, f"Test '{test_id}': JSON objects do not match."
    except json.JSONDecodeError:
        pytest.fail(f"Test '{test_id}': Could not decode result or expected value as JSON.")


class TestMarkdownBlockExtraction:
    """Tests focused on extracting JSON from ```...``` blocks."""

    def test_extract_from_json_specific_block(self):
        text = 'Preamble.\n```json\n{"key": "value", "arr": [1, 2]}\n```\nPostamble.'
        expected = '{"key": "value", "arr": [1, 2]}'
        assert_json_equal(extract_json_block(text), expected, "markdown_json_clean")

    def test_extract_from_generic_block(self):
        text = 'Preamble.\n```\n{"generic": true}\n```\nPostamble.'
        expected = '{"generic": true}'
        assert_json_equal(extract_json_block(text), expected, "markdown_generic_clean")

    def test_markdown_block_with_extra_spacing(self):
        text = '```json  \n\n  {"spacing": "test"} \n\n```'
        expected = '{"spacing": "test"}'
        assert_json_equal(extract_json_block(text), expected, "markdown_spacing")

    def test_malformed_json_in_json_block_is_skipped(self):
        # The function should skip the malformed block and find nothing else.
        text = '```json\n{"key": "value", }\n```'
        assert extract_json_block(text) is None, "malformed_json_block"

    def test_non_json_in_generic_block_is_skipped(self):
        text = "```python\nprint('hello')\n```"
        assert extract_json_block(text) is None, "markdown_non_json_content"


class TestRawStringExtraction:
    """Tests focused on extracting JSON directly from the string content."""

    def test_extract_raw_object_embedded(self):
        text = 'Here is the data: {"id": 123, "status": "ok"} and that is all.'
        expected = '{"id": 123, "status": "ok"}'
        assert_json_equal(extract_json_block(text), expected, "raw_object_embedded")

    def test_extract_raw_array_embedded(self):
        text = 'Here are the items: [ "A", "B" ] that we need.'
        expected = '[ "A", "B" ]'
        assert_json_equal(extract_json_block(text), expected, "raw_array_embedded")

    def test_extract_raw_object_at_start(self):
        text = '{"id": 456, "status": "start"} is the first thing.'
        expected = '{"id": 456, "status": "start"}'
        assert_json_equal(extract_json_block(text), expected, "raw_object_at_start")

    def test_finds_first_complete_object_only(self):
        text = 'This is the first: {"a": 1}. This is the second: {"b": 2}.'
        expected = '{"a": 1}'
        assert_json_equal(extract_json_block(text), expected, "raw_first_of_many")

    def test_finds_array_before_object_if_it_appears_first(self):
        text = 'An array [1,2] comes before an object {"a":1}.'
        expected = "[1,2]"
        assert_json_equal(extract_json_block(text), expected, "raw_array_first")

    def test_finds_object_before_array_if_it_appears_first(self):
        text = 'An object {"a":1} comes before an array [1,2].'
        expected = '{"a":1}'
        assert_json_equal(extract_json_block(text), expected, "raw_object_first")

    def test_skips_malformed_raw_object_and_finds_next_valid(self):
        text = 'This is { "bad": json, then {"good": true}'
        expected = '{"good": true}'
        assert_json_equal(extract_json_block(text), expected, "raw_malformed_skip")


class TestPriorityAndEdgeCases:
    """Tests the interaction between different extraction methods and edge cases."""

    def test_priority_markdown_json_block_over_raw_object(self):
        text = '```json\n{"priority": "markdown"}\n``` some text {"priority": "raw"}'
        expected = '{"priority": "markdown"}'
        assert_json_equal(extract_json_block(text), expected, "priority_md_json_vs_raw")

    def test_priority_markdown_generic_block_over_raw_object(self):
        text = '```\n{"priority": "markdown_generic"}\n``` some text {"priority": "raw"}'
        expected = '{"priority": "markdown_generic"}'
        assert_json_equal(extract_json_block(text), expected, "priority_md_generic_vs_raw")

    def test_malformed_markdown_block_falls_back_to_raw_object(self):
        text = '```json\n{"malformed":, }\n``` some text {"fallback": "correct"}'
        expected = '{"fallback": "correct"}'
        assert_json_equal(extract_json_block(text), expected, "priority_fallback_to_raw")

    def test_handles_empty_and_none_inputs_gracefully(self):
        assert extract_json_block("") is None, "empty_string"
        assert extract_json_block("   \n\t  ") is None, "whitespace_only"
        assert extract_json_block(None) is None, "none_input" # type: ignore

    def test_no_json_at_all(self):
        text = "This string contains no JSON structures like objects or arrays at all."
        assert extract_json_block(text) is None, "no_json_present"
