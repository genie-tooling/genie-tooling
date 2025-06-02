### tests/unit/tools/formatters/test_formatters.py
"""Unit tests for Tool Definition Formatters."""
from typing import Any, Dict

import pytest
from genie_tooling.definition_formatters import (
    CompactTextFormatter,
    HumanReadableJSONFormatter,
    OpenAIFunctionFormatter,
)

# --- Sample Tool Metadata for testing formatters ---

@pytest.fixture
def sample_tool_metadata() -> Dict[str, Any]:
    return {
        "identifier": "sample_tool_v1",
        "name": "Sample Data Tool",
        "description_human": "A sample tool that processes data and returns a summary. For testing formatters.",
        "description_llm": "Processes input data (text) and provides a short summary (text).",
        "input_schema": {
            "type": "object",
            "properties": {
                "input_text": {"type": "string", "description": "The text data to be processed."},
                "processing_level": {
                    "type": "integer",
                    "description": "Level of processing (1-5).",
                    "default": 3,
                    "enum": [1, 2, 3, 4, 5]
                },
                "optional_flag": {"type": "boolean", "description": "An optional flag."}
            },
            "required": ["input_text", "processing_level"]
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "The generated summary."},
                "char_count": {"type": "integer"}
            },
            "required": ["summary"]
        },
        "key_requirements": [{"name": "SAMPLE_API_KEY", "description": "A dummy API key for this sample."}],
        "tags": ["sample", "data_processing", "formatter_test"],
        "version": "1.0.2",
        "cacheable": True,
        "cache_ttl_seconds": 600
    }

@pytest.fixture
def minimal_tool_metadata() -> Dict[str, Any]:
    return {
        "identifier": "minimal_tool_id",
        "name": "Minimal Tool",
        "description_llm": "A very basic tool.",
        "input_schema": {"type": "object", "properties": {}}, # No params
        "output_schema": {"type": "object", "properties": {}}
        # Other fields omitted
    }

# --- Tests for CompactTextFormatter ---

@pytest.fixture
def compact_formatter() -> CompactTextFormatter:
    return CompactTextFormatter()

def test_compact_formatter_basic(compact_formatter: CompactTextFormatter, sample_tool_metadata: Dict[str, Any]):
    formatted_str = compact_formatter.format(sample_tool_metadata)
    assert isinstance(formatted_str, str)
    assert "ToolName: Sample Data Tool" in formatted_str
    assert "ToolID: sample_tool_v1" in formatted_str
    assert "Purpose: Processes input data (text) and provides a short summary (text)." in formatted_str
    assert "input_text(string, req)" in formatted_str
    assert "processing_level(integer, req, enum[1,2,3,...])" in formatted_str
    assert "optional_flag(boolean, opt)" in formatted_str
    assert "Tags: sample, data_processing, formatter_test" in formatted_str

def test_compact_formatter_minimal_tool(compact_formatter: CompactTextFormatter, minimal_tool_metadata: Dict[str, Any]):
    formatted_str = compact_formatter.format(minimal_tool_metadata)
    assert "ToolName: Minimal Tool" in formatted_str
    assert "ToolID: minimal_tool_id" in formatted_str
    assert "Purpose: A very basic tool." in formatted_str
    assert "Args: no parameters" in formatted_str
    assert "Tags:" not in formatted_str # No tags in minimal metadata

def test_compact_formatter_long_description_truncation(compact_formatter: CompactTextFormatter):
    long_desc_metadata = {
        "identifier": "long_desc_tool", "name": "Long Desc",
        "description_llm": "This is a very long description that definitely exceeds two hundred characters and should be truncated by the formatter to ensure that the output remains compact and does not overwhelm the context window when provided to a large language model or used in other constrained environments where brevity is key for performance and readability. It keeps going and going and going and going."
    }
    formatted_str = compact_formatter.format(long_desc_metadata)
    assert "Purpose: This is a very long description that definitely exceeds two hundred characters" in formatted_str
    assert "..." in formatted_str
    assert "keeps going and going" not in formatted_str # Check that it's actually truncated

# --- Tests for HumanReadableJSONFormatter ---

@pytest.fixture
def hr_json_formatter() -> HumanReadableJSONFormatter:
    return HumanReadableJSONFormatter()

def test_hr_json_formatter_returns_dict(hr_json_formatter: HumanReadableJSONFormatter, sample_tool_metadata: Dict[str, Any]):
    formatted_data = hr_json_formatter.format(sample_tool_metadata)
    assert isinstance(formatted_data, dict)
    assert formatted_data["identifier"] == sample_tool_metadata["identifier"]
    assert formatted_data["name"] == sample_tool_metadata["name"]
    assert "input_schema" in formatted_data
    assert "output_schema" in formatted_data
    assert formatted_data.get("cacheable") is True # Checks if defaults are included for clarity

# --- Tests for OpenAIFunctionFormatter ---

@pytest.fixture
def openai_formatter() -> OpenAIFunctionFormatter:
    return OpenAIFunctionFormatter()

def test_openai_formatter_basic_structure(openai_formatter: OpenAIFunctionFormatter, sample_tool_metadata: Dict[str, Any]):
    formatted_func = openai_formatter.format(sample_tool_metadata)
    assert isinstance(formatted_func, dict)
    assert formatted_func["type"] == "function"
    assert "function" in formatted_func

    func_details = formatted_func["function"]
    assert func_details["name"] == "sample_tool_v1" # Identifier is preferred and valid
    assert func_details["description"] == sample_tool_metadata["description_llm"]

    params_schema = func_details["parameters"]
    assert params_schema["type"] == "object"
    assert "input_text" in params_schema["properties"]
    assert params_schema["properties"]["input_text"]["type"] == "string"
    assert "processing_level" in params_schema["properties"]
    assert params_schema["properties"]["processing_level"]["enum"] == [1, 2, 3, 4, 5]
    assert "optional_flag" in params_schema["properties"]
    assert params_schema["required"] == ["input_text", "processing_level"]

def test_openai_formatter_tool_name_sanitization(openai_formatter: OpenAIFunctionFormatter):
    metadata_bad_name = {
        "identifier": "tool with spaces-and-symbols!",
        "description_llm": "Test name sanitization."
    }
    formatted_func = openai_formatter.format(metadata_bad_name)
    assert formatted_func["function"]["name"] == "tool_with_spaces_and_symbols_" # Sanitized

    metadata_long_name = {
        "identifier": "a_very_very_long_tool_identifier_that_exceeds_the_openai_limit_of_sixty_four_characters_for_sure",
        "description_llm": "Long name."
    }
    formatted_func = openai_formatter.format(metadata_long_name)
    assert len(formatted_func["function"]["name"]) == 64
    assert formatted_func["function"]["name"].startswith("a_very_very_long_tool_identifier_that_exceeds_the_openai_limit")

def test_openai_formatter_minimal_tool(openai_formatter: OpenAIFunctionFormatter, minimal_tool_metadata: Dict[str, Any]):
    formatted_func = openai_formatter.format(minimal_tool_metadata)
    func_details = formatted_func["function"]
    assert func_details["name"] == "minimal_tool_id"
    assert func_details["description"] == minimal_tool_metadata["description_llm"]
    assert func_details["parameters"]["type"] == "object"
    assert func_details["parameters"]["properties"] == {} # No params defined

def test_openai_formatter_missing_description(openai_formatter: OpenAIFunctionFormatter):
    metadata_no_desc = {"identifier": "no_desc_tool"}
    formatted_func = openai_formatter.format(metadata_no_desc)
    assert "Executes the 'no_desc_tool' tool." in formatted_func["function"]["description"]

def test_openai_formatter_cleans_additional_properties(openai_formatter: OpenAIFunctionFormatter):
    metadata_with_add_props = {
        "identifier": "add_props_tool",
        "description_llm": "Test additionalProperties.",
        "input_schema": {
            "type": "object",
            "properties": {"param1": {"type": "string"}},
            "additionalProperties": True # This should be removed by the formatter
        }
    }
    formatted_func = openai_formatter.format(metadata_with_add_props)
    assert "additionalProperties" not in formatted_func["function"]["parameters"]

    metadata_nested_add_props = {
        "identifier": "nested_add_props_tool",
        "description_llm": "Test nested additionalProperties.",
        "input_schema": {
            "type": "object",
            "properties": {
                "param1": {"type": "string"},
                "nested_obj": {
                    "type": "object",
                    "properties": {"sub_param": {"type": "integer"}},
                    "additionalProperties": False # This should be kept
                }
            }
        }
    }
    formatted_func = openai_formatter.format(metadata_nested_add_props)
    params_schema = formatted_func["function"]["parameters"]
    assert "additionalProperties" not in params_schema # Root level
    assert params_schema["properties"]["nested_obj"]["additionalProperties"] is False # Nested level
