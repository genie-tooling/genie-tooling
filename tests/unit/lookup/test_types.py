### tests/unit/lookup/test_types.py
"""Unit tests for lookup.types module."""

from genie_tooling.lookup.types import RankedToolResult


def test_ranked_tool_result_initialization_all_params():
    tool_id = "test_tool_001"
    score = 0.95
    matched_data = {"name": "Test Tool", "description": "A tool for testing."}
    description_snippet = "Found matching 'test'."

    result = RankedToolResult(
        tool_identifier=tool_id,
        score=score,
        matched_tool_data=matched_data,
        description_snippet=description_snippet
    )

    assert result.tool_identifier == tool_id
    assert result.score == score
    assert result.matched_tool_data == matched_data
    assert result.description_snippet == description_snippet

def test_ranked_tool_result_initialization_minimal_params():
    tool_id = "test_tool_002"
    score = 0.80

    result = RankedToolResult(tool_identifier=tool_id, score=score)

    assert result.tool_identifier == tool_id
    assert result.score == score
    assert result.matched_tool_data == {}  # Default value
    assert result.description_snippet is None  # Default value

def test_ranked_tool_result_repr():
    result = RankedToolResult(tool_identifier="repr_tool", score=0.7512345)
    assert repr(result) == "RankedToolResult(id='repr_tool', score=0.7512)"

    result_no_score = RankedToolResult(tool_identifier="repr_tool_no_score_val", score=0.0) # Score will be formatted
    assert repr(result_no_score) == "RankedToolResult(id='repr_tool_no_score_val', score=0.0000)"


def test_ranked_tool_result_to_dict():
    tool_id = "dict_tool_003"
    score = 0.65
    matched_data = {"keywords": ["dict", "convert"]}
    description_snippet = "Converts to dictionary."

    result = RankedToolResult(
        tool_identifier=tool_id,
        score=score,
        matched_tool_data=matched_data,
        description_snippet=description_snippet
    )

    expected_dict = {
        "tool_identifier": tool_id,
        "score": score,
        "matched_tool_data": matched_data,
        "description_snippet": description_snippet,
    }
    assert result.to_dict() == expected_dict

def test_ranked_tool_result_to_dict_with_defaults():
    result = RankedToolResult(tool_identifier="default_dict_tool", score=0.5)
    expected_dict = {
        "tool_identifier": "default_dict_tool",
        "score": 0.5,
        "matched_tool_data": {},
        "description_snippet": None,
    }
    assert result.to_dict() == expected_dict
