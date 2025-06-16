# tests/unit/utils/test_placeholder_resolution.py
import pytest
from genie_tooling.utils.placeholder_resolution import resolve_placeholders


@pytest.fixture()
def sample_scratchpad():
    """Provides a sample scratchpad for testing."""
    return {
        "outputs": {
            "step1_search": {
                "results": [
                    {"url": "http://example.com/1", "title": "First Result"},
                    {"url": "http://example.com/2", "title": "Second Result"},
                ]
            },
            "step2_extraction": {
                "score": 95,
                "confidence": 0.98,
                "details": {"status": "complete", "items": ["a", "b"]},
            },
            "step3_page_content": {
                "content": "Full text of the web page.",
                "url": "http://example.com/1",
            },
            "step4_failed_tool": None,
        }
    }


# --- Test Cases ---


def test_resolve_no_placeholders(sample_scratchpad):
    """Test that structures without placeholders are returned unchanged."""
    assert resolve_placeholders("Just a plain string", sample_scratchpad) == "Just a plain string"
    assert resolve_placeholders(["a", 1, True], sample_scratchpad) == ["a", 1, True]
    assert resolve_placeholders({"key": "value"}, sample_scratchpad) == {"key": "value"}


def test_resolve_full_string_placeholder(sample_scratchpad):
    """Test when the entire string is a placeholder."""
    # Should return the actual object, not its string representation
    assert resolve_placeholders("{outputs.step1_search}", sample_scratchpad) == sample_scratchpad["outputs"]["step1_search"]
    assert resolve_placeholders("{{outputs.step2_extraction.score}}", sample_scratchpad) == 95
    assert resolve_placeholders("{outputs.step4_failed_tool}", sample_scratchpad) is None


def test_resolve_embedded_string_placeholder(sample_scratchpad):
    """Test when a placeholder is part of a larger string."""
    input_str = "The score was {outputs.step2_extraction.score} with confidence {{outputs.step2_extraction.confidence}}."
    expected_str = "The score was 95 with confidence 0.98."
    assert resolve_placeholders(input_str, sample_scratchpad) == expected_str


def test_resolve_embedded_dict_placeholder(sample_scratchpad):
    """Test that embedded dicts/lists are JSON stringified."""
    input_str = "Search results are: {outputs.step1_search.results}"
    expected_str = 'Search results are: [{"url": "http://example.com/1", "title": "First Result"}, {"url": "http://example.com/2", "title": "Second Result"}]'
    assert resolve_placeholders(input_str, sample_scratchpad) == expected_str


def test_resolve_nested_list_and_dict_access(sample_scratchpad):
    """Test complex pathing through lists and dicts."""
    assert resolve_placeholders("{outputs.step1_search.results.0.url}", sample_scratchpad) == "http://example.com/1"
    assert resolve_placeholders("Status is {{outputs.step2_extraction.details.status}}", sample_scratchpad) == "Status is complete"


def test_resolve_in_nested_structure(sample_scratchpad):
    """Test placeholder resolution inside lists and dicts."""
    input_structure = {
        "summary": "URL of first result: {outputs.step1_search.results.0.url}",
        "details": ["Score: {{outputs.step2_extraction.score}}", "Just a string"],
    }
    expected_structure = {
        "summary": "URL of first result: http://example.com/1",
        "details": ["Score: 95", "Just a string"],
    }
    assert resolve_placeholders(input_structure, sample_scratchpad) == expected_structure


def test_resolve_missing_outputs_key_in_scratchpad():
    """Test that a ValueError is raised if 'outputs' key is missing."""
    with pytest.raises(ValueError, match="'outputs' key not found in scratchpad"):
        resolve_placeholders("{outputs.var.key}", {"other_key": {}})


def test_resolve_invalid_path_key_not_found(sample_scratchpad):
    """Test path resolution when a key does not exist."""
    # FIX: The stricter logic now raises KeyError. The test must expect this.
    with pytest.raises(KeyError, match="Key 'non_existent_key' not found"):
        resolve_placeholders("{outputs.step1_search.non_existent_key}", sample_scratchpad)

    # FIX: For embedded placeholders, the KeyError is wrapped in a ValueError by the top-level function.
    with pytest.raises(ValueError, match="Error resolving placeholder"):
        resolve_placeholders("Value: {outputs.step1_search.non_existent_key}", sample_scratchpad)


def test_resolve_invalid_path_index_out_of_bounds(sample_scratchpad):
    """Test path resolution when a list index is out of bounds."""
    # FIX: The stricter logic now raises IndexError. The test must expect this.
    with pytest.raises(IndexError, match="List index 5 out of bounds"):
        resolve_placeholders("{outputs.step1_search.results.5}", sample_scratchpad)


def test_resolve_invalid_path_key_on_list(sample_scratchpad):
    """Test trying to access a dictionary key on a list."""
    with pytest.raises(ValueError, match="Invalid list index 'title'"):
        resolve_placeholders("{outputs.step1_search.results.title}", sample_scratchpad)


def test_resolve_invalid_path_index_on_primitive(sample_scratchpad):
    """Test trying to access a sub-path on a primitive value (int, str, etc.)."""
    with pytest.raises(TypeError, match="Cannot access key/index 'invalid_subpath' on non-dict/list type 'int'"):
        resolve_placeholders("{outputs.step2_extraction.score.invalid_subpath}", sample_scratchpad)