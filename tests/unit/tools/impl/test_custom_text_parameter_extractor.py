### tests/unit/tools/impl/test_custom_text_parameter_extractor.py
import pytest
from genie_tooling.tools.impl.custom_text_parameter_extractor import (
    custom_text_parameter_extractor,
)


@pytest.mark.asyncio
class TestCustomTextParameterExtractor:
    async def test_successful_extraction(self):
        text = "The MMLU score is 85.7 and the release date was 2024-05-12."
        params = ["score", "release_date"]
        regex = [r"MMLU score is ([\d\.]+)", r"release date was ([\d-]+)"]
        result = await custom_text_parameter_extractor(text, params, regex)
        assert result == {"score": 85.7, "release_date": "2024-05-12"}

    async def test_numeric_coercion(self):
        text = "Value: 123, Ratio: -0.75, Flag: True"
        params = ["int_val", "float_val", "bool_val"]
        regex = [r"Value: ([\d]+)", r"Ratio: ([-]?[\d\.]+)", r"Flag: (True)"]
        result = await custom_text_parameter_extractor(text, params, regex)
        assert result == {"int_val": 123, "float_val": -0.75, "bool_val": "True"}

    async def test_no_match_returns_none(self):
        text = "No relevant data here."
        params = ["missing_item"]
        regex = [r"item: (\w+)"]
        result = await custom_text_parameter_extractor(text, params, regex)
        assert result == {"missing_item": None}

    async def test_mismatched_arg_lengths(self):
        text = "any text"
        params = ["param1"]
        regex = ["regex1", "regex2"]
        result = await custom_text_parameter_extractor(text, params, regex)
        assert "error" in result
        assert "Mismatched argument lengths" in result["error"]

    async def test_empty_text_content(self):
        text = ""
        params = ["some_param"]
        regex = ["some_regex"]
        result = await custom_text_parameter_extractor(text, params, regex)
        assert result == {"some_param": None}

    async def test_invalid_regex_pattern(self):
        text = "some text"
        params = ["bad_pattern_param"]
        regex = ["([a-z]"]  # Unbalanced parenthesis
        result = await custom_text_parameter_extractor(text, params, regex)
        assert result == {"bad_pattern_param": None}

    async def test_empty_param_name_or_pattern(self):
        text = "data1=10 data2=20"
        params = ["good_param", ""]
        regex = ["data1=(\\d+)", "data2=(\\d+)"]
        result = await custom_text_parameter_extractor(text, params, regex)
        assert result.get("good_param") == 10
        assert "extraction_error_param_1" in result