"""Unit tests for the C_F constraint translator (A1).

The translator is the audit-facing surface between rule actions and LLM
behavior. Every audit record reproduces the exact instruction text, so
this needs to be deterministic and stable across runs.
"""
from __future__ import annotations

import pytest
from genie_tooling.context.constraints import (
    formulation_constraints_to_instructions,
)


def test_empty_constraints_returns_none():
    assert formulation_constraints_to_instructions(None) is None
    assert formulation_constraints_to_instructions({}) is None


def test_only_non_behavioral_keys_returns_none():
    """`prompt_template_id` controls template selection; instructing the LLM
    to "apply prompt_template_id = direct_fact_formulation" would be nonsense."""
    constraints = {
        "prompt_template_id": "direct_fact_formulation",
        "derivation_strategy_id": "generic_tool_derivation_v1",
        "command_processor_id": "some_processor_v1",
        "tool_id": "calculator_tool",
        "params": {"x": 1},
    }
    assert formulation_constraints_to_instructions(constraints) is None


def test_tone_is_rendered():
    out = formulation_constraints_to_instructions({"tone": "calm and reassuring"})
    assert out is not None
    assert "Response guidelines:" in out
    assert "calm and reassuring" in out
    assert "tone" in out.lower()


@pytest.mark.parametrize(
    "verbosity,expected_phrase",
    [
        ("low", "concise"),
        ("concise", "concise"),
        ("short", "concise"),
        ("brief", "concise"),
        ("moderate", "moderate length"),
        ("medium", "moderate length"),
        ("balanced", "moderate length"),
        ("high", "detailed"),
        ("detailed", "detailed"),
        ("verbose", "detailed"),
    ],
)
def test_verbosity_levels_map_to_distinct_phrases(verbosity, expected_phrase):
    out = formulation_constraints_to_instructions({"verbosity": verbosity})
    assert out is not None
    assert expected_phrase in out.lower()


def test_verbosity_unknown_value_falls_through():
    out = formulation_constraints_to_instructions({"verbosity": "telegraphic"})
    assert out is not None
    assert "telegraphic" in out


@pytest.mark.parametrize(
    "format_value,expected_phrase",
    [
        ("direct_answer", "direct answer only"),
        ("bullet_list", "bulleted list"),
        ("bulleted", "bulleted list"),
        ("numbered_list", "numbered list"),
        ("json", "valid JSON only"),
    ],
)
def test_format_directives(format_value, expected_phrase):
    out = formulation_constraints_to_instructions({"format": format_value})
    assert out is not None
    assert expected_phrase in out


def test_empathy_high():
    out = formulation_constraints_to_instructions({"empathy_level": "high"})
    assert out is not None
    assert "empathy" in out.lower()


def test_empathy_low_means_neutral():
    out = formulation_constraints_to_instructions({"empathy_level": "low"})
    assert out is not None
    assert "neutral" in out.lower() or "factual" in out.lower()


def test_redact_list_renders_as_joined_list():
    out = formulation_constraints_to_instructions(
        {"redact": ["internal_terms", "customer_names"]}
    )
    assert out is not None
    assert "internal_terms" in out
    assert "customer_names" in out
    assert "Do NOT" in out


def test_redact_single_string():
    out = formulation_constraints_to_instructions({"redact": "PII"})
    assert out is not None
    assert "PII" in out
    assert "Do NOT" in out


def test_unknown_key_falls_through_with_generic_template():
    out = formulation_constraints_to_instructions({"made_up_constraint": "weird value"})
    assert out is not None
    assert "made_up_constraint" in out
    assert "weird value" in out


def test_unknown_key_with_dict_value_uses_json():
    out = formulation_constraints_to_instructions(
        {"complex_setting": {"a": 1, "b": [2, 3]}}
    )
    assert out is not None
    assert "complex_setting" in out
    # The dict gets json-serialized so it stays readable
    assert '"a"' in out or "'a'" in out


def test_multiple_constraints_compose_in_order():
    """Insertion order matters for determinism — same constraint dict in,
    same instruction text out."""
    out = formulation_constraints_to_instructions(
        {
            "tone": "formal",
            "verbosity": "concise",
            "empathy_level": "low",
        }
    )
    assert out is not None
    tone_idx = out.find("formal")
    verbosity_idx = out.find("concise")
    empathy_idx = out.find("neutral")
    assert tone_idx >= 0 and verbosity_idx >= 0 and empathy_idx >= 0
    assert tone_idx < verbosity_idx < empathy_idx


def test_non_behavioral_keys_filtered_when_mixed_with_behavioral():
    """Real-world C_F dicts include both `prompt_template_id` and stylistic
    keys — the translator must skip the wiring key but render the rest."""
    out = formulation_constraints_to_instructions(
        {
            "prompt_template_id": "direct_fact_formulation",
            "tone": "encyclopedic",
            "verbosity": "concise",
        }
    )
    assert out is not None
    assert "encyclopedic" in out
    assert "concise" in out
    # The wiring key must NOT appear as an instruction.
    assert "prompt_template_id" not in out
    assert "direct_fact_formulation" not in out


def test_deterministic_same_input_same_output():
    """Audit reproducibility: same dict yields same text twice."""
    constraints = {
        "tone": "calm and reassuring",
        "verbosity": "moderate",
        "empathy_level": "high",
    }
    a = formulation_constraints_to_instructions(constraints)
    b = formulation_constraints_to_instructions(constraints)
    assert a == b


def test_output_starts_with_header():
    """The header `Response guidelines:` signals to the LLM that the lines
    that follow are instructions, not user content."""
    out = formulation_constraints_to_instructions({"tone": "formal"})
    assert out is not None
    assert out.startswith("Response guidelines:")


def test_each_instruction_is_bulleted():
    out = formulation_constraints_to_instructions(
        {"tone": "formal", "verbosity": "concise"}
    )
    assert out is not None
    bullet_lines = [line for line in out.split("\n") if line.startswith("- ")]
    assert len(bullet_lines) == 2
