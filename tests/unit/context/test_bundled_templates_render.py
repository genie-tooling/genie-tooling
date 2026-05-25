"""
Verifies the bundled cqs prompt templates are syntactically valid Jinja2 and
render against representative raw_data shapes. The bundled templates use
dotted attribute access (`raw_data.result.fact.value`) which only works under
a Jinja-style engine; this test guards against regressions if someone tries
to switch the default engine back to str.format.
"""
from __future__ import annotations

import importlib.resources
from pathlib import Path

import pytest

from genie_tooling.prompts.impl.jinja2_chat_template import Jinja2ChatTemplatePlugin


def _bundled_templates_dir() -> Path:
    return Path(str(importlib.resources.files("genie_tooling.context") / "prompt_templates"))


@pytest.mark.asyncio
async def test_direct_fact_formulation_template_renders():
    """`direct_fact_formulation.prompt` mirrors the karta-style lookup output
    where raw_data.result.fact.value is the answer string."""
    template_path = _bundled_templates_dir() / "direct_fact_formulation.prompt"
    assert template_path.is_file(), f"bundled template missing: {template_path}"
    template_content = template_path.read_text()

    plugin = Jinja2ChatTemplatePlugin()
    await plugin.setup()

    data = {
        "original_query": "What is the boiling point of water?",
        "raw_data": {"result": {"fact": {"value": "100 degrees Celsius"}}},
        "formulation_constraints": {"tone": "encyclopedic"},
    }
    rendered = await plugin.render(template_content, data)
    assert "What is the boiling point of water?" in rendered
    assert "100 degrees Celsius" in rendered
    # The template must not leak unfilled placeholders.
    assert "{{" not in rendered and "}}" not in rendered


@pytest.mark.asyncio
async def test_summarize_agent_output_template_renders():
    """`summarize_agent_output.prompt` consumes the DeepResearchAgent's
    final_answer field nested inside the agent output dict."""
    template_path = _bundled_templates_dir() / "summarize_agent_output.prompt"
    assert template_path.is_file(), f"bundled template missing: {template_path}"
    template_content = template_path.read_text()

    plugin = Jinja2ChatTemplatePlugin()
    await plugin.setup()

    data = {
        "original_query": "What's the latest on transformer interpretability?",
        "raw_data": {
            "result": {
                "final_answer": (
                    "Sparse autoencoders applied to residual streams have been "
                    "showing strong polysemanticity decomposition in 2024–2025."
                )
            }
        },
    }
    rendered = await plugin.render(template_content, data)
    assert "transformer interpretability" in rendered
    assert "Sparse autoencoders" in rendered
    assert "{{" not in rendered and "}}" not in rendered


@pytest.mark.asyncio
async def test_default_formulation_prompt_renders():
    """``default_formulation_prompt.prompt`` is the fallback the
    LlmPromptFormulationPlugin uses when a rule doesn't set
    ``prompt_template_id``. Must be present and renderable for the
    Phase 4 bundled rules to produce real responses."""
    template_path = _bundled_templates_dir() / "default_formulation_prompt.prompt"
    assert template_path.is_file(), f"bundled template missing: {template_path}"
    template_content = template_path.read_text()

    plugin = Jinja2ChatTemplatePlugin()
    await plugin.setup()

    data = {
        "original_query": "calculate 12 times 5",
        "raw_data": {"tool_result": {"result": 60.0, "error_message": None}},
        "formulation_constraints": {"format": "direct_answer"},
    }
    rendered = await plugin.render(template_content, data)
    assert "calculate 12 times 5" in rendered
    # The raw_data stringification must be visible to the LLM so it can
    # extract the numeric value.
    assert "60" in rendered
    # No HTML-encoded entities — the LLM should see real quotes.
    assert "&#39;" not in rendered and "&quot;" not in rendered
    # Jinja must have processed the placeholders (no unrendered `{{ var }}`).
    # We check for the specific original placeholder name rather than `{{`
    # alone because dict reprs contain `{}` braces.
    assert "{{ original_query }}" not in rendered
    assert "{{ raw_data }}" not in rendered


@pytest.mark.asyncio
async def test_template_render_handles_missing_nested_field_gracefully():
    """When the raw_data shape doesn't match the template's expected access
    path, Jinja2's default behavior should not blow up — it returns the
    error message string from the plugin, never raising past the formulation
    plugin's try/except. The cqs LlmPromptFormulationPlugin catches anyway."""
    plugin = Jinja2ChatTemplatePlugin()
    await plugin.setup()

    template = "{{ raw_data.result.nonexistent.deep.field }}"
    rendered = await plugin.render(template, {"raw_data": {"result": {}}})
    # The plugin's render returns an error string on UndefinedError rather than
    # raising. Either way, the formulation plugin downstream handles it.
    assert isinstance(rendered, str)
