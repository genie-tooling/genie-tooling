"""Phase 4 — D1-D4 coverage gaps:

  * D1: every rule condition operator (`==`, `!=`, `>`, `<`, `>=`, `<=`,
    `in`, `contains`) exercised against representative inputs.
  * D2: `_aggregate_constraints` ops `set`, `default`, `add` all exercised.
  * D3: VectorDBRuleEnginePlugin.load_rules audit-mode caveat.
  * D4: HeuristicPredicateExtractorPlugin keyword table-driven test.
"""
from __future__ import annotations

from typing import Tuple
from unittest.mock import MagicMock

import pytest
from genie_tooling.context.manager import ContextManager
from genie_tooling.context.plugins.predicate_extractors.heuristic_extractor import (
    HeuristicPredicateExtractorPlugin,
)
from genie_tooling.context.plugins.rule_engines.filesystem_engine import (
    FileSystemRuleEnginePlugin,
)

# ---------------------------------------------------------------------------
# D1: rule condition operators
# ---------------------------------------------------------------------------


@pytest.fixture()
def engine():
    return FileSystemRuleEnginePlugin()


@pytest.mark.parametrize(
    "actual, op, expected, should_match",
    [
        # ==
        ("admin", "==", "admin", True),
        ("admin", "==", "user", False),
        # !=
        ("admin", "!=", "user", True),
        ("admin", "!=", "admin", False),
        # >
        (5, ">", 3, True),
        (3, ">", 5, False),
        (5.0, ">", 4.9, True),
        # <
        (3, "<", 5, True),
        (5, "<", 3, False),
        # >=
        (5, ">=", 5, True),
        (5, ">=", 6, False),
        (5, ">=", 4, True),
        # <=
        (5, "<=", 5, True),
        (5, "<=", 4, False),
        (3, "<=", 5, True),
        # in (membership)
        ("admin", "in", ["admin", "manager"], True),
        ("intern", "in", ["admin", "manager"], False),
        # contains (substring of stringified)
        ("hello world", "contains", "world", True),
        ("hello world", "contains", "xyz", False),
        # Defensive: ValueError/TypeError paths in numeric ops fall through to False
        ("not a number", ">", 5, False),
        (None, "<=", 5, False),
        # Defensive: unknown op falls through to False (engine returns False)
        ("x", "weird_op", "x", False),
    ],
)
def test_evaluate_condition_all_operators(engine, actual, op, expected, should_match):
    assert engine._evaluate_condition(actual, op, expected) is should_match


# ---------------------------------------------------------------------------
# D2: _aggregate_constraints ops
# ---------------------------------------------------------------------------


@pytest.fixture()
def context_manager():
    mock_genie = MagicMock()
    return ContextManager(genie=mock_genie, config={})


def _ranked_rule(actions: list) -> list[Tuple[dict, float]]:
    return [({"rule_id": "TEST", "actions": actions}, 1.0)]


@pytest.mark.asyncio()
async def test_aggregate_op_set(context_manager):
    c_d, c_f = await context_manager._aggregate_constraints(
        _ranked_rule(
            [
                ["C_D", "set", "tool_id", "calculator_tool"],
                ["C_F", "set", "tone", "formal"],
            ]
        )
    )
    assert c_d == {"tool_id": "calculator_tool"}
    assert c_f == {"tone": "formal"}


@pytest.mark.asyncio()
async def test_aggregate_op_default_only_sets_if_absent(context_manager):
    """`default` semantically means "set if not already present". With one
    rule this is equivalent to `set`, but the behavior is preserved."""
    c_d, c_f = await context_manager._aggregate_constraints(
        _ranked_rule(
            [
                ["C_D", "default", "tool_id", "calculator_tool"],
            ]
        )
    )
    assert c_d == {"tool_id": "calculator_tool"}


@pytest.mark.asyncio()
async def test_aggregate_op_default_does_not_overwrite_prior_set(context_manager):
    """When `set` and `default` both target the same key, `set` wins
    because it ran first (the manager processes actions in order)."""
    c_d, c_f = await context_manager._aggregate_constraints(
        _ranked_rule(
            [
                ["C_D", "set", "tool_id", "first_winner"],
                ["C_D", "default", "tool_id", "would_overwrite_but_default"],
            ]
        )
    )
    assert c_d == {"tool_id": "first_winner"}


@pytest.mark.asyncio()
async def test_aggregate_op_add_appends_to_list(context_manager):
    """`add` builds a list of values under the key."""
    c_d, c_f = await context_manager._aggregate_constraints(
        _ranked_rule(
            [
                ["C_F", "add", "redact", "internal_terms"],
                ["C_F", "add", "redact", "customer_names"],
                ["C_F", "add", "redact", "pii"],
            ]
        )
    )
    assert c_f == {"redact": ["internal_terms", "customer_names", "pii"]}


@pytest.mark.asyncio()
async def test_aggregate_empty_ranked_returns_empty(context_manager):
    c_d, c_f = await context_manager._aggregate_constraints([])
    assert c_d == {}
    assert c_f == {}


# ---------------------------------------------------------------------------
# D4: heuristic predicate extractor — every keyword + fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio()
@pytest.mark.parametrize(
    "query, expected_predicate",
    [
        # Each keyword in the extractor's table should produce predicate_<keyword>.
        ("is this thing on", "predicate_is"),
        ("are we there yet", "predicate_are"),
        ("what is the speed of light", "predicate_what"),
        ("who designed this protocol", "predicate_who"),
        ("where is the file", "predicate_where"),
        ("when did this start", "predicate_when"),
        ("can you help me", "predicate_can"),
        ("do this for me please", "predicate_do"),
        ("does it support that", "predicate_does"),
        ("compare these two approaches", "predicate_compare"),
        ("evaluate this expression", "predicate_evaluate"),
        ("summarize the conversation", "predicate_summarize"),
        ("find the maximum value", "predicate_find"),
        ("lookup the user record", "predicate_lookup"),
        ("calculate the integral", "predicate_calculate"),
        # No keyword match — fallback predicate.
        ("hello world", "predicate_generic_inquiry"),
        ("just chatting about life", "predicate_generic_inquiry"),
        # Mixed case still matches (extractor lowercases).
        ("WHAT exactly happened here", "predicate_what"),
        # Punctuation: the heuristic splits on whitespace, so "what?" doesn't
        # match "what". That's an audit-relevant deterministic fact — pinning
        # it here keeps the behavior stable.
        ("what?", "predicate_generic_inquiry"),
    ],
)
async def test_heuristic_extractor_keyword_table(query, expected_predicate):
    plugin = HeuristicPredicateExtractorPlugin()
    pred = await plugin.extract(query, genie=MagicMock())
    assert pred == expected_predicate, f"query={query!r} expected {expected_predicate!r} got {pred!r}"


# ---------------------------------------------------------------------------
# D3: VectorDBRuleEnginePlugin audit-mode caveat
# ---------------------------------------------------------------------------


def test_vectordb_documents_audit_caveat():
    """The plugin's module + class docstrings must explicitly warn about
    the LLM-enriched mode's non-determinism. Removing those warnings
    would mislead operators into using the plugin for audit-bound
    workloads. (D3.)"""
    from genie_tooling.context.plugins.rule_engines import vectordb_engine

    combined = (vectordb_engine.__doc__ or "") + (
        vectordb_engine.VectorDBRuleEnginePlugin.__doc__ or ""
    )
    lower = combined.lower()
    assert "deterministic" in lower, (
        "VectorDBRuleEnginePlugin docstring no longer mentions the "
        "deterministic mode. Restore the audit-safety guidance."
    )
    assert "audit" in lower, (
        "VectorDBRuleEnginePlugin docstring no longer mentions audit "
        "implications. Restore the warning."
    )


def test_vectordb_deterministic_rule_text_is_stable():
    """The deterministic-mode text builder produces the same output for
    the same rule across calls. Determinism is the whole point — pin it."""
    from genie_tooling.context.plugins.rule_engines.vectordb_engine import (
        _deterministic_rule_text,
    )

    rule = {
        "rule_id": "AUDIT_TEST",
        "predicate": "predicate_calculate",
        "description": "Routes calculation requests through the calculator tool.",
        "conditions": [["AudienceProfile.intent", "==", "computation"]],
        "actions": [
            ["C_D", "set", "tool_id", "calculator_tool"],
            ["C_F", "set", "format", "direct_answer"],
        ],
    }
    a = _deterministic_rule_text(rule)
    b = _deterministic_rule_text(rule)
    assert a == b
    # Prefers the explicit description when one is provided.
    assert "calculation requests" in a


def test_vectordb_deterministic_text_falls_back_for_undescribed_rule():
    """A rule with no description should still produce useful, deterministic
    embedding text built from its structured fields."""
    from genie_tooling.context.plugins.rule_engines.vectordb_engine import (
        _deterministic_rule_text,
    )

    rule = {
        "rule_id": "NO_DESC",
        "predicate": "predicate_what",
        "conditions": [["AudienceProfile.expertise", "==", "expert"]],
        "actions": [
            ["C_D", "set", "tool_id", "calculator_tool"],
            ["C_F", "set", "tone", "formal"],
        ],
    }
    text = _deterministic_rule_text(rule)
    assert "NO_DESC" in text
    assert "predicate_what" in text
    assert "AudienceProfile.expertise == expert" in text
    assert "calculator_tool" in text
    # Stable across calls
    assert text == _deterministic_rule_text(rule)


@pytest.mark.asyncio()
async def test_vectordb_default_mode_is_deterministic():
    """Default config = deterministic mode = no LLM call required at indexing."""
    from genie_tooling.context.plugins.rule_engines.vectordb_engine import (
        VectorDBRuleEnginePlugin,
        _deterministic_rule_text,
    )

    plugin = VectorDBRuleEnginePlugin()
    await plugin.setup(config={"rules_path": "/tmp/whatever"})  # path doesn't need to exist
    assert plugin._use_llm_descriptions is False

    rule = {
        "rule_id": "X",
        "predicate": "*",
        "description": "Some description.",
        "conditions": [],
        "actions": [],
    }
    # The genie facade isn't used in deterministic mode; pass None to prove it.
    text = await plugin._rule_to_text(rule, genie=None)  # type: ignore[arg-type]
    assert text == _deterministic_rule_text(rule)


@pytest.mark.asyncio()
async def test_vectordb_llm_mode_calls_genie_llm():
    """When opted-in, the plugin invokes genie.llm.generate to enrich the
    rule text. Verified by mocking the facade and checking the call."""
    from unittest.mock import AsyncMock, MagicMock

    from genie_tooling.context.plugins.rule_engines.vectordb_engine import (
        VectorDBRuleEnginePlugin,
    )

    plugin = VectorDBRuleEnginePlugin()
    await plugin.setup(
        config={"rules_path": "/tmp/whatever", "use_llm_descriptions": True}
    )
    assert plugin._use_llm_descriptions is True

    mock_genie = MagicMock()
    mock_genie.llm = MagicMock()
    mock_genie.llm.generate = AsyncMock(
        return_value={"text": "LLM-enriched description of this rule."}
    )
    rule = {
        "rule_id": "X",
        "predicate": "*",
        "description": "fallback",
        "conditions": [],
        "actions": [],
    }
    text = await plugin._rule_to_text(rule, genie=mock_genie)
    assert text == "LLM-enriched description of this rule."
    mock_genie.llm.generate.assert_awaited_once()


@pytest.mark.asyncio()
async def test_vectordb_llm_mode_falls_back_to_deterministic_on_error():
    """If the LLM call fails in enriched mode, fall back to the
    deterministic text so indexing still completes."""
    from unittest.mock import AsyncMock, MagicMock

    from genie_tooling.context.plugins.rule_engines.vectordb_engine import (
        VectorDBRuleEnginePlugin,
        _deterministic_rule_text,
    )

    plugin = VectorDBRuleEnginePlugin()
    await plugin.setup(
        config={"rules_path": "/tmp/whatever", "use_llm_descriptions": True}
    )

    mock_genie = MagicMock()
    mock_genie.llm = MagicMock()
    mock_genie.llm.generate = AsyncMock(side_effect=RuntimeError("LLM down"))
    rule = {
        "rule_id": "Y",
        "predicate": "*",
        "description": "Anchor text.",
        "conditions": [],
        "actions": [],
    }
    text = await plugin._rule_to_text(rule, genie=mock_genie)
    assert text == _deterministic_rule_text(rule)


@pytest.mark.asyncio()
async def test_vectordb_llm_mode_emits_warning(caplog):
    """Setup must emit a loud warning when LLM-enriched mode is active so
    operators notice if it's accidentally on in production."""
    import logging

    from genie_tooling.context.plugins.rule_engines.vectordb_engine import (
        VectorDBRuleEnginePlugin,
    )

    caplog.set_level(logging.WARNING)
    plugin = VectorDBRuleEnginePlugin()
    await plugin.setup(
        config={"rules_path": "/tmp/whatever", "use_llm_descriptions": True}
    )
    warnings = [
        rec.message
        for rec in caplog.records
        if rec.levelno >= logging.WARNING and "use_llm_descriptions" in rec.message
    ]
    assert warnings, "expected a WARNING log when LLM-enriched mode is active"
    assert any("not deterministic" in w.lower() or "non-determinis" in w.lower() for w in warnings)
