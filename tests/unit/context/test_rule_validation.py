"""Unit tests for rule-load validation (A2).

The validator runs at load_rules time and fails loud on:
  * duplicate rule_ids (non-deterministic load order)
  * malformed action tuples (wrong arity or non-list)
  * references to unknown plugins (derivation_strategy_id, command_processor_id,
    tool_id pointing at something not registered)

Corporate use: broken rules must fail at startup, not at first match.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from genie_tooling.context.plugins.rule_engines.filesystem_engine import (
    FileSystemRuleEnginePlugin,
    RuleValidationError,
)


def _write_rule(rules_dir: Path, rule: dict, filename: str | None = None) -> None:
    name = filename or f"{rule['rule_id']}.yml"
    (rules_dir / name).write_text(yaml.dump(rule))


@pytest.mark.asyncio
async def test_valid_rule_set_passes_strict_validation(tmp_path: Path):
    """A rule that references only real registered plugins loads cleanly."""
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    _write_rule(
        rules_dir,
        {
            "rule_id": "VALID",
            "predicate": "*",
            "priority": 1,
            "conditions": [],
            "actions": [
                ["C_D", "set", "derivation_strategy_id", "generic_tool_derivation_v1"],
                ["C_D", "set", "tool_id", "calculator_tool"],
            ],
        },
    )
    engine = FileSystemRuleEnginePlugin()
    await engine.setup(config={"rules_path": str(rules_dir)})
    ok = await engine.load_rules()
    assert ok is True
    assert len(engine._rules) == 1


@pytest.mark.asyncio
async def test_unknown_derivation_strategy_fails_loud(tmp_path: Path):
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    _write_rule(
        rules_dir,
        {
            "rule_id": "BAD_DERIVATION",
            "predicate": "*",
            "priority": 1,
            "conditions": [],
            "actions": [
                [
                    "C_D",
                    "set",
                    "derivation_strategy_id",
                    "agentic_derivation_v1",  # Pre-fix typo
                ],
            ],
        },
    )
    engine = FileSystemRuleEnginePlugin()
    await engine.setup(config={"rules_path": str(rules_dir)})
    with pytest.raises(RuleValidationError) as exc_info:
        await engine.load_rules()
    msg = str(exc_info.value)
    assert "BAD_DERIVATION" in msg
    assert "agentic_derivation_v1" in msg
    assert "derivation_strategy_id" in msg


@pytest.mark.asyncio
async def test_unknown_tool_fails_loud(tmp_path: Path):
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    _write_rule(
        rules_dir,
        {
            "rule_id": "BAD_TOOL",
            "predicate": "*",
            "priority": 1,
            "conditions": [],
            "actions": [
                ["C_D", "set", "derivation_strategy_id", "generic_tool_derivation_v1"],
                ["C_D", "set", "tool_id", "fake_lookup_tool"],
            ],
        },
    )
    engine = FileSystemRuleEnginePlugin()
    await engine.setup(config={"rules_path": str(rules_dir)})
    with pytest.raises(RuleValidationError) as exc_info:
        await engine.load_rules()
    assert "BAD_TOOL" in str(exc_info.value)
    assert "fake_lookup_tool" in str(exc_info.value)


@pytest.mark.asyncio
async def test_duplicate_rule_id_fails_loud(tmp_path: Path):
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    rule_template = {
        "predicate": "*",
        "priority": 1,
        "conditions": [],
        "actions": [
            ["C_D", "set", "derivation_strategy_id", "generic_tool_derivation_v1"],
        ],
    }
    _write_rule(rules_dir, {"rule_id": "DUPED", **rule_template}, "a.yml")
    _write_rule(rules_dir, {"rule_id": "DUPED", **rule_template}, "b.yml")
    engine = FileSystemRuleEnginePlugin()
    await engine.setup(config={"rules_path": str(rules_dir)})
    with pytest.raises(RuleValidationError) as exc_info:
        await engine.load_rules()
    msg = str(exc_info.value)
    assert "DUPED" in msg
    assert "declared 2 times" in msg


@pytest.mark.asyncio
async def test_malformed_action_arity_fails_loud(tmp_path: Path):
    """Action tuples must be 4 elements: [target, op, key, value]."""
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    _write_rule(
        rules_dir,
        {
            "rule_id": "MALFORMED",
            "predicate": "*",
            "priority": 1,
            "conditions": [],
            "actions": [
                ["C_D", "set", "tool_id"],  # missing value -> arity 3
            ],
        },
    )
    engine = FileSystemRuleEnginePlugin()
    await engine.setup(config={"rules_path": str(rules_dir)})
    with pytest.raises(RuleValidationError) as exc_info:
        await engine.load_rules()
    assert "malformed action" in str(exc_info.value)


@pytest.mark.asyncio
async def test_validation_reports_all_errors_at_once(tmp_path: Path):
    """One pass should surface every defect — operators want a punch list,
    not a slow drip of one error per restart."""
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    _write_rule(
        rules_dir,
        {
            "rule_id": "BAD_1",
            "predicate": "*",
            "priority": 1,
            "conditions": [],
            "actions": [
                ["C_D", "set", "derivation_strategy_id", "nope_v1"],
            ],
        },
        "a.yml",
    )
    _write_rule(
        rules_dir,
        {
            "rule_id": "BAD_2",
            "predicate": "*",
            "priority": 1,
            "conditions": [],
            "actions": [
                ["C_D", "set", "tool_id", "also_nope_v1"],
            ],
        },
        "b.yml",
    )
    engine = FileSystemRuleEnginePlugin()
    await engine.setup(config={"rules_path": str(rules_dir)})
    with pytest.raises(RuleValidationError) as exc_info:
        await engine.load_rules()
    msg = str(exc_info.value)
    assert "BAD_1" in msg
    assert "nope_v1" in msg
    assert "BAD_2" in msg
    assert "also_nope_v1" in msg
    # The error count in the header should reflect both defects.
    assert "2 error" in msg


@pytest.mark.asyncio
async def test_strict_validation_false_loads_broken_rules_with_warning(tmp_path: Path):
    """For the legitimate case where rule references resolve later (e.g. a
    bootstrap registers more plugins), users can opt out. Test that the
    rules still load — runtime evaluation will reject them at first match
    instead, which is the pre-A2 behavior."""
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    _write_rule(
        rules_dir,
        {
            "rule_id": "BROKEN_BUT_OPT_OUT",
            "predicate": "*",
            "priority": 1,
            "conditions": [],
            "actions": [
                ["C_D", "set", "derivation_strategy_id", "future_plugin_v1"],
            ],
        },
    )
    engine = FileSystemRuleEnginePlugin()
    await engine.setup(
        config={"rules_path": str(rules_dir), "strict_validation": False}
    )
    ok = await engine.load_rules()
    assert ok is True
    assert len(engine._rules) == 1


@pytest.mark.asyncio
async def test_bundled_rules_pass_validation():
    """The bundled rule set must pass strict validation — this guards the
    P0 'fix bundled rules' commitment in Phase 4A4."""
    engine = FileSystemRuleEnginePlugin()
    await engine.setup(config=None)  # use bundled default
    await engine.load_rules()  # raises on failure
    assert engine._rules
