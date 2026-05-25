"""
Non-mock validation of the bundled cqs rules. The other tests in this directory
use handcrafted in-memory rules with mocked plugins; this test loads the real
bundled YAML files and verifies that:

1. They parse without errors.
2. There are no duplicate rule_ids.
3. Every derivation_strategy_id, command_processor_id, and tool_id action
   references an actually-registered plugin.

This catches the class of bug where a sample rule references a plugin that
doesn't exist in the codebase (e.g. `agentic_derivation_v1` vs. the real
`generic_agent_derivation_v1`).
"""
from __future__ import annotations

import importlib.metadata
import importlib.resources
from collections import Counter
from pathlib import Path
from typing import Set

import pytest
from genie_tooling.context.plugins.rule_engines.filesystem_engine import (
    FileSystemRuleEnginePlugin,
)


def _bundled_rules_dir() -> Path:
    return Path(str(importlib.resources.files("genie_tooling.context") / "rules"))


def _registered_plugin_ids() -> Set[str]:
    """All plugin IDs registered in pyproject.toml entry-points."""
    eps = importlib.metadata.entry_points()
    return {ep.name for ep in eps.select(group="genie_tooling.plugins")}


@pytest.mark.asyncio()
async def test_bundled_rules_parse() -> None:
    rules_dir = _bundled_rules_dir()
    assert rules_dir.is_dir(), f"Bundled rules dir not found: {rules_dir}"

    engine = FileSystemRuleEnginePlugin()
    await engine.setup(config={"rules_path": str(rules_dir)})
    loaded = await engine.load_rules()

    assert loaded, "FileSystemRuleEnginePlugin.load_rules returned False"
    assert engine._rules, "No bundled rules were loaded"


@pytest.mark.asyncio()
async def test_bundled_rules_have_unique_ids() -> None:
    engine = FileSystemRuleEnginePlugin()
    await engine.setup(config={"rules_path": str(_bundled_rules_dir())})
    await engine.load_rules()

    counts = Counter(r.get("rule_id") for r in engine._rules)
    duplicates = {rid: n for rid, n in counts.items() if n > 1}
    assert not duplicates, (
        f"Bundled rules have duplicate rule_ids — load order is non-deterministic. "
        f"Duplicates: {duplicates}"
    )


@pytest.mark.asyncio()
async def test_bundled_rules_reference_registered_plugins_and_tools() -> None:
    """
    Every derivation_strategy_id / command_processor_id / tool_id mentioned in a
    rule's actions must resolve to a registered plugin. (tool_id is also checked
    against the plugin registry — tools register under genie_tooling.plugins too.)
    """
    engine = FileSystemRuleEnginePlugin()
    await engine.setup(config={"rules_path": str(_bundled_rules_dir())})
    await engine.load_rules()

    registered = _registered_plugin_ids()
    REFERENCE_KEYS = {"derivation_strategy_id", "command_processor_id", "tool_id"}

    bad_refs: list[tuple[str, str, str]] = []  # (rule_id, key, value)
    for rule in engine._rules:
        rule_id = rule.get("rule_id", "<no id>")
        for action in rule.get("actions", []):
            if len(action) != 4:
                continue
            _target, op, key, value = action
            if op != "set" or key not in REFERENCE_KEYS:
                continue
            if not isinstance(value, str):
                continue
            if value not in registered:
                bad_refs.append((rule_id, key, value))

    assert not bad_refs, (
        "Bundled rules reference plugins/tools that aren't registered:\n  "
        + "\n  ".join(f"{rid}: {k}={v!r}" for rid, k, v in bad_refs)
    )


@pytest.mark.asyncio()
async def test_bundled_rules_use_correct_action_key_names() -> None:
    """
    Catches typos like `derivation_strategy` (missing `_id`) that ContextManager
    silently ignores. Whitelist the keys the manager and derivation plugins
    actually read.
    """
    KNOWN_KEYS = {
        # Read by ContextManager._derivation_step
        "derivation_strategy_id",
        # Read by GenericAgentDerivationPlugin.derive
        "command_processor_id",
        # Read by GenericToolDerivationPlugin.derive
        "tool_id",
        "params",
        # Read by LlmPromptFormulationPlugin.formulate
        "prompt_template_id",
        # Formulation knobs — free-form, used by prompts
        "tone",
        "verbosity",
        "format",
        "empathy_level",
    }

    engine = FileSystemRuleEnginePlugin()
    await engine.setup(config={"rules_path": str(_bundled_rules_dir())})
    await engine.load_rules()

    unknown: list[tuple[str, str]] = []
    for rule in engine._rules:
        rule_id = rule.get("rule_id", "<no id>")
        for action in rule.get("actions", []):
            if len(action) != 4:
                continue
            _target, _op, key, _value = action
            if key not in KNOWN_KEYS:
                unknown.append((rule_id, key))

    assert not unknown, (
        "Bundled rules use action keys the engine doesn't read (likely typos):\n  "
        + "\n  ".join(f"{rid}: {k!r}" for rid, k in unknown)
    )
