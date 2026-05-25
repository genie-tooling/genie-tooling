import asyncio
import importlib.metadata
import importlib.resources
import logging
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import yaml

from genie_tooling.context.protocols import RuleEnginePlugin
from genie_tooling.context.types import RuleObject

if TYPE_CHECKING:
    from genie_tooling.genie import Genie

logger = logging.getLogger(__name__)


class RuleValidationError(RuntimeError):
    """Raised at rule-load time when one or more rules reference plugins,
    processors, or tools that aren't registered, or when the rule set is
    structurally broken (duplicate IDs, malformed actions). Audit/governance
    teams need broken rules to fail loud at startup, not silently at first
    match.
    """


# Action keys whose value must resolve to a registered plugin ID. The
# validator only checks these — rule authors are free to introduce new
# C_F / C_D keys without code changes.
_PLUGIN_REFERENCE_KEYS = frozenset(
    {"derivation_strategy_id", "command_processor_id", "tool_id"}
)


def _default_rules_path() -> Path:
    """Resolve the bundled rules directory shipped inside the installed package.

    The legacy default was `Path('./context_rules')` (CWD-relative), which only
    worked when the user happened to launch from a directory containing a
    rules dir. Now we prefer the bundled rules; users can still override via
    config.
    """
    return Path(str(importlib.resources.files("genie_tooling.context") / "rules"))


class FileSystemRuleEnginePlugin(RuleEnginePlugin):
    """A deterministic rule engine that loads rules from YAML files on disk."""

    plugin_id: str = "filesystem_rule_engine_v1"
    description: str = "Loads and evaluates context rules from local YAML/JSON files."

    _rules: List[RuleObject]
    _rules_path: Path

    async def setup(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        path_override = cfg.get("rules_path")
        self._rules_path = Path(path_override) if path_override else _default_rules_path()
        # strict_validation=True (default) fails loud at load_rules if any rule
        # references an unknown plugin/processor/tool, or if rule_ids collide.
        # Corporate audit needs broken rules to fail at startup, not at first
        # match. Set False for the rare case where the rule set legitimately
        # references plugins registered by a bootstrap that hasn't run yet.
        self._strict_validation = bool(cfg.get("strict_validation", True))
        self._rules = []

    @staticmethod
    def _read_rule_file_sync(file_path: Path) -> Any:
        """Synchronous helper; called via asyncio.to_thread from load_rules."""
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    async def load_rules(self, genie: Optional["Genie"] = None) -> bool:
        # genie is accepted to satisfy the RuleEnginePlugin protocol but
        # FileSystem doesn't need it; loading is pure filesystem I/O.
        is_dir = await asyncio.to_thread(self._rules_path.is_dir)
        if not is_dir:
            logger.error(
                f"[{self.plugin_id}] Rules path '{self._rules_path}' is not a valid directory."
            )
            return False

        rule_paths = await asyncio.to_thread(lambda: list(self._rules_path.glob("*.yml")))
        loaded_rules = []
        for file_path in rule_paths:
            try:
                rule_data = await asyncio.to_thread(self._read_rule_file_sync, file_path)
                if isinstance(rule_data, dict):
                    if "rule_id" in rule_data and "actions" in rule_data:
                        loaded_rules.append(rule_data)
                    else:
                        logger.warning(f"Skipping malformed rule file: {file_path}")
            except Exception as e:
                logger.error(
                    f"[{self.plugin_id}] Failed to load or parse rule file {file_path}: {e}"
                )

        self._rules = sorted(loaded_rules, key=lambda r: r.get("priority", 999))
        logger.info(
            f"[{self.plugin_id}] Loaded and sorted {len(self._rules)} rules from '{self._rules_path}'."
        )

        if self._strict_validation:
            self._validate_loaded_rules()
        return True

    async def reload_rules(self, genie: Optional["Genie"] = None) -> bool:
        """Re-read the rules directory and atomically swap the in-memory
        rule set (C2). Governance/compliance teams update YAML files in
        version control; this lets the engine pick those up without a
        process restart.

        Failures during reload (including A2 validation errors) leave the
        existing rule set unchanged. Returns True on a successful swap,
        False otherwise.
        """
        previous_rules = self._rules
        try:
            ok = await self.load_rules(genie=genie)
            if not ok:
                # Restore previous state on a non-exception failure
                self._rules = previous_rules
                return False
            logger.info(
                f"[{self.plugin_id}] Reload complete; "
                f"{len(self._rules)} rules now active."
            )
            return True
        except Exception as e:
            self._rules = previous_rules
            logger.error(
                f"[{self.plugin_id}] Reload failed; keeping previous {len(previous_rules)} rules. Error: {e}",
                exc_info=True,
            )
            return False

    @staticmethod
    def _registered_plugin_ids() -> Set[str]:
        """All plugin IDs registered via the `genie_tooling.plugins`
        entry-point group. The canonical source of "what's available" to a
        running framework."""
        eps = importlib.metadata.entry_points()
        return {ep.name for ep in eps.select(group="genie_tooling.plugins")}

    def _validate_loaded_rules(self) -> None:
        """Cross-check every rule action's reference value against the
        registered plugin set, and assert rule_id uniqueness. Raises
        RuleValidationError with a list of all defects (not just the first)
        so a corporate operator can fix them in one pass.
        """
        errors: List[str] = []

        # 1. Duplicate rule_id check — non-deterministic load order if two
        # files declare the same rule_id with different contents.
        id_counts = Counter(r.get("rule_id") for r in self._rules)
        for rule_id, count in id_counts.items():
            if count > 1:
                errors.append(
                    f"rule_id {rule_id!r} declared {count} times; "
                    "load order is non-deterministic across files. "
                    "Each rule_id must be unique."
                )

        # 2. Plugin reference resolution — every action whose key looks like
        # a plugin ID reference must resolve to something registered.
        registered = self._registered_plugin_ids()
        for rule in self._rules:
            rule_id = rule.get("rule_id", "<no id>")
            actions = rule.get("actions", [])
            if not isinstance(actions, list):
                errors.append(
                    f"rule_id {rule_id!r}: actions must be a list, got {type(actions).__name__}"
                )
                continue
            for i, action in enumerate(actions):
                if not isinstance(action, list) or len(action) != 4:
                    errors.append(
                        f"rule_id {rule_id!r}, action {i}: malformed action "
                        f"{action!r}; expected [target, op, key, value] 4-tuple"
                    )
                    continue
                _target, _op, key, value = action
                if key in _PLUGIN_REFERENCE_KEYS and isinstance(value, str):
                    if value not in registered:
                        errors.append(
                            f"rule_id {rule_id!r}: {key}={value!r} is not a "
                            f"registered plugin. Available: see "
                            f"`importlib.metadata.entry_points(group='genie_tooling.plugins')`."
                        )

        if errors:
            joined = "\n  - ".join(errors)
            raise RuleValidationError(
                f"Rule validation failed with {len(errors)} error(s):\n  - {joined}"
            )

    def _evaluate_condition(
        self, actual_value: Any, op: str, expected_value: Any
    ) -> bool:
        try:
            if op == "==": return actual_value == expected_value
            if op == "!=": return actual_value != expected_value
            if op == ">": return float(actual_value) > float(expected_value)
            if op == "<": return float(actual_value) < float(expected_value)
            if op == ">=": return float(actual_value) >= float(expected_value)
            if op == "<=": return float(actual_value) <= float(expected_value)
            # Natural reading: `["state", "in", ["Stressed", "Agitated"]]`
            # means "is the state value IN this list of options". So the
            # left operand (actual) is the item, the right (expected) is
            # the container.
            if op == "in": return actual_value in expected_value
            if op == "contains": return str(expected_value) in str(actual_value)

        except (ValueError, TypeError):
            return False
        return False

    async def evaluate(
        self,
        inferred_context: Dict[str, Any],
        query_predicate: str,
        genie: Optional["Genie"] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        # genie unused; deterministic evaluation against in-memory rules.
        for rule in self._rules:
            # Check predicate
            if rule.get("predicate") != query_predicate and rule.get("predicate") != "*":
                continue

            # Check all conditions for this rule
            all_conditions_met = True
            for key_path, op, expected_value in rule.get("conditions", []):
                actual_value = inferred_context
                try:
                    for key in key_path.split("."):
                        actual_value = actual_value[key]
                except (KeyError, TypeError):
                    all_conditions_met = False
                    break

                if not self._evaluate_condition(actual_value, op, expected_value):
                    all_conditions_met = False
                    break

            if all_conditions_met:
                logger.info(f"Rule evaluation matched: '{rule.get('rule_id')}'.")
                return [(rule, 1.0)]

        logger.info("No deterministic rule matched the context.")
        return []
