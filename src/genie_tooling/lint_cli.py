"""Phase 6C.11 — ``genie-lint`` policy-as-code linter.

CI-runnable equivalent of the runtime ``RuleValidationError`` check. Lets
governance/compliance teams gate YAML rule changes on `genie-lint
src/genie_tooling/context/rules/ --strict` BEFORE merging.

Lints two kinds of YAML files:

* λ-CQS rule files (the same shape ``FileSystemRuleEnginePlugin`` loads):
  cross-checks plugin references + asserts rule_id uniqueness.
* MCP overlay files: validates that ``side_effects`` values are within the
  allowed taxonomy and warns about implausible combinations.

Exit codes:
  0 — clean
  1 — one or more errors
  2 — argument / I/O error
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ALLOWED_SIDE_EFFECTS = {"none", "read", "write", "destructive", "unknown"}


def lint_rules_dir(rules_dir: Path) -> List[str]:
    """Apply ``FileSystemRuleEnginePlugin._validate_loaded_rules`` semantics
    against an on-disk directory of YAML files. Returns a flat list of
    error strings (empty if clean)."""
    import importlib.metadata
    import yaml

    errors: List[str] = []
    rules: List[Dict[str, Any]] = []

    if not rules_dir.is_dir():
        return [f"{rules_dir}: not a directory"]

    for yaml_path in sorted(rules_dir.glob("*.yml")) + sorted(rules_dir.glob("*.yaml")):
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                doc = yaml.safe_load(f)
        except Exception as e:
            errors.append(f"{yaml_path}: failed to parse YAML: {e}")
            continue
        if doc is None:
            continue
        if isinstance(doc, dict):
            rules.append(doc)
        elif isinstance(doc, list):
            rules.extend(d for d in doc if isinstance(d, dict))
        else:
            errors.append(f"{yaml_path}: top-level must be a dict or list of dicts; got {type(doc).__name__}")

    # Cross-check plugin references against entry points.
    eps = importlib.metadata.entry_points()
    registered = {ep.name for ep in eps.select(group="genie_tooling.plugins")}
    _PLUGIN_REFERENCE_KEYS = {
        "derivation_strategy_id",
        "command_processor_id",
        "tool_id",
        "formulation_strategy_id",
        "prompt_template_id",
    }

    from collections import Counter
    id_counts = Counter(r.get("rule_id") for r in rules if r.get("rule_id"))
    for rid, count in id_counts.items():
        if count > 1:
            errors.append(f"rule_id {rid!r} declared {count} times; must be unique.")

    for rule in rules:
        rule_id = rule.get("rule_id", "<no id>")
        actions = rule.get("actions", [])
        if not isinstance(actions, list):
            errors.append(f"rule_id {rule_id!r}: actions must be a list")
            continue
        for i, action in enumerate(actions):
            if not isinstance(action, list) or len(action) != 4:
                errors.append(f"rule_id {rule_id!r} action {i}: must be [target, op, key, value] 4-tuple")
                continue
            _target, _op, key, value = action
            if key in _PLUGIN_REFERENCE_KEYS and isinstance(value, str):
                if value not in registered:
                    errors.append(
                        f"rule_id {rule_id!r}: {key}={value!r} is not a registered plugin."
                    )

    return errors


def lint_overlay_dir(overlay_dir: Path) -> List[str]:
    """Validate MCP overlay YAML files: each overlay entry must have a
    valid side_effects value (or omit the key)."""
    import yaml

    errors: List[str] = []
    if not overlay_dir.is_dir():
        return [f"{overlay_dir}: not a directory"]

    for yaml_path in sorted(overlay_dir.glob("*.yml")):
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                doc = yaml.safe_load(f) or {}
        except Exception as e:
            errors.append(f"{yaml_path}: failed to parse YAML: {e}")
            continue
        overlays = doc.get("overlays", doc) if isinstance(doc, dict) else None
        if not isinstance(overlays, dict):
            errors.append(f"{yaml_path}: expected an 'overlays:' map or flat dict at top level")
            continue
        for tool_name, overlay in overlays.items():
            if not isinstance(overlay, dict):
                errors.append(f"{yaml_path}: overlay {tool_name!r} is not a dict")
                continue
            se = overlay.get("side_effects")
            if se is not None and se not in ALLOWED_SIDE_EFFECTS:
                errors.append(
                    f"{yaml_path}: overlay {tool_name!r}: side_effects={se!r} is not one of "
                    f"{sorted(ALLOWED_SIDE_EFFECTS)}"
                )
            # Implausible combinations
            if se == "destructive" and overlay.get("requires_approval") is False:
                errors.append(
                    f"{yaml_path}: overlay {tool_name!r}: destructive tool with "
                    f"requires_approval=false is dangerous; require approval or downgrade side_effects."
                )
            if se in ("none", "read") and overlay.get("requires_approval") is True:
                # Not an error — could be intentional — but flag as warning.
                errors.append(
                    f"{yaml_path}: WARNING overlay {tool_name!r}: requires_approval=true on a "
                    f"side_effects={se} tool is unusual."
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="genie-lint",
        description="Lint Genie YAML configs (λ-CQS rules + MCP overlays).",
    )
    parser.add_argument("targets", nargs="+", type=Path, help="Directories or files to lint.")
    parser.add_argument(
        "--kind",
        choices=["auto", "rules", "overlays"],
        default="auto",
        help="Force a specific lint kind, or auto-detect from path.",
    )
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors (default already strict).")
    args = parser.parse_args()

    all_errors: List[str] = []
    for target in args.targets:
        kind = args.kind
        if kind == "auto":
            kind = "overlays" if "overlay" in str(target).lower() else "rules"
        if kind == "rules":
            all_errors.extend(lint_rules_dir(target))
        else:
            all_errors.extend(lint_overlay_dir(target))

    if all_errors:
        print(f"genie-lint: {len(all_errors)} issue(s) found:", file=sys.stderr)
        for err in all_errors:
            print(f"  - {err}", file=sys.stderr)
        # Warnings are folded into errors when --strict (default behaviour)
        return 1

    print(f"genie-lint: OK ({len(args.targets)} target(s) lint clean).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
