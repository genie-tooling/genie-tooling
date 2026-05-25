"""F13 — smoke tests for the console-script entry points."""
from __future__ import annotations

import sys

# ---- genie-lint ----


def test_genie_lint_main_importable():
    """The console-script entry must import cleanly."""
    from genie_tooling.lint_cli import main
    assert callable(main)


def test_genie_lint_clean_rules_dir(tmp_path, monkeypatch, capsys):
    """A directory with a valid rule file lints clean."""
    from genie_tooling.lint_cli import main

    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    (rules_dir / "ok.yml").write_text(
        """
rule_id: TEST_OK
predicate: predicate_test
priority: 1
conditions: []
actions: []
"""
    )
    monkeypatch.setattr(sys, "argv", ["genie-lint", str(rules_dir), "--kind", "rules"])
    rc = main()
    assert rc == 0


def test_genie_lint_catches_unknown_plugin_reference(tmp_path, monkeypatch):
    """A rule referencing a nonexistent plugin should fail lint."""
    from genie_tooling.lint_cli import main

    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    (rules_dir / "bad.yml").write_text(
        """
rule_id: BAD_RULE
predicate: x
priority: 1
conditions: []
actions:
  - ["C_D", "set", "derivation_strategy_id", "nonexistent_strategy_v999"]
"""
    )
    monkeypatch.setattr(sys, "argv", ["genie-lint", str(rules_dir), "--kind", "rules"])
    rc = main()
    assert rc == 1


def test_genie_lint_overlays_clean(tmp_path, monkeypatch):
    """A valid MCP overlay file lints clean."""
    from genie_tooling.lint_cli import main

    ovr = tmp_path / "overlays"
    ovr.mkdir()
    (ovr / "ok.yml").write_text(
        """
overlays:
  list_things:
    side_effects: read
    idempotent: true
  create_thing:
    side_effects: write
    requires_approval: true
"""
    )
    monkeypatch.setattr(sys, "argv", ["genie-lint", str(ovr), "--kind", "overlays"])
    rc = main()
    assert rc == 0


def test_genie_lint_overlays_catches_bad_side_effects(tmp_path, monkeypatch):
    from genie_tooling.lint_cli import main

    ovr = tmp_path / "overlays"
    ovr.mkdir()
    (ovr / "bad.yml").write_text(
        """
overlays:
  thing:
    side_effects: catastrophic   # not in allowed set
"""
    )
    monkeypatch.setattr(sys, "argv", ["genie-lint", str(ovr), "--kind", "overlays"])
    rc = main()
    assert rc == 1


def test_genie_lint_destructive_no_approval_is_an_error(tmp_path, monkeypatch):
    """A destructive op with requires_approval=False should fail lint."""
    from genie_tooling.lint_cli import main

    ovr = tmp_path / "overlays"
    ovr.mkdir()
    (ovr / "dangerous.yml").write_text(
        """
overlays:
  delete_all:
    side_effects: destructive
    requires_approval: false
"""
    )
    monkeypatch.setattr(sys, "argv", ["genie-lint", str(ovr), "--kind", "overlays"])
    rc = main()
    assert rc == 1


# ---- genie-mcp-serve ----


def test_genie_mcp_serve_main_importable():
    from genie_tooling.mcp_server_cli import main
    assert callable(main)


def test_genie_mcp_serve_missing_config(monkeypatch):
    """Pointing at a nonexistent config returns a clear error exit."""
    from genie_tooling.mcp_server_cli import main

    monkeypatch.setattr(sys, "argv", ["genie-mcp-serve", "--config", "/nonexistent/path.yml"])
    rc = main()
    assert rc == 2


def test_genie_mcp_serve_malformed_yaml(tmp_path, monkeypatch):
    cfg = tmp_path / "bad.yml"
    cfg.write_text("not: valid: yaml: :::\n")
    from genie_tooling.mcp_server_cli import main

    monkeypatch.setattr(sys, "argv", ["genie-mcp-serve", "--config", str(cfg)])
    rc = main()
    assert rc == 2


def test_console_scripts_registered():
    """Both console scripts should be discoverable from importlib.metadata."""
    import importlib.metadata as md
    scripts = {ep.name for ep in md.entry_points().select(group="console_scripts")}
    assert "genie-lint" in scripts
    assert "genie-mcp-serve" in scripts
