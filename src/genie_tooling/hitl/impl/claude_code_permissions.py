"""ClaudeCodePermissionsPlugin — three-tier (allow/ask/deny) permission model.

A deterministic policy approver modelled on the Claude Code permission system.
Used as the **first link** in a HITLManager chain — for any request that the
policy doesn't decisively allow or deny, it emits ``status="ask_human"`` so
the manager passes the request to the next approver in the chain (typically
a webhook approver or CLI approver).

## Permissions YAML (default path: ``./permissions.yml``)::

    # Optional defaults section
    defaults:
      # What to do when no rule matches.
      on_no_match: ask                # allow / ask / deny

      # Default mapping when no rule matches AND on_no_match is "auto",
      # using the tool's side_effects metadata as the decision driver.
      side_effects_defaults:
        none: allow
        read: allow
        write: ask
        destructive: ask
        unknown: ask

    # First-match-wins rules.
    rules:
      # Allow kubectl reads outright
      - id: ALLOW_KUBECTL_READS
        match:
          tool_id: kubectl_tool_v1
          params_match:
            operation: get             # exact value
        decision: allow
        reason: "kubectl read operations are safe"

      # Block kubectl namespace deletion always
      - id: DENY_NAMESPACE_DELETE
        match:
          tool_id: kubectl_tool_v1
          params_match:
            operation: delete
            resource: namespace
        decision: deny
        reason: "namespace deletion is forbidden"

      # Glob match on tool_id + parameter pattern
      - id: ALLOW_SLACK_POST_TO_ENG
        match:
          tool_id_in: ["slack_tool_v1", "slack:postMessage"]
          params_match:
            channel: "#engineering*"   # fnmatch glob on value
        decision: allow

      # Side-effects-based fall-through: ask for any write or destructive tool
      - id: ASK_FOR_WRITES
        match:
          side_effects_in: ["write", "destructive"]
        decision: ask
        reason: "writes require human review"

## Supported match keys

Top-level ``match`` dict:

* ``tool_id``: str — exact match against ``data_to_approve['tool_id']``
* ``tool_id_in``: list[str] — membership test, supports ``"prefix_*"`` fnmatch glob
* ``side_effects``: str — exact match against tool side-effects classification
  (requires the caller to populate ``data_to_approve['tool_metadata']['side_effects']``,
  done automatically by ``genie.run_command(...)`` and ``ReActAgent``)
* ``side_effects_in``: list[str] — membership
* ``params_match``: dict — sub-match against ``data_to_approve['params']``
  (each key required; value is matched exactly OR as an fnmatch glob if a string
  containing ``*``, ``?`` or ``[``)
* ``user_identity``: dict — sub-match against ``request.context['user_identity']``

Decisions: ``allow`` / ``deny`` / ``ask`` — first match wins.

Session-level overrides (e.g., the user clicked "always allow for this session"
in a wrapper UI) are configurable via ``add_session_allow(session_id, tool_id, params_match)``.
"""
from __future__ import annotations

import asyncio
import fnmatch
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

from genie_tooling.hitl.abc import HumanApprovalRequestPlugin
from genie_tooling.hitl.types import ApprovalRequest, ApprovalResponse

logger = logging.getLogger(__name__)

_DEFAULT_SIDE_EFFECT_DEFAULTS: Dict[str, str] = {
    "none": "allow",
    "read": "allow",
    "write": "ask",
    "destructive": "ask",
    "unknown": "ask",
}


class ClaudeCodePermissionsPlugin(HumanApprovalRequestPlugin):
    plugin_id: str = "claude_code_permissions_v1"
    description: str = (
        "Three-tier (allow/ask/deny) permission model with glob match on "
        "tool_id and parameters. First link in the HITLManager chain — "
        "emits 'ask_human' to delegate to the next approver for anything it "
        "doesn't decide outright."
    )

    _policy_path: Optional[Path]
    _rules: List[Dict[str, Any]]
    _on_no_match: str
    _side_effects_defaults: Dict[str, str]
    _session_allows: Dict[str, List[Dict[str, Any]]]

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        path_str = cfg.get("policy_path", "./permissions.yml")
        self._policy_path = Path(path_str) if path_str else None
        self._session_allows = {}

        # Inline policy wins over file
        inline_doc = cfg.get("policy_inline")
        if inline_doc is not None:
            doc = inline_doc
        elif self._policy_path and self._policy_path.is_file():
            doc = await asyncio.to_thread(self._load_policy_file)
        else:
            if self._policy_path:
                logger.info(
                    f"{self.plugin_id}: policy file {self._policy_path} not found; "
                    "falling back to defaults (on_no_match=ask, write/destructive→ask)."
                )
            doc = {}

        if isinstance(doc, list):
            # Legacy/policy_approval shape (a bare list of rules) — accept it
            self._rules = doc
            self._on_no_match = "ask"
            self._side_effects_defaults = dict(_DEFAULT_SIDE_EFFECT_DEFAULTS)
        elif isinstance(doc, dict):
            self._rules = list(doc.get("rules") or [])
            defaults_dict = doc.get("defaults") or {}
            self._on_no_match = str(defaults_dict.get("on_no_match", "ask")).lower()
            se_def = defaults_dict.get("side_effects_defaults") or {}
            self._side_effects_defaults = dict(_DEFAULT_SIDE_EFFECT_DEFAULTS)
            self._side_effects_defaults.update({str(k): str(v).lower() for k, v in se_def.items()})
        else:
            self._rules = []
            self._on_no_match = "ask"
            self._side_effects_defaults = dict(_DEFAULT_SIDE_EFFECT_DEFAULTS)

        logger.info(
            f"{self.plugin_id}: ready with {len(self._rules)} rule(s); "
            f"on_no_match={self._on_no_match!r}; "
            f"side_effects_defaults={self._side_effects_defaults}"
        )

    def _load_policy_file(self) -> Any:
        try:
            with open(self._policy_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(
                f"{self.plugin_id}: failed to load policy file {self._policy_path}: {e}",
                exc_info=True,
            )
            return {}

    def add_session_allow(
        self,
        session_id: str,
        tool_id: str,
        params_match: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Programmatic 'always allow for this session' analogous to Claude Code's
        per-session pattern. Cleared automatically at session end (caller drops
        the session key)."""
        self._session_allows.setdefault(session_id, []).append(
            {"tool_id": tool_id, "params_match": params_match or {}}
        )

    def clear_session_allows(self, session_id: str) -> None:
        self._session_allows.pop(session_id, None)

    async def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        request_id = request.get("request_id", "")
        now = time.time()
        data = request.get("data_to_approve") or {}
        context = request.get("context") or {}
        user_identity = context.get("user_identity") if isinstance(context, dict) else None
        session_id = context.get("session_id") if isinstance(context, dict) else None

        tool_id = data.get("tool_id", "")
        params = data.get("params") or {}
        tool_metadata = data.get("tool_metadata") or {}
        side_effects = str(tool_metadata.get("side_effects") or "unknown")

        # 1. Session-level overrides
        if session_id and session_id in self._session_allows:
            for entry in self._session_allows[session_id]:
                if entry["tool_id"] == tool_id or fnmatch.fnmatchcase(tool_id, entry["tool_id"]):
                    if self._params_match(entry.get("params_match", {}), params):
                        return ApprovalResponse(
                            request_id=request_id,
                            status="approved",
                            approver_id=f"{self.plugin_id}:session_allow",
                            reason="session-level always-allow matched",
                            timestamp=now,
                        )

        # 2. Tool metadata: explicit requires_approval=False short-circuits to allow
        if tool_metadata.get("requires_approval") is False:
            return ApprovalResponse(
                request_id=request_id,
                status="approved",
                approver_id=f"{self.plugin_id}:tool_explicit_allow",
                reason="tool declared requires_approval=False",
                timestamp=now,
            )

        # 3. Rules
        for idx, rule in enumerate(self._rules):
            if not isinstance(rule, dict):
                continue
            rule_id = str(rule.get("id", f"rule_{idx}"))
            if self._rule_matches(rule.get("match") or {}, tool_id, params, side_effects, user_identity):
                decision = str(rule.get("decision", "")).lower()
                reason = rule.get("reason") or f"policy {rule_id} matched"
                if decision == "allow":
                    return ApprovalResponse(
                        request_id=request_id,
                        status="approved",
                        approver_id=f"{self.plugin_id}:{rule_id}",
                        reason=reason,
                        timestamp=now,
                    )
                if decision == "deny":
                    return ApprovalResponse(
                        request_id=request_id,
                        status="denied",
                        approver_id=f"{self.plugin_id}:{rule_id}",
                        reason=reason,
                        timestamp=now,
                    )
                if decision == "ask":
                    return ApprovalResponse(
                        request_id=request_id,
                        status="ask_human",
                        approver_id=f"{self.plugin_id}:{rule_id}",
                        reason=reason,
                        timestamp=now,
                    )
                logger.warning(
                    f"{self.plugin_id}: rule {rule_id} has invalid decision={decision!r}; "
                    "expected allow/ask/deny. Skipping."
                )

        # 4. Tool metadata: explicit requires_approval=True short-circuits to ask
        if tool_metadata.get("requires_approval") is True:
            return ApprovalResponse(
                request_id=request_id,
                status="ask_human",
                approver_id=f"{self.plugin_id}:tool_explicit_ask",
                reason="tool declared requires_approval=True",
                timestamp=now,
            )

        # 5. Side-effects-driven default
        se_decision = self._side_effects_defaults.get(side_effects, "ask")
        if self._on_no_match == "allow" or se_decision == "allow":
            return ApprovalResponse(
                request_id=request_id,
                status="approved",
                approver_id=f"{self.plugin_id}:default_allow",
                reason=f"no rule matched; side_effects={side_effects} → allow",
                timestamp=now,
            )
        if self._on_no_match == "deny" or se_decision == "deny":
            return ApprovalResponse(
                request_id=request_id,
                status="denied",
                approver_id=f"{self.plugin_id}:default_deny",
                reason=f"no rule matched; side_effects={side_effects} → deny",
                timestamp=now,
            )
        # Fallback: ask
        return ApprovalResponse(
            request_id=request_id,
            status="ask_human",
            approver_id=f"{self.plugin_id}:default_ask",
            reason=f"no rule matched; side_effects={side_effects} → ask",
            timestamp=now,
        )

    def _rule_matches(
        self,
        match: Mapping[str, Any],
        tool_id: str,
        params: Mapping[str, Any],
        side_effects: str,
        user_identity: Optional[Mapping[str, Any]],
    ) -> bool:
        if not match:
            return True

        if "tool_id" in match:
            if tool_id != match["tool_id"]:
                return False

        if "tool_id_in" in match:
            allowed = match["tool_id_in"] or []
            if not any(fnmatch.fnmatchcase(tool_id, pat) for pat in allowed):
                return False

        if "side_effects" in match:
            if side_effects != match["side_effects"]:
                return False

        if "side_effects_in" in match:
            allowed_se = match["side_effects_in"] or []
            if side_effects not in allowed_se:
                return False

        if "params_match" in match:
            if not self._params_match(match["params_match"] or {}, params):
                return False

        if "user_identity" in match:
            ui_match = match["user_identity"] or {}
            ui = user_identity or {}
            if "role" in ui_match and ui.get("role") != ui_match["role"]:
                return False
            if "role_in" in ui_match:
                if ui.get("role") not in (ui_match["role_in"] or []):
                    return False

        return True

    @staticmethod
    def _params_match(spec: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
        """Each key in ``spec`` must match the corresponding key in ``actual``.

        - If the spec value is a string containing glob characters (``*``, ``?``, ``[``),
          match via ``fnmatch.fnmatchcase``.
        - Otherwise, match by equality.
        - A spec key not present in ``actual`` is a miss.
        """
        for key, expected in spec.items():
            if key not in actual:
                return False
            actual_val = actual[key]
            if isinstance(expected, str) and any(c in expected for c in "*?["):
                if not isinstance(actual_val, str) or not fnmatch.fnmatchcase(actual_val, expected):
                    return False
            else:
                if actual_val != expected:
                    return False
        return True

    async def teardown(self) -> None:
        self._session_allows.clear()
