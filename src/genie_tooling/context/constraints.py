"""Formulation-constraint translator (the C_F policy interface).

The cqs aggregator builds a `C_F` dict from rule actions like
`["C_F", "set", "tone", "calm and reassuring"]`. Historically that dict was
passed to the formulation plugin under the key ``formulation_constraints``
but no bundled prompt template ever read it — meaning every tone, verbosity,
empathy, and redaction directive set by a rule was load-bearing nothing.

For a corporate harness the constraint dict IS the policy interface: an
auditor needs to read a YAML rule and know exactly how the LLM was instructed
to behave. This module translates the structured constraint dict into a
single natural-language instruction block that gets prepended to the
formulation prompt, so the constraints demonstrably reach the model.

Translation is deterministic — same constraint dict in, same instruction
text out. That's a hard requirement: the rendered instruction text appears
in the audit record (`DecisionRecord.formulation_constraints_text`), and
audit teams must be able to reproduce it from the rule alone.

Known constraints are rendered with explicit templated phrasing. Unknown keys
fall through to a generic ``"Apply this guideline: <key> = <value>."`` line
so a rule author can introduce a new key without code changes — at the cost
of less precise LLM steering until a known-key entry is added here.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

# Reserved keys whose semantics are routing/wiring, not LLM-behavior steering.
# These ride alongside C_F constraints in the same dict but should never be
# rendered as instructions. (E.g. ``prompt_template_id`` controls *which*
# template gets rendered, not how the model should respond.)
_NON_BEHAVIORAL_KEYS = frozenset(
    {
        "prompt_template_id",
        # Defensive: derivation-side keys occasionally leak into C_F if a rule
        # author writes the wrong target. Don't accidentally instruct the LLM
        # to "apply derivation_strategy_id = ...".
        "derivation_strategy_id",
        "command_processor_id",
        "tool_id",
        "params",
    }
)


def _render_tone(value: Any) -> str:
    return f"Use a {value} tone in your response."


def _render_verbosity(value: Any) -> str:
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in ("low", "concise", "short", "brief"):
            return "Keep your response concise — one or two sentences."
        if normalised in ("moderate", "medium", "balanced"):
            return "Keep your response of moderate length — three to five sentences."
        if normalised in ("high", "detailed", "long", "verbose"):
            return "Provide a detailed, thorough response."
    return f"Adjust your response verbosity to: {value}."


def _render_format(value: Any) -> str:
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised == "direct_answer":
            return "Provide the direct answer only, with no preamble or commentary."
        if normalised in ("bullet_list", "bulleted", "bullets"):
            return "Structure your response as a bulleted list."
        if normalised in ("numbered_list", "numbered"):
            return "Structure your response as a numbered list."
        if normalised == "json":
            return "Return your response as valid JSON only."
    return f"Format your response as: {value}."


def _render_empathy(value: Any) -> str:
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in ("high", "strong"):
            return "Show genuine empathy in your phrasing. Acknowledge the user's emotional state."
        if normalised in ("low", "minimal", "none"):
            return "Keep the response neutral and factual; do not editorialise."
    return f"Empathy level: {value}."


def _render_redact(value: Any) -> str:
    """``redact`` is an audit/compliance directive — instruct the model to
    avoid specific content categories."""
    if isinstance(value, list):
        joined = ", ".join(str(v) for v in value)
        return f"Do NOT mention any of the following in your response: {joined}."
    return f"Do NOT mention {value} in your response."


def _render_audience_register(value: Any) -> str:
    return f"Match your language register to a {value} audience."


def _render_persona(value: Any) -> str:
    return f"Adopt this persona for your response: {value}."


# Registry maps known constraint keys to their rendering function. New
# constraint keys should be added here once their semantics are stable; until
# then they render via the generic fallback.
_KNOWN_RENDERERS: Dict[str, Any] = {
    "tone": _render_tone,
    "verbosity": _render_verbosity,
    "format": _render_format,
    "empathy_level": _render_empathy,
    "redact": _render_redact,
    "audience_register": _render_audience_register,
    "persona": _render_persona,
}


def _render_generic(key: str, value: Any) -> str:
    """Fallback for unknown constraint keys. Stringifies the value with json
    to keep complex values (lists, dicts) readable to the LLM."""
    if isinstance(value, (dict, list)):
        try:
            value_repr = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            value_repr = str(value)
    else:
        value_repr = str(value)
    return f"Apply this guideline: {key} = {value_repr}."


def formulation_constraints_to_instructions(
    constraints: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Translate a `C_F` constraint dict into a natural-language instruction
    block suitable for prepending to a formulation prompt.

    Returns ``None`` when the input is empty or contains only non-behavioral
    keys — callers can skip the instruction prepend in that case.

    The output is a single block of plain text with one instruction per line,
    introduced by an explicit ``"Response guidelines:"`` header so the LLM
    treats the lines as instructions rather than user content. Stable
    ordering: instructions are emitted in the order constraint keys appear
    in the dict (Python ≥3.7 preserves insertion order).
    """
    if not constraints:
        return None

    lines: List[str] = []
    for key, value in constraints.items():
        if key in _NON_BEHAVIORAL_KEYS:
            continue
        renderer = _KNOWN_RENDERERS.get(key)
        if renderer is not None:
            lines.append(renderer(value))
        else:
            lines.append(_render_generic(key, value))

    if not lines:
        return None

    body = "\n".join(f"- {line}" for line in lines)
    return f"Response guidelines:\n{body}"


__all__ = ["formulation_constraints_to_instructions"]
