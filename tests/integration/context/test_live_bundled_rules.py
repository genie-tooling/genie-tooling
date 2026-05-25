"""Live integration tests for the six bundled cqs rules (Phase 4 — A4).

Each bundled rule has a profile + query that uniquely triggers it. These
tests boot a real Genie wired against qwen3.6:35b, send the trigger query,
and verify both:

  1. **Routing**: the right rule was selected. We inspect
     ``genie.context.last_decision.winning_rule_id``. This is the
     deterministic, auditable side of the pipeline.

  2. **Constraint propagation**: the rule's C_F constraints appear in
     ``last_decision.c_f`` and the translator's instruction text appears
     in ``last_decision.formulation_constraints_text``. This is the
     evidence that C_F isn't theater anymore (the core A1 commitment).

  3. **Where feasible**: the LLM-formulated response is sensible.

Skipped automatically when Ollama is unreachable or qwen3.6:35b isn't pulled.
"""
from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import pytest

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://192.168.68.58:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.6:35b")
OLLAMA_PROVIDER_ID = "ollama_llm_provider_v1"


def _ollama_reachable() -> bool:
    try:
        host = OLLAMA_BASE_URL.split("://", 1)[1].split(":", 1)[0]
        port = int(OLLAMA_BASE_URL.split(":")[-1].split("/", 1)[0])
        with socket.create_connection((host, port), timeout=2):
            pass
        r = httpx.get(f"{OLLAMA_BASE_URL}/api/version", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _model_present() -> bool:
    try:
        r = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if r.status_code != 200:
            return False
        names = [m.get("name", "") for m in r.json().get("models", [])]
        if OLLAMA_MODEL in names:
            return True
        target = OLLAMA_MODEL.split(":", 1)[0]
        return any(n.split(":", 1)[0] == target for n in names)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _ollama_reachable() or not _model_present(),
    reason=f"Ollama not reachable at {OLLAMA_BASE_URL} or model '{OLLAMA_MODEL}' not pulled.",
)


# ---------------------------------------------------------------------------
# Genie construction
# ---------------------------------------------------------------------------


async def _make_bundled_rules_genie(
    profile: Dict[str, Any],
    *,
    rule_engine_config_override: Optional[Dict[str, Any]] = None,
):
    """Boot a real Genie wired exactly as a corporate deployment would be:
       * Ollama against the configured remote.
       * cqs bootstrap with the BUNDLED rules dir (default).
       * Bundled prompt templates registry (for templates like
         summarize_agent_output that research_rule.yml references).
       * Calculator tool explicitly enabled so the LLM-assisted processor
         has a real tool to pick.
       * Auto-approve HITL so PlanAndExecute / ReWOO don't block.
    """
    from genie_tooling.config.features import FeatureSettings
    from genie_tooling.config.models import MiddlewareConfig
    from genie_tooling.genie import Genie

    # The bundled prompt-templates path lives next to the bundled rules dir,
    # but importlib.resources stays the right way to resolve it.
    import importlib.resources

    bundled_templates = str(
        importlib.resources.files("genie_tooling.context") / "prompt_templates"
    )

    rule_engine_cfg = rule_engine_config_override or {}

    config = MiddlewareConfig(
        features=FeatureSettings(
            llm="ollama",
            llm_ollama_model_name=OLLAMA_MODEL,
            llm_ollama_base_url=OLLAMA_BASE_URL,
            command_processor="llm_assisted",
            conversation_state_provider="in_memory_convo_provider",
            default_llm_output_parser="pydantic_output_parser",
            tool_lookup="none",
            hitl_approver="auto_approve_hitl",
        ),
        llm_provider_configurations={
            OLLAMA_PROVIDER_ID: {"request_timeout_seconds": 240.0},
        },
        tool_configurations={"calculator_tool": {}},
        prompt_registry_configurations={
            "file_system_prompt_registry_v1": {
                "base_path": bundled_templates,
                "template_suffix": ".prompt",
            }
        },
        default_prompt_registry_id="file_system_prompt_registry_v1",
        extension_configurations={
            "context_engine": {
                "context_source_plugin_id": "configurable_context_source_v1",
                "context_source_config": {"default_profile": profile},
                "context_inference_plugin_id": "llm_context_inference_v1",
                # Heuristic predicate extractor is deterministic — exactly
                # what audit demands. The LLM extractor is non-deterministic
                # and unsuitable for rules-driven corporate use.
                "predicate_extractor_plugin_id": "heuristic_predicate_extractor_v1",
                "rule_engine_plugin_id": "filesystem_rule_engine_v1",
                "rule_engine_config": rule_engine_cfg,
                "formulation_strategy_plugin_id": "llm_prompt_formulation_v1",
            }
        },
        auto_enable_registered_tools=True,
    )
    return await Genie.create(config=config)


# ---------------------------------------------------------------------------
# 01_fact_finding.yml — RULE_FACT_FINDING
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rule_fact_finding_fires_with_right_constraints():
    """Profile (intent=fact_finding) + predicate_what query triggers
    RULE_FACT_FINDING. Its C_F (tone=encyclopedic, verbosity=concise)
    must propagate into both the C_F dict AND the translator's
    instruction text."""
    genie = await _make_bundled_rules_genie(
        profile={"intent": "fact_finding", "expertise": "expert", "state": "Neutral"},
    )
    try:
        await genie.context.resolve_and_formulate(
            query="what is two plus three",
            session_id="fact-finding-session",
        )
        rec = genie.context.last_decision
        assert rec is not None, "DecisionRecord must be produced"
        assert rec.winning_rule_id == "RULE_FACT_FINDING", (
            f"expected RULE_FACT_FINDING to win, got {rec.winning_rule_id!r}; "
            f"all matches: {[(r.rule_id, r.score) for r in rec.ranked_rules]}"
        )
        assert rec.c_f.get("tone") == "encyclopedic"
        assert rec.c_f.get("verbosity") == "concise"
        # The translator's text appears in the audit record (the most
        # important audit signal — proves the LLM saw the instructions).
        assert rec.formulation_constraints_text is not None
        assert "encyclopedic" in rec.formulation_constraints_text
        assert "concise" in rec.formulation_constraints_text
        # Final response acknowledges the answer in some form.
        assert rec.final_response and len(rec.final_response) > 0
    finally:
        await genie.close()


# ---------------------------------------------------------------------------
# 02_calculate.yml — RULE_CALCULATION
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rule_calculation_fires_with_direct_answer_format():
    """Profile (intent=computation) + predicate_calculate query triggers
    RULE_CALCULATION. The format=direct_answer C_F shapes the
    formulation."""
    genie = await _make_bundled_rules_genie(
        profile={"intent": "computation", "expertise": "expert", "state": "Neutral"},
    )
    try:
        await genie.context.resolve_and_formulate(
            query="calculate twelve multiplied by five",
            session_id="calc-session",
        )
        rec = genie.context.last_decision
        assert rec is not None
        assert rec.winning_rule_id == "RULE_CALCULATION", (
            f"expected RULE_CALCULATION, got {rec.winning_rule_id!r}"
        )
        assert rec.c_f.get("format") == "direct_answer"
        assert rec.formulation_constraints_text is not None
        assert "direct answer only" in rec.formulation_constraints_text
        # The LLM-assisted processor + calculator tool should produce 60.
        assert "60" in rec.final_response or "sixty" in rec.final_response.lower(), (
            f"expected 60 / sixty in response: {rec.final_response!r}"
        )
    finally:
        await genie.close()


# ---------------------------------------------------------------------------
# 10_expert_user.yml — RULE_EXPERT_AUDIENCE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rule_expert_user_fires_with_technical_tone():
    """When no more-specific rule matches but the user is an expert,
    RULE_EXPERT_AUDIENCE fires (priority 50, wildcard predicate). Routes
    through ReWOO for richer planning; we only assert on rule selection +
    C_F here because ReWOO end-to-end is exercised separately."""
    genie = await _make_bundled_rules_genie(
        profile={
            "expertise": "expert",
            "state": "Neutral",
            # Deliberately NO `intent` so 01/02/research-rule don't match.
        },
    )
    try:
        # Use a query whose predicate WON'T match any specific-predicate rule.
        # "compare" is in the heuristic keyword set, so predicate=predicate_compare.
        # 01 (predicate_what), 02 (predicate_calculate), research (predicate_generic_inquiry)
        # all skip → 10 (wildcard + expertise=expert) matches.
        await genie.context.resolve_and_formulate(
            query="compare floating-point precision under IEEE 754 across single and double",
            session_id="expert-session",
        )
        rec = genie.context.last_decision
        assert rec is not None
        assert rec.winning_rule_id == "RULE_EXPERT_AUDIENCE", (
            f"expected RULE_EXPERT_AUDIENCE, got {rec.winning_rule_id!r}; "
            f"predicate={rec.predicate!r}, ranked={[(r.rule_id, r.score) for r in rec.ranked_rules]}"
        )
        assert rec.c_f.get("tone") == "formal and technical"
        assert rec.c_f.get("verbosity") == "high"
        assert rec.formulation_constraints_text is not None
        assert "formal and technical" in rec.formulation_constraints_text
        assert "detailed" in rec.formulation_constraints_text  # 'high' -> 'detailed'
        # ReWOO downstream may error or succeed; we don't assert on response
        # text because ReWOO + LLM-only-no-tool can produce wildly varying
        # outputs. The audit signal (rule + constraints) is the contract.
    finally:
        await genie.close()


# ---------------------------------------------------------------------------
# 11_stressed_user.yml — RULE_STRESSED_AUDIENCE  (the rule we just fixed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rule_stressed_user_overrides_expert():
    """A stressed expert should be routed through RULE_STRESSED_AUDIENCE
    (priority 40) ahead of RULE_EXPERT_AUDIENCE (priority 50). This rule
    was structurally broken pre-Phase-4A4 (no derivation_strategy_id);
    test proves it now works end-to-end."""
    genie = await _make_bundled_rules_genie(
        profile={
            "expertise": "expert",
            "state": "Stressed",
        },
    )
    try:
        await genie.context.resolve_and_formulate(
            query="evaluate why my analysis keeps breaking",
            session_id="stressed-session",
        )
        rec = genie.context.last_decision
        assert rec is not None
        assert rec.winning_rule_id == "RULE_STRESSED_AUDIENCE", (
            f"expected RULE_STRESSED_AUDIENCE to outrank expert rule; got "
            f"{rec.winning_rule_id!r}"
        )
        assert rec.c_f.get("tone") == "calm and reassuring"
        assert rec.c_f.get("empathy_level") == "high"
        assert rec.formulation_constraints_text is not None
        assert "calm and reassuring" in rec.formulation_constraints_text
        assert "empathy" in rec.formulation_constraints_text.lower()
        # The pipeline must complete — pre-fix it errored at derivation.
        assert rec.derivation_status == "success", (
            f"stressed-user pipeline failed at derivation: {rec.derivation_error!r}"
        )
        assert rec.final_response
    finally:
        await genie.close()


# ---------------------------------------------------------------------------
# 99_default_fallback.yml — RULE_DEFAULT_FALLBACK
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rule_default_fallback_when_nothing_specific_matches():
    """No specific predicate, no expert, not stressed → falls through to
    the priority-999 catch-all."""
    genie = await _make_bundled_rules_genie(
        profile={
            "expertise": "layperson",
            "state": "Neutral",
            # No `intent` field — none of the intent-conditioned rules match.
        },
    )
    try:
        await genie.context.resolve_and_formulate(
            # "tell me about" has no keywords from the heuristic extractor
            # list → predicate_generic_inquiry. With no intent=research,
            # the research rule fails too. Wildcard rules check next:
            # 11 (stressed) fails (Neutral), 10 (expert) fails (layperson),
            # 99 (no conditions) wins.
            query="tell me about cloud cover patterns",
            session_id="fallback-session",
        )
        rec = genie.context.last_decision
        assert rec is not None
        assert rec.winning_rule_id == "RULE_DEFAULT_FALLBACK", (
            f"expected RULE_DEFAULT_FALLBACK, got {rec.winning_rule_id!r}; "
            f"predicate={rec.predicate!r}, ranked={[(r.rule_id, r.score) for r in rec.ranked_rules]}"
        )
        assert rec.c_f.get("tone") == "friendly and helpful"
        assert rec.formulation_constraints_text is not None
        assert "friendly and helpful" in rec.formulation_constraints_text
        assert rec.final_response
    finally:
        await genie.close()


# ---------------------------------------------------------------------------
# research_rule.yml — RULE_DEEP_RESEARCH (rule selection only)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rule_deep_research_fires_on_research_intent():
    """research_rule routes to deep_research_agent_v1 which is too heavy
    to run end-to-end here — that processor needs web/arxiv tools, vector
    stores, and multiple LLM rounds. We assert on the audit contract:
    correct rule selected, correct C_F constraints, correct template ID."""
    genie = await _make_bundled_rules_genie(
        profile={
            "intent": "research",
            "expertise": "expert",
            "state": "Neutral",
        },
    )
    try:
        await genie.context.resolve_and_formulate(
            query="tell me about advances in protein folding prediction",
            session_id="research-session",
        )
        rec = genie.context.last_decision
        assert rec is not None
        assert rec.winning_rule_id == "RULE_DEEP_RESEARCH", (
            f"expected RULE_DEEP_RESEARCH, got {rec.winning_rule_id!r}; "
            f"predicate={rec.predicate!r}, ranked={[(r.rule_id, r.score) for r in rec.ranked_rules]}"
        )
        assert rec.c_d.get("command_processor_id") == "deep_research_agent_v1"
        assert rec.c_f.get("prompt_template_id") == "summarize_agent_output"
        # prompt_template_id is wiring, not behavior — it must NOT appear in
        # the instruction text the LLM saw (the translator filters
        # non-behavioral keys). This rule has no behavioral C_F so the
        # translator returns None.
        assert rec.formulation_constraints_text is None
        # We don't assert on the actual final response — deep research is
        # heavy and the agent may report an error if it can't reach the
        # tools it needs. The audit contract is met regardless.
    finally:
        await genie.close()


# ---------------------------------------------------------------------------
# Differential test: same query, two profiles → demonstrably different
# response. This is the headline "context-aware response shaping" demo
# that justifies the cqs subpackage existing.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stressed_vs_expert_produces_different_formulation_instructions():
    """The same query routed via two different rules produces DIFFERENT
    formulation instructions in the audit record. This is the smoking gun
    that proves C_F constraints are no longer theater."""
    query = "explain why this calculation is failing"

    expert_genie = await _make_bundled_rules_genie(
        profile={"expertise": "expert", "state": "Neutral"},
    )
    try:
        await expert_genie.context.resolve_and_formulate(
            query=query, session_id="expert-tone-session"
        )
        expert_rec = expert_genie.context.last_decision
    finally:
        await expert_genie.close()

    stressed_genie = await _make_bundled_rules_genie(
        profile={"expertise": "expert", "state": "Stressed"},
    )
    try:
        await stressed_genie.context.resolve_and_formulate(
            query=query, session_id="stressed-tone-session"
        )
        stressed_rec = stressed_genie.context.last_decision
    finally:
        await stressed_genie.close()

    assert expert_rec is not None and stressed_rec is not None

    # Different rules fired
    assert expert_rec.winning_rule_id == "RULE_EXPERT_AUDIENCE"
    assert stressed_rec.winning_rule_id == "RULE_STRESSED_AUDIENCE"

    # Different C_F constraints
    assert expert_rec.c_f.get("tone") == "formal and technical"
    assert stressed_rec.c_f.get("tone") == "calm and reassuring"

    # Different translator instructions — this is what the LLM literally
    # sees. The audit-side proof.
    assert expert_rec.formulation_constraints_text is not None
    assert stressed_rec.formulation_constraints_text is not None
    assert (
        expert_rec.formulation_constraints_text
        != stressed_rec.formulation_constraints_text
    )
    assert "formal and technical" in expert_rec.formulation_constraints_text
    assert "calm and reassuring" in stressed_rec.formulation_constraints_text
    assert "empathy" in stressed_rec.formulation_constraints_text.lower()
