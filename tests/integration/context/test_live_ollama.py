"""
Live integration tests for the cqs pipeline against a real Ollama instance.

These tests exercise the LLM-driven plugins (LlmContextInferencePlugin,
LlmPromptFormulationPlugin) and the full genie.context.resolve_and_formulate
pipeline end-to-end. They require a reachable Ollama server with a model
loaded.

Configure with env vars:

    OLLAMA_BASE_URL   default http://192.168.68.58:11434
    OLLAMA_MODEL      default gemma3:4b

Skip when Ollama isn't reachable so unit-test runs aren't dependent on it.
"""
from __future__ import annotations

import asyncio
import os
import socket
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import httpx
import pytest
import yaml

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://192.168.68.58:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:31b")
OLLAMA_PROVIDER_ID = "ollama_llm_provider_v1"


def _ollama_reachable() -> bool:
    try:
        # Quick TCP check first
        host = OLLAMA_BASE_URL.split("://", 1)[1].split(":", 1)[0]
        port_part = OLLAMA_BASE_URL.split(":")[-1].split("/", 1)[0]
        port = int(port_part) if port_part.isdigit() else 11434
        with socket.create_connection((host, port), timeout=2):
            pass
        # Then verify it actually speaks Ollama
        r = httpx.get(f"{OLLAMA_BASE_URL}/api/version", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _model_present() -> bool:
    """Confirm the configured model is actually pulled on the Ollama host.
    Matches by name prefix so `gemma4` matches `gemma4:latest`, `gemma4:4b`, etc."""
    try:
        r = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if r.status_code != 200:
            return False
        names = [m.get("name", "") for m in r.json().get("models", [])]
        target = OLLAMA_MODEL.split(":", 1)[0]
        return any(n.split(":", 1)[0] == target or n == OLLAMA_MODEL for n in names)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _ollama_reachable() or not _model_present(),
    reason=(
        f"Ollama not reachable at {OLLAMA_BASE_URL} or model "
        f"'{OLLAMA_MODEL}' not pulled. Override via OLLAMA_BASE_URL / OLLAMA_MODEL env vars."
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _make_genie(extension_configurations=None, prompt_registry_configurations=None,
                       default_prompt_registry_id=None):
    """Construct a real Genie with Ollama as the LLM and a local in-memory
    conversation store. Uses the new llm_ollama_base_url feature setting
    (F13) so we don't need to bypass it via llm_provider_configurations."""
    from genie_tooling.config.features import FeatureSettings
    from genie_tooling.config.models import MiddlewareConfig
    from genie_tooling.genie import Genie

    config = MiddlewareConfig(
        features=FeatureSettings(
            llm="ollama",
            llm_ollama_model_name=OLLAMA_MODEL,
            llm_ollama_base_url=OLLAMA_BASE_URL,
            conversation_state_provider="in_memory_convo_provider",
            default_llm_output_parser="pydantic_output_parser",
        ),
        # Boost the timeout for slower-than-default requests via direct override.
        # Larger models (e.g. gemma4:31b) routinely take >120s for a single call
        # at first-token latency, especially when the previous request flushed
        # the model from VRAM. 240s is comfortably above that.
        llm_provider_configurations={
            OLLAMA_PROVIDER_ID: {"request_timeout_seconds": 240.0},
        },
        extension_configurations=extension_configurations or {},
        prompt_registry_configurations=prompt_registry_configurations or {},
        default_prompt_registry_id=default_prompt_registry_id,
        auto_enable_registered_tools=True,
    )
    genie = await Genie.create(config=config)
    return genie


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ollama_basic_chat_works():
    """Sanity check: confirm we can chat with the configured model at all.
    If this fails everything downstream is unreliable."""
    genie = await _make_genie()
    try:
        response = await genie.llm.chat(
            [{"role": "user", "content": "Reply with the single word: PONG"}]
        )
        content = response["message"]["content"] or ""
        assert content.strip(), f"Empty response from {OLLAMA_MODEL}"
        # Don't be too strict about exact match; small models embellish.
        assert "PONG" in content.upper() or "pong" in content.lower()
    finally:
        await genie.close()


@pytest.mark.asyncio
async def test_llm_context_inference_returns_structured_profile():
    """Drives LlmContextInferencePlugin against a real Ollama. Verifies the
    LLM produces a JSON object the Pydantic parser accepts and the result
    shape matches what ContextManager expects (AudienceProfile/DiscourseTopic)."""
    from genie_tooling.context.plugins.inference.llm_inference import (
        LlmContextInferencePlugin,
    )

    genie = await _make_genie()
    try:
        plugin = LlmContextInferencePlugin()
        await plugin.setup(config={})

        raw_context = {
            "history": [
                {"role": "user", "content": "I need help calculating the gas constant."},
                {"role": "assistant", "content": "Sure, what would you like to know?"},
                {"role": "user", "content": "What is R in J/(mol*K)?"},
            ],
            "profile": {"expertise": "expert", "state": "Curious"},
        }
        inferred = await plugin.infer_context_properties(raw_context, genie)

        assert isinstance(inferred, dict), f"Expected dict, got {type(inferred)}"
        # Either the LLM responded cleanly (full shape) or the plugin returned {} on parse failure.
        # The {} case is a real failure of the LLM/parser interaction we want to know about.
        assert inferred, (
            f"LLM context inference returned empty dict — the model likely "
            f"failed to produce parseable JSON. Model: {OLLAMA_MODEL}"
        )
        assert "AudienceProfile" in inferred, f"missing AudienceProfile in {inferred}"
        assert "DiscourseTopic" in inferred, f"missing DiscourseTopic in {inferred}"
        ap = inferred["AudienceProfile"]
        assert "expertise" in ap and "state" in ap and "intent" in ap, (
            f"AudienceProfile missing required fields: {ap}"
        )
    finally:
        await genie.close()


@pytest.mark.asyncio
async def test_full_cqs_pipeline_with_real_llm_and_calculator(tmp_path: Path):
    """
    End-to-end through genie.context.resolve_and_formulate with:

      - Real Ollama for inference and formulation.
      - Heuristic predicate extractor (no LLM, deterministic).
      - Filesystem rule engine reading a tmp rule that routes to calculator_tool.
      - Real GenericToolDerivationPlugin executing the real calculator.

    Asserts a complete, sensible LLM-formulated final response that contains
    the calculator result. This is the highest-signal test that cqs works.
    """
    from genie_tooling.config.features import FeatureSettings
    from genie_tooling.config.models import MiddlewareConfig
    from genie_tooling.genie import Genie

    # 1. Write a custom rule that fires deterministically.
    rule = {
        "rule_id": "TEST_CALC_LIVE",
        "predicate": "predicate_calculate",
        "priority": 1,
        "conditions": [],  # Match any inferred context as long as predicate hits.
        "actions": [
            ["C_D", "set", "derivation_strategy_id", "generic_tool_derivation_v1"],
            ["C_D", "set", "tool_id", "calculator_tool"],
            ["C_D", "set", "params", {"num1": 7.0, "num2": 5.0, "operation": "add"}],
            ["C_F", "set", "prompt_template_id", "live_test_format"],
        ],
    }
    (tmp_path / "calc.yml").write_text(yaml.dump(rule))

    # 2. Write a Jinja-friendly formulation template the LLM will use.
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    # LlmPromptFormulationPlugin now defaults to Jinja2 so we use Jinja syntax.
    (template_dir / "live_test_format.prompt").write_text(
        "User question: '{{ original_query }}'.\n"
        "Calculator returned the result: {{ raw_data.result.result }}\n"
        "In one short sentence, give the numeric answer in plain English. "
        "Do not echo the dict. Do not include any extra commentary."
    )

    genie = await _make_genie(
        prompt_registry_configurations={
            "file_system_prompt_registry_v1": {
                "base_path": str(template_dir),
                "template_suffix": ".prompt",
            }
        },
        default_prompt_registry_id="file_system_prompt_registry_v1",
        extension_configurations={
            "context_engine": {
                "context_source_plugin_id": "configurable_context_source_v1",
                "context_source_config": {
                    "default_profile": {"expertise": "expert", "intent": "computation"},
                },
                "context_inference_plugin_id": "llm_context_inference_v1",
                "predicate_extractor_plugin_id": "heuristic_predicate_extractor_v1",
                "rule_engine_plugin_id": "filesystem_rule_engine_v1",
                "rule_engine_config": {"rules_path": str(tmp_path)},
                "formulation_strategy_plugin_id": "llm_prompt_formulation_v1",
            }
        },
    )
    try:
        assert hasattr(genie, "context"), "cqs bootstrap did not attach genie.context"
        final = await genie.context.resolve_and_formulate(
            query="please calculate 7 plus 5",
            session_id="live-session-1",
        )
        assert isinstance(final, str), f"expected str, got {type(final)}: {final!r}"
        # The deterministic calculator returns 12 — the LLM should mention it
        # either as a digit ("12") or as a word ("twelve"). Don't be stricter
        # than that; small models pick either form interchangeably.
        final_lower = final.lower()
        assert "12" in final or "twelve" in final_lower, (
            f"calculator answer (12 / 'twelve') missing from LLM-formulated "
            f"response: {final!r}"
        )
    finally:
        await genie.close()


# ---------------------------------------------------------------------------
# Extended live tests
# ---------------------------------------------------------------------------


def _calc_rule(num1: float, num2: float, op: str, template_id: str = "live_test_format") -> dict:
    return {
        "rule_id": f"TEST_RULE_{op.upper()}",
        "predicate": "predicate_calculate",
        "priority": 1,
        "conditions": [],
        "actions": [
            ["C_D", "set", "derivation_strategy_id", "generic_tool_derivation_v1"],
            ["C_D", "set", "tool_id", "calculator_tool"],
            ["C_D", "set", "params", {"num1": num1, "num2": num2, "operation": op}],
            ["C_F", "set", "prompt_template_id", template_id],
        ],
    }


def _write_jinja_template(template_dir: Path, name: str, content: str) -> None:
    template_dir.mkdir(parents=True, exist_ok=True)
    (template_dir / f"{name}.prompt").write_text(content)


async def _make_cqs_genie(tmp_path: Path, *, rule_yaml: dict, templates: dict[str, str],
                           source_profile: Optional[Dict[str, Any]] = None):
    """Compose a Genie with one rule and a template registry pointed at tmp."""
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    (rules_dir / f"{rule_yaml['rule_id']}.yml").write_text(yaml.dump(rule_yaml))

    template_dir = tmp_path / "templates"
    for name, content in templates.items():
        _write_jinja_template(template_dir, name, content)

    return await _make_genie(
        prompt_registry_configurations={
            "file_system_prompt_registry_v1": {
                "base_path": str(template_dir),
                "template_suffix": ".prompt",
            }
        },
        default_prompt_registry_id="file_system_prompt_registry_v1",
        extension_configurations={
            "context_engine": {
                "context_source_plugin_id": "configurable_context_source_v1",
                "context_source_config": {
                    "default_profile": source_profile or {},
                },
                "context_inference_plugin_id": "llm_context_inference_v1",
                "predicate_extractor_plugin_id": "heuristic_predicate_extractor_v1",
                "rule_engine_plugin_id": "filesystem_rule_engine_v1",
                "rule_engine_config": {"rules_path": str(rules_dir)},
                "formulation_strategy_plugin_id": "llm_prompt_formulation_v1",
            }
        },
    )


@pytest.mark.asyncio
async def test_feature_settings_ollama_base_url_takes_effect():
    """F13 regression: setting features.llm_ollama_base_url should resolve
    into the Ollama provider's base_url without manual override.
    """
    genie = await _make_genie()
    try:
        provider = await genie._llm_provider_manager.get_llm_provider(OLLAMA_PROVIDER_ID)  # type: ignore[attr-defined]
        assert provider is not None, "Ollama provider failed to load"
        assert provider._base_url == OLLAMA_BASE_URL.rstrip("/"), (
            f"Expected base_url {OLLAMA_BASE_URL!r}, got {provider._base_url!r}. "
            "FeatureSettings.llm_ollama_base_url is not propagating."
        )
    finally:
        await genie.close()


@pytest.mark.asyncio
async def test_cqs_arithmetic_subtraction(tmp_path: Path):
    """Confirms cqs handles a different op without surprises; locks in
    that the rule's params dict round-trips through YAML and reaches
    calculator_tool intact."""
    genie = await _make_cqs_genie(
        tmp_path,
        rule_yaml=_calc_rule(20.0, 8.0, "subtract"),
        templates={
            "live_test_format": (
                "A calculator computed: {{ raw_data.result.result }}\n"
                "The user asked: {{ original_query }}\n\n"
                "Repeat the calculator's exact numeric answer in one short, "
                "natural-language sentence. Do NOT recalculate, just restate "
                "the value above."
            )
        },
        source_profile={"intent": "computation", "expertise": "expert"},
    )
    try:
        final = await genie.context.resolve_and_formulate(
            query="please calculate 20 minus 8", session_id="sub-session"
        )
        final_lower = final.lower()
        assert "12" in final or "twelve" in final_lower, (
            f"expected 12 in subtraction result; got: {final!r}"
        )
    finally:
        await genie.close()


@pytest.mark.asyncio
async def test_cqs_no_rule_match_falls_through_to_default(tmp_path: Path):
    """When no rule matches the query's predicate, cqs should still produce
    a response (via the default-fallback derivation strategy) rather than
    raising. Verifies the graceful fallback path."""
    # A rule that matches a totally different predicate so it won't fire.
    rule = {
        "rule_id": "WONT_MATCH",
        "predicate": "predicate_who",  # Heuristic extractor for "what is..." emits predicate_what / predicate_is.
        "priority": 1,
        "conditions": [],
        "actions": [
            ["C_D", "set", "derivation_strategy_id", "generic_tool_derivation_v1"],
            ["C_D", "set", "tool_id", "calculator_tool"],
            ["C_D", "set", "params", {"num1": 1.0, "num2": 1.0, "operation": "add"}],
        ],
    }
    genie = await _make_cqs_genie(
        tmp_path,
        rule_yaml=rule,
        templates={
            # Default template name used when no C_F prompt_template_id is set.
            "default_formulation_prompt": (
                "Original query: {{ original_query }}\n"
                "Raw data: {{ raw_data }}\n"
                "In one short sentence, respond to the user."
            )
        },
        source_profile={"intent": "general", "expertise": "layperson"},
    )
    try:
        # The query won't match the rule above; cqs will fall through to the
        # default derivation strategy (generic_agent_derivation_v1) which then
        # errors because no command_processor_id is configured. The pipeline
        # should still return a string (the formulation plugin's error path
        # or an error-formulated response), not raise.
        final = await genie.context.resolve_and_formulate(
            query="tell me a fun fact", session_id="no-match-session"
        )
        assert isinstance(final, str)
        assert final, "expected non-empty response even on the no-rule-match path"
    finally:
        await genie.close()


@pytest.mark.asyncio
async def test_cqs_different_sessions_independent(tmp_path: Path):
    """Two distinct session_ids resolve through the same pipeline
    independently — no state leaks between sessions."""
    genie = await _make_cqs_genie(
        tmp_path,
        rule_yaml=_calc_rule(2.0, 3.0, "multiply"),
        templates={
            "live_test_format": (
                "A calculator computed: {{ raw_data.result.result }}\n"
                "The user asked: {{ original_query }}\n\n"
                "Repeat the calculator's exact numeric answer in one short, "
                "natural-language sentence. Do NOT recalculate, just restate "
                "the value above."
            )
        },
        source_profile={"intent": "computation", "expertise": "expert"},
    )
    async def _resolve_with_retry(session_id: str, attempts: int = 2) -> str:
        """The formulation LLM occasionally hedges and refuses to restate the
        calculator result. The deterministic part of the pipeline always runs;
        only the final LLM utterance is flaky. Retry up to `attempts` times
        looking for a response that mentions the answer."""
        last = ""
        for _ in range(attempts):
            r = await genie.context.resolve_and_formulate(
                query="please calculate two times three", session_id=session_id
            )
            last = r
            r_lower = r.lower()
            if "6" in r or "six" in r_lower:
                return r
        return last

    try:
        r1 = await _resolve_with_retry("session-A")
        r2 = await _resolve_with_retry("session-B")
        for tag, r in (("A", r1), ("B", r2)):
            r_lower = r.lower()
            assert "6" in r or "six" in r_lower, (
                f"session {tag} missing expected '6' / 'six' after 2 retries. "
                f"Response: {r!r}"
            )
    finally:
        await genie.close()


@pytest.mark.asyncio
async def test_cqs_division_by_zero_propagates_error(tmp_path: Path):
    """When the underlying tool itself reports an error, cqs derivation
    should surface that — the formulation must not silently produce a
    bogus number. Verifies the error path in GenericToolDerivationPlugin
    plus the LLM's behavior when given an error in raw_data."""
    genie = await _make_cqs_genie(
        tmp_path,
        rule_yaml=_calc_rule(10.0, 0.0, "divide"),
        templates={
            "live_test_format": (
                "Tool returned: {{ raw_data }}. If the tool reports an error, "
                "state in one sentence that an error occurred — do NOT invent a "
                "numeric answer."
            )
        },
        source_profile={"intent": "computation"},
    )
    try:
        final = await genie.context.resolve_and_formulate(
            query="please calculate 10 divide by 0", session_id="div-zero"
        )
        final_lower = final.lower()
        # The calculator raises on division by zero. The exception comes back
        # as a derivation error which is fed to the LLM. We accept either of
        # the two reasonable LLM behaviors: the model mentions "error" / "cannot"
        # or refuses with phrasing like "undefined" / "not possible".
        assert any(kw in final_lower for kw in (
            "error", "cannot", "undefined", "not possible", "divide by zero",
            "division by zero", "impossible", "invalid",
        )), f"expected error-acknowledgment from LLM; got: {final!r}"
    finally:
        await genie.close()
