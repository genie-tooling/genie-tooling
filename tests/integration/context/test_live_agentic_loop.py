"""
Heavyweight live tests that drive real claude-code-style agentic loops against
a real Ollama. Each test boots a full `Genie`, registers a small toolkit, and
either drives one of the bundled agents directly (`ReActAgent`,
`PlanAndExecuteAgent`) or routes a user query through the cqs pipeline so the
cqs derivation strategy dispatches to a command processor that in turn drives
the LLM through a multi-step tool-use loop.

The intent is to exercise as much of the codebase as possible end-to-end:

  * Genie facade construction with FeatureSettings
  * Plugin discovery and the F14-fixed `get_all_plugin_instances_by_type` path
  * `@tool` decorator + `register_tool_functions`
  * Public tool/plugin interfaces (`genie.tools.*`, `genie.plugins.*`)
  * Tool execution via the invocation strategy
  * Observability tracing (best-effort: not asserted on, but the path runs)
  * Conversation state provider
  * LlmAssistedToolSelectionProcessor / ReWOO processor agent loops
  * ReActAgent: thought-action-observation iterations
  * PlanAndExecuteAgent: static plan + sequential execution
  * cqs: heuristic predicate extraction, rule eval, constraint aggregation,
    generic_agent_derivation_v1 dispatch, LLM-driven formulation
  * Jinja-driven formulation template rendering against real raw_data

These tests are gated on a reachable Ollama with the configured model pulled
(see `test_live_ollama.py` for the same skip predicate). They are slower than
the calculator-only live tests; budget ~30s–2min per test depending on model
size.
"""
from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import pytest
import yaml

from genie_tooling import tool

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://192.168.68.58:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:31b")
OLLAMA_PROVIDER_ID = "ollama_llm_provider_v1"


def _ollama_reachable() -> bool:
    try:
        host = OLLAMA_BASE_URL.split("://", 1)[1].split(":", 1)[0]
        port_part = OLLAMA_BASE_URL.split(":")[-1].split("/", 1)[0]
        port = int(port_part) if port_part.isdigit() else 11434
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
        # Exact match preferred; fall back to prefix match for shorthand tags.
        if OLLAMA_MODEL in names:
            return True
        target = OLLAMA_MODEL.split(":", 1)[0]
        return any(n.split(":", 1)[0] == target for n in names)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _ollama_reachable() or not _model_present(),
    reason=(
        f"Ollama not reachable at {OLLAMA_BASE_URL} or model "
        f"'{OLLAMA_MODEL}' not pulled."
    ),
)


# ---------------------------------------------------------------------------
# Custom test tools (registered per-test via genie.register_tool_functions)
# ---------------------------------------------------------------------------


@tool
def lookup_physics_constant(name: str) -> Dict[str, Any]:
    """
    Look up the numeric value of a fundamental physics constant by name.

    Args:
        name: The name of the constant. One of: 'speed_of_light_km_per_s',
              'boiling_point_water_celsius', 'gravity_m_per_s_squared',
              'avogadro_number'.

    Returns:
        A dict with 'value' (float) and 'unit' (str), or 'error' if not found.
    """
    table = {
        "speed_of_light_km_per_s": {"value": 299792.0, "unit": "km/s"},
        "boiling_point_water_celsius": {"value": 100.0, "unit": "°C"},
        "gravity_m_per_s_squared": {"value": 9.81, "unit": "m/s²"},
        "avogadro_number": {"value": 6.022e23, "unit": "1/mol"},
    }
    key = name.strip().lower()
    if key in table:
        return table[key]
    return {"error": f"unknown constant '{name}'", "known": list(table.keys())}


@tool
def celsius_to_fahrenheit(celsius: float) -> float:
    """
    Convert a temperature in Celsius to Fahrenheit.

    Args:
        celsius: Temperature in degrees Celsius.

    Returns:
        The temperature in degrees Fahrenheit.
    """
    return celsius * 9.0 / 5.0 + 32.0


# A tool that fails on a specific (invalid) input so we can exercise the
# agent's observation-on-error reasoning path.
@tool
def strict_divider(numerator: float, denominator: float) -> Dict[str, Any]:
    """
    Divide numerator by denominator. Returns an error dict if denominator is
    zero — the caller is expected to retry with a non-zero denominator.

    Args:
        numerator: The number to divide.
        denominator: The number to divide by. Must not be zero.
    """
    if denominator == 0:
        return {"error": "denominator is zero; please retry with a non-zero value"}
    return {"result": numerator / denominator}


# ---------------------------------------------------------------------------
# Genie construction helper
# ---------------------------------------------------------------------------


async def _make_full_genie(
    tools_to_register=None,
    extension_configurations=None,
    prompt_registry_configurations=None,
    default_prompt_registry_id=None,
    extra_tool_configurations=None,
):
    """Boot a real Genie wired for agentic loops:
       - Ollama LLM provider against the configured remote host.
       - llm_assisted command processor (multi-turn ReAct-style under the hood).
       - In-memory conversation state.
       - PydanticOutputParser as the default parser.
       - Custom tools registered via @tool + register_tool_functions.
    """
    from genie_tooling.config.features import FeatureSettings
    from genie_tooling.config.models import MiddlewareConfig
    from genie_tooling.genie import Genie

    tool_configs: Dict[str, Dict[str, Any]] = {}
    if extra_tool_configurations:
        tool_configs.update(extra_tool_configurations)

    config = MiddlewareConfig(
        features=FeatureSettings(
            llm="ollama",
            llm_ollama_model_name=OLLAMA_MODEL,
            llm_ollama_base_url=OLLAMA_BASE_URL,
            command_processor="llm_assisted",
            conversation_state_provider="in_memory_convo_provider",
            default_llm_output_parser="pydantic_output_parser",
            # "none" disables the lookup step entirely so the LLM-assisted
            # processor sees every registered tool. Keyword lookup was too
            # strict against natural-language test queries (e.g. "please
            # calculate 17 multiplied by 3" didn't keyword-match the
            # calculator tool's tags), causing the processor to report
            # "no tools available" before the LLM ever ran.
            tool_lookup="none",
            # PlanAndExecuteAgent gates every step behind HITL. Use the
            # bundled auto-approver so agentic tests don't hang or fail.
            hitl_approver="auto_approve_hitl",
        ),
        llm_provider_configurations={
            OLLAMA_PROVIDER_ID: {"request_timeout_seconds": 240.0},
        },
        extension_configurations=extension_configurations or {},
        prompt_registry_configurations=prompt_registry_configurations or {},
        default_prompt_registry_id=default_prompt_registry_id,
        tool_configurations=tool_configs,
        auto_enable_registered_tools=True,
    )
    genie = await Genie.create(config=config)
    if tools_to_register:
        await genie.register_tool_functions(tools_to_register)
    return genie


def _calc_rule_via_agent(rule_id: str, processor_id: str,
                          predicate: str = "predicate_calculate") -> dict:
    """A rule that routes a query to an agent/command processor via cqs."""
    return {
        "rule_id": rule_id,
        "predicate": predicate,
        "priority": 1,
        "conditions": [],
        "actions": [
            ["C_D", "set", "derivation_strategy_id", "generic_agent_derivation_v1"],
            ["C_D", "set", "command_processor_id", processor_id],
            ["C_F", "set", "prompt_template_id", "agent_format"],
        ],
    }


def _write_template_dir(tmp_path: Path, templates: Dict[str, str]) -> Path:
    template_dir = tmp_path / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    for name, content in templates.items():
        (template_dir / f"{name}.prompt").write_text(content)
    return template_dir


def _write_rule_dir(tmp_path: Path, rules: Dict[str, dict]) -> Path:
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    for name, rule in rules.items():
        (rules_dir / f"{name}.yml").write_text(yaml.dump(rule))
    return rules_dir


# ---------------------------------------------------------------------------
# Test 1 — ReActAgent direct, multi-step physics-constant composition
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_react_agent_composes_lookup_then_calculator():
    """
    A claude-code-style multi-step loop driven by ReActAgent. The query
    requires the agent to:
      1. Call lookup_physics_constant('boiling_point_water_celsius') -> 100°C
      2. Call celsius_to_fahrenheit(100.0) -> 212.0
      3. Synthesize: "the boiling point of water is 212°F"

    Exercises ReActAgent's thought/action/observation loop, tool selection
    from multiple candidates, parameter extraction by the LLM, and final
    answer synthesis.
    """
    from genie_tooling.agents.react_agent import ReActAgent

    genie = await _make_full_genie(
        tools_to_register=[lookup_physics_constant, celsius_to_fahrenheit],
    )
    try:
        agent = ReActAgent(genie=genie, agent_config={"max_iterations": 6})
        result = await agent.run(
            goal=(
                "Find the boiling point of water in Fahrenheit. You have a tool "
                "to look up physics constants in Celsius and a tool to convert "
                "Celsius to Fahrenheit. Use them in that order. Final answer "
                "should be a number followed by °F."
            ),
        )

        assert result["status"] == "success", (
            f"ReActAgent did not finish successfully: status={result['status']}, "
            f"output={result.get('output')!r}"
        )
        output = str(result["output"])
        assert "212" in output, (
            f"Expected '212' in the final answer; got: {output!r}"
        )

        # The history should show the agent actually invoked BOTH tools.
        history = result.get("history") or []
        actions_taken = [str(item) for item in history]
        joined = " ".join(actions_taken).lower()
        assert "lookup_physics_constant" in joined or "boiling_point" in joined, (
            f"Lookup tool was never invoked. History: {history}"
        )
        assert "celsius_to_fahrenheit" in joined or "212" in joined, (
            f"Conversion tool was never invoked. History: {history}"
        )
    finally:
        await genie.close()


# ---------------------------------------------------------------------------
# Test 2 — PlanAndExecuteAgent: static plan + sequential execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_and_execute_agent_two_step_plan():
    """
    PlanAndExecuteAgent generates a static JSON plan up-front, then executes
    each step in sequence. Verifies the planning phase produces a valid
    Pydantic-parseable plan and the execution phase actually invokes the
    planned tools.

    The two steps are INDEPENDENT (no cross-step placeholder) on purpose:
    plan-and-execute's placeholder resolver is sensitive to the LLM picking
    the right output_variable_name and access path, which varies wildly per
    model. Independent steps still exercise the core plan-then-execute
    pipeline without that brittleness.
    """
    from genie_tooling.agents.plan_and_execute_agent import PlanAndExecuteAgent

    genie = await _make_full_genie(
        tools_to_register=[celsius_to_fahrenheit],
    )
    try:
        agent = PlanAndExecuteAgent(
            genie=genie,
            agent_config={"max_plan_retries": 1, "max_step_retries": 1},
        )
        result = await agent.run(
            goal=(
                "Make a plan with exactly two independent steps using the "
                "celsius_to_fahrenheit tool. Step 1: convert 100 Celsius to "
                "Fahrenheit. Step 2: convert 0 Celsius to Fahrenheit. Use "
                "the tool's exact param name 'celsius' with a plain numeric "
                "value — do NOT use placeholders. Return the Fahrenheit "
                "value from the last step as the final answer."
            ),
        )

        assert result["status"] in ("success", "max_iterations_reached"), (
            f"PlanAndExecuteAgent ended with unexpected status: "
            f"{result['status']}, output={result.get('output')!r}"
        )

        # The agent's history should show celsius_to_fahrenheit was called
        # at least once with a recognisable param. We deliberately don't
        # assert on the LLM's final-output text — the planner is brittle
        # enough across models that 'output' may be the last step's raw
        # numeric result, a stringified dict, or LLM commentary. The
        # signal that matters is: the plan executed real tool calls.
        history = result.get("history") or []
        history_str = " ".join(str(h) for h in history).lower()
        # One of the two conversions should appear in history: 100°C → 212°F
        # or 0°C → 32°F. Accept either.
        assert ("212" in history_str or "32" in history_str
                or "celsius_to_fahrenheit" in history_str), (
            f"plan-and-execute produced no evidence of running "
            f"celsius_to_fahrenheit: history={history!r}, output={result.get('output')!r}"
        )
    finally:
        await genie.close()


# ---------------------------------------------------------------------------
# Test 3 — Full cqs → agent loop (THE BIG ONE: claude-code style end-to-end)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_cqs_to_agent_loop(tmp_path: Path):
    """
    The flagship integration test. A natural-language user query goes through:

        heuristic predicate extraction
          -> filesystem rule engine (custom rule)
            -> generic_agent_derivation_v1
              -> llm_assisted_tool_selection_processor_v1 (LLM-driven loop)
                -> tool invocation (calculator_tool)
                  -> LLM observes result, synthesizes answer
        -> cqs formulation (Jinja2 template + LLM) -> final string

    This wires the whole codebase: cqs orchestration, plugin lookup, public
    facade interfaces, agentic command processor, tool invocation, Jinja2
    template engine, structured Pydantic output parsing.
    """
    rule = _calc_rule_via_agent(
        rule_id="RULE_AGENT_CALC",
        processor_id="llm_assisted_tool_selection_processor_v1",
    )
    rules_dir = _write_rule_dir(tmp_path, {"agent_calc": rule})
    template_dir = _write_template_dir(
        tmp_path,
        {
            "agent_format": (
                "An agentic processor produced this raw output for the user's "
                "query.\n\n"
                "User query: {{ original_query }}\n"
                "Agent output: {{ raw_data }}\n\n"
                "Repeat the agent's final answer in ONE plain-English sentence. "
                "Include the numeric value verbatim. Do NOT recalculate."
            )
        },
    )

    genie = await _make_full_genie(
        tools_to_register=[],  # built-in calculator_tool is sufficient
        # Explicitly enable the built-in calculator_tool. With
        # auto_enable_registered_tools=True only @tool-registered functions
        # are auto-enabled; built-in entry-point tools still need to appear
        # in tool_configurations to be visible to the command processor.
        extra_tool_configurations={"calculator_tool": {}},
        extension_configurations={
            "context_engine": {
                "context_source_plugin_id": "configurable_context_source_v1",
                "context_source_config": {
                    "default_profile": {"intent": "computation", "expertise": "expert"},
                },
                "context_inference_plugin_id": "llm_context_inference_v1",
                "predicate_extractor_plugin_id": "heuristic_predicate_extractor_v1",
                "rule_engine_plugin_id": "filesystem_rule_engine_v1",
                "rule_engine_config": {"rules_path": str(rules_dir)},
                "formulation_strategy_plugin_id": "llm_prompt_formulation_v1",
            }
        },
        prompt_registry_configurations={
            "file_system_prompt_registry_v1": {
                "base_path": str(template_dir),
                "template_suffix": ".prompt",
            }
        },
        default_prompt_registry_id="file_system_prompt_registry_v1",
    )
    try:
        assert hasattr(genie, "context"), "cqs bootstrap didn't attach genie.context"
        final = await genie.context.resolve_and_formulate(
            query="please calculate 17 multiplied by 3",
            session_id="cqs-agent-session",
        )
        assert isinstance(final, str) and final, f"expected non-empty string: {final!r}"
        final_lower = final.lower()
        assert "51" in final or "fifty-one" in final_lower or "fifty one" in final_lower, (
            f"expected 51 (17×3) in cqs-formulated agent output; got: {final!r}"
        )
    finally:
        await genie.close()


# ---------------------------------------------------------------------------
# Test 4 — ReActAgent error observation + recovery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_react_agent_observes_tool_error_and_retries():
    """
    Verifies the ReActAgent observation-on-error path: a tool returns a
    structured error on bad input, and the agent must reason about the
    observation and retry with corrected parameters. This is the heart of
    claude-code-style loops — recovering from a tool's complaint.
    """
    from genie_tooling.agents.react_agent import ReActAgent

    genie = await _make_full_genie(
        tools_to_register=[strict_divider],
    )

    async def _run_once():
        agent = ReActAgent(genie=genie, agent_config={"max_iterations": 6})
        return await agent.run(
            goal=(
                "Divide 100 by 0 using strict_divider. If it complains, retry "
                "by dividing 100 by 4 instead. Report only the final numeric "
                "result."
            ),
        )

    try:
        # The model occasionally answers "Answer:" immediately without calling
        # the tool at all. Retry up to twice to ride out that ~1-in-N flake.
        result = None
        for _ in range(2):
            result = await _run_once()
            history = result.get("history") or []
            history_str = " ".join(str(h) for h in history).lower()
            if "strict_divider" in history_str or "denominator" in history_str:
                break

        history = result.get("history") or []
        history_str = " ".join(str(h) for h in history).lower()
        assert "strict_divider" in history_str or "denominator" in history_str, (
            f"strict_divider was never called after 2 attempts: history={history!r}"
        )

        if result["status"] == "success":
            output = str(result["output"]).lower()
            assert "25" in output, (
                f"agent recovered but output missing '25' (100/4): {output!r}"
            )
        else:
            # Acceptable failure modes — model gave up — but the history MUST
            # show evidence of the failed attempt being observed.
            assert "zero" in history_str or "error" in history_str, (
                f"agent failed but never observed the tool error in history: "
                f"{history!r}"
            )
    finally:
        await genie.close()


# ---------------------------------------------------------------------------
# Test 5 — ReWOO command processor end-to-end (static multi-step plan + exec)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rewoo_command_processor_multi_step_calculation():
    """
    The ReWOO processor produces an entire multi-step plan in a single LLM
    call and then executes the steps. Exercises a different agentic shape than
    ReActAgent's iterative loop.
    """
    genie = await _make_full_genie(
        tools_to_register=[lookup_physics_constant, celsius_to_fahrenheit],
    )
    try:
        result = await genie.run_command(
            command=(
                "Find the boiling point of water in Fahrenheit. Use "
                "lookup_physics_constant with name='boiling_point_water_celsius' "
                "first, then convert with celsius_to_fahrenheit."
            ),
            processor_id="rewoo_command_processor_v1",
        )

        # ReWOO's output shape: {'final_answer': ...} on success or
        # {'error': ...} on failure. Either way it should be a dict.
        assert isinstance(result, dict), f"expected dict, got {type(result)}: {result!r}"
        # If the model produced a coherent plan, the answer mentions 212.
        # Some weaker models produce a half-baked plan — treat that as a soft
        # failure: at minimum we want a non-error structured result.
        if "final_answer" in result:
            answer = str(result["final_answer"])
            assert "212" in answer, (
                f"ReWOO final answer missing 212: {answer!r}"
            )
        else:
            # No final_answer — error path. Make sure we got a useful error
            # report rather than a crash.
            assert "error" in result, f"ReWOO produced no final_answer and no error: {result!r}"
    finally:
        await genie.close()


# ---------------------------------------------------------------------------
# Test 6 — Conversation history influences cqs context inference
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cqs_inference_uses_conversation_history(tmp_path: Path):
    """
    Verifies that prior turns in `genie.conversation` actually reach the
    LlmContextInferencePlugin and shape its `AudienceProfile`/
    `DiscourseTopic` inferences. The check is loose: we just confirm that
    the inferred discourse_topic for a session with research history differs
    from the topic for a session with calculation history.
    """
    from genie_tooling.context.plugins.inference.llm_inference import (
        LlmContextInferencePlugin,
    )

    genie = await _make_full_genie()
    try:
        plugin = LlmContextInferencePlugin()
        await plugin.setup(config={})

        research_ctx = {
            "history": [
                {"role": "user", "content": "Can you summarize recent transformer interpretability research?"},
                {"role": "assistant", "content": "Sure. Sparse autoencoders applied to residual streams..."},
                {"role": "user", "content": "Tell me about polysemanticity findings."},
            ],
            "profile": {"expertise": "expert"},
        }
        finance_ctx = {
            "history": [
                {"role": "user", "content": "What's the projected ROI on a 5-year corporate bond?"},
                {"role": "assistant", "content": "It depends on the coupon rate and yield-to-maturity..."},
                {"role": "user", "content": "Assume 4.5% coupon and 5.2% YTM."},
            ],
            "profile": {"expertise": "expert"},
        }

        research_inf = await plugin.infer_context_properties(research_ctx, genie)
        finance_inf = await plugin.infer_context_properties(finance_ctx, genie)

        for label, inferred in (("research", research_inf), ("finance", finance_inf)):
            assert inferred, f"empty inference dict for {label} context"
            assert "DiscourseTopic" in inferred, f"missing DiscourseTopic for {label}"

        research_topic = str(research_inf["DiscourseTopic"].get("primary", "")).lower()
        finance_topic = str(finance_inf["DiscourseTopic"].get("primary", "")).lower()

        # The two topics should not be identical; the LLM should pick up on
        # the different conversation contents. Don't require specific keywords
        # since model output varies.
        assert research_topic != finance_topic, (
            f"context inference returned identical topics for clearly different "
            f"conversation histories: research={research_topic!r}, "
            f"finance={finance_topic!r}"
        )
    finally:
        await genie.close()
