"""Unit tests for DecisionRecord schema + ContextManager emission (A3).

Verifies:
  * The schema accepts representative inputs and produces clean JSON.
  * ContextManager.resolve_and_formulate assembles one DecisionRecord per
    call, populating every stage that ran.
  * `ContextInterface.last_decision` exposes the record post-call.
  * The audit.decision_record trace event is emitted.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from genie_tooling.context.audit import (
    DecisionRecord,
    RankedRuleEntry,
    _DecisionRecorder,
)
from genie_tooling.context.interface import ContextInterface
from genie_tooling.context.manager import ContextManager
from genie_tooling.context.protocols import (
    ContextInferencePlugin,
    ContextSourcePlugin,
    DerivationStrategyPlugin,
    FormulationStrategyPlugin,
    PredicateExtractorPlugin,
    RuleEnginePlugin,
)


def test_decision_record_round_trip_json():
    """The model must serialize/deserialize cleanly — audit downstream
    consumers will ingest it as JSON."""
    rec = DecisionRecord(
        query="hello",
        session_id="s1",
        user_identity={"sub": "user-42", "role": "analyst"},
        predicate="predicate_what",
        ranked_rules=[RankedRuleEntry(rule_id="R1", score=1.0, priority=10)],
        winning_rule_id="R1",
        c_d={"derivation_strategy_id": "generic_tool_derivation_v1"},
        c_f={"tone": "formal"},
        derivation_status="success",
        derivation_result_preview="{'result': 42}",
        formulation_template_id="default_formulation_prompt",
        formulation_constraints_text="Response guidelines:\n- Use a formal tone in your response.",
        final_response="The answer is 42.",
        stage_timings_ms={"inference": 100.0, "derivation": 50.0},
    )
    payload = rec.to_jsonable()
    assert payload["query"] == "hello"
    assert payload["ranked_rules"][0]["rule_id"] == "R1"
    assert payload["c_f"]["tone"] == "formal"
    # Round-trip
    rec2 = DecisionRecord.model_validate(payload)
    assert rec2.decision_id == rec.decision_id
    assert rec2.formulation_constraints_text == rec.formulation_constraints_text


def test_decision_recorder_stage_timing_is_recorded():
    rec = _DecisionRecorder(query="q", session_id="s")
    with rec.stage("test_stage"):
        # do some work
        import time
        time.sleep(0.001)
    assert "test_stage" in rec.record.stage_timings_ms
    assert rec.record.stage_timings_ms["test_stage"] > 0.0


@pytest.mark.asyncio()
async def test_decision_recorder_async_stage_timing():
    rec = _DecisionRecorder(query="q")
    async with rec.stage("async_stage"):
        import asyncio
        await asyncio.sleep(0.001)
    assert "async_stage" in rec.record.stage_timings_ms


def _spec_mock(protocol_cls, **methods):
    """MagicMock with spec= so it passes ContextManager's isinstance check,
    plus AsyncMock overrides for the methods the pipeline calls."""
    m = MagicMock(spec=protocol_cls)
    for name, return_value in methods.items():
        setattr(m, name, AsyncMock(return_value=return_value))
    # Plugin's plugin_id attribute - propagate for the audit record
    m.plugin_id = methods.get("_plugin_id", f"{protocol_cls.__name__.lower()}_mock_v1")
    return m


@pytest.fixture()
def fully_mocked_manager():
    mock_genie = MagicMock()
    mock_genie.conversation = MagicMock()
    mock_genie.conversation.load_state = AsyncMock(return_value={"history": []})
    mock_genie.observability = MagicMock()
    mock_genie.observability.trace_event = AsyncMock()

    source = MagicMock(spec=ContextSourcePlugin)
    source.plugin_id = "mock_source_v1"
    source.get_profile = AsyncMock(return_value={"expertise": "expert"})

    inference = MagicMock(spec=ContextInferencePlugin)
    inference.plugin_id = "mock_inference_v1"
    inference.infer_context_properties = AsyncMock(
        return_value={"AudienceProfile": {"expertise": "expert"}, "DiscourseTopic": {"primary": "math"}}
    )

    predicate_ext = MagicMock(spec=PredicateExtractorPlugin)
    predicate_ext.plugin_id = "mock_predicate_v1"
    predicate_ext.extract = AsyncMock(return_value="predicate_calculate")

    rule_engine = MagicMock(spec=RuleEnginePlugin)
    rule_engine.plugin_id = "mock_rule_engine_v1"
    rule_engine.evaluate = AsyncMock(
        return_value=[
            (
                {
                    "rule_id": "TEST_RULE",
                    "priority": 5,
                    "actions": [
                        ["C_D", "set", "derivation_strategy_id", "generic_tool_derivation_v1"],
                        ["C_F", "set", "tone", "formal"],
                        ["C_F", "set", "verbosity", "concise"],
                    ],
                },
                1.0,
            )
        ]
    )

    derivation = MagicMock(spec=DerivationStrategyPlugin)
    derivation.plugin_id = "generic_tool_derivation_v1"
    derivation.derive = AsyncMock(
        return_value={"status": "success", "result": {"result": 42}}
    )
    mock_genie.plugins = MagicMock()
    mock_genie.plugins.get_instance = AsyncMock(return_value=derivation)

    formulation = MagicMock(spec=FormulationStrategyPlugin)
    formulation.plugin_id = "llm_prompt_formulation_v1"
    formulation.formulate = AsyncMock(return_value="The answer is 42 in a formal tone.")

    mgr = ContextManager(genie=mock_genie, config={})
    mgr._context_source = source
    mgr._inference_engine = inference
    mgr._predicate_extractor = predicate_ext
    mgr._rule_engine = rule_engine
    mgr._formulation_strategy = formulation
    return mgr, mock_genie


@pytest.mark.asyncio()
async def test_resolve_and_formulate_emits_one_decision_record(fully_mocked_manager):
    mgr, genie = fully_mocked_manager
    response = await mgr.resolve_and_formulate(
        query="compute 6 times 7",
        session_id="session-x",
        user_identity={"sub": "user-1"},
    )
    # The final response goes back to the caller as before
    assert isinstance(response, str)

    # AND the audit record is now in-memory
    rec = mgr.last_decision
    assert rec is not None
    assert rec.session_id == "session-x"
    assert rec.user_identity == {"sub": "user-1"}
    assert rec.query == "compute 6 times 7"

    # AND exactly one audit.decision_record trace event was emitted
    audit_calls = [
        call
        for call in genie.observability.trace_event.await_args_list
        if call.args and call.args[0] == "audit.decision_record"
    ]
    assert len(audit_calls) == 1
    payload = audit_calls[0].args[1]
    assert payload["session_id"] == "session-x"
    assert payload["query"] == "compute 6 times 7"


@pytest.mark.asyncio()
async def test_decision_record_captures_full_pipeline(fully_mocked_manager):
    mgr, _ = fully_mocked_manager
    await mgr.resolve_and_formulate(query="q", session_id="s")
    rec = mgr.last_decision
    assert rec is not None

    # Each stage's signal is recorded
    assert rec.profile == {"expertise": "expert"}
    assert rec.history_length == 0
    assert rec.inferred_context == {
        "AudienceProfile": {"expertise": "expert"},
        "DiscourseTopic": {"primary": "math"},
    }
    assert rec.predicate == "predicate_calculate"
    assert rec.predicate_extractor_id == "mock_predicate_v1"
    assert rec.rule_engine_id == "mock_rule_engine_v1"
    assert len(rec.ranked_rules) == 1
    assert rec.ranked_rules[0].rule_id == "TEST_RULE"
    assert rec.winning_rule_id == "TEST_RULE"

    # Constraint aggregation flowed through
    assert rec.c_d == {"derivation_strategy_id": "generic_tool_derivation_v1"}
    assert rec.c_f == {"tone": "formal", "verbosity": "concise"}

    # The C_F translator's text is captured for audit reproducibility
    assert rec.formulation_constraints_text is not None
    assert "formal" in rec.formulation_constraints_text
    assert "concise" in rec.formulation_constraints_text

    # Derivation captured
    assert rec.derivation_strategy_id == "generic_tool_derivation_v1"
    assert rec.derivation_status == "success"
    assert rec.derivation_result_preview is not None
    assert "42" in rec.derivation_result_preview

    # Formulation captured
    assert rec.formulation_strategy_id == "llm_prompt_formulation_v1"
    assert rec.final_response == "The answer is 42 in a formal tone."

    # Every pipeline stage has a timing entry
    expected_stages = {
        "context_load",
        "inference",
        "predicate_extract",
        "rule_evaluate",
        "aggregate",
        "derivation",
        "formulation",
    }
    assert expected_stages.issubset(rec.stage_timings_ms.keys())
    assert all(v >= 0.0 for v in rec.stage_timings_ms.values())


@pytest.mark.asyncio()
async def test_decision_record_when_pipeline_aborts_on_misconfig():
    """Even when the engine isn't fully configured, an audit record is still
    emitted — failure to launch is an audit-relevant event."""
    mock_genie = MagicMock()
    mock_genie.observability = MagicMock()
    mock_genie.observability.trace_event = AsyncMock()
    mgr = ContextManager(genie=mock_genie, config={})
    # Don't set up plugins; pipeline should bail
    result = await mgr.resolve_and_formulate(query="q", session_id="s")
    assert isinstance(result, dict) and "error" in result

    rec = mgr.last_decision
    assert rec is not None
    assert rec.error is not None
    assert "not fully configured" in rec.error.lower()


@pytest.mark.asyncio()
async def test_context_interface_exposes_last_decision(fully_mocked_manager):
    """ContextInterface.last_decision passes through to the manager's slot."""
    mgr, _ = fully_mocked_manager
    iface = ContextInterface(manager=mgr)
    assert iface.last_decision is None  # nothing yet
    await iface.resolve_and_formulate(query="q", session_id="s")
    rec = iface.last_decision
    assert rec is not None
    assert rec.query == "q"


@pytest.mark.asyncio()
async def test_decision_records_have_unique_ids(fully_mocked_manager):
    """Each resolve_and_formulate produces a fresh UUID."""
    mgr, _ = fully_mocked_manager
    await mgr.resolve_and_formulate(query="q1", session_id="s")
    id1 = mgr.last_decision.decision_id
    await mgr.resolve_and_formulate(query="q2", session_id="s")
    id2 = mgr.last_decision.decision_id
    assert id1 != id2
