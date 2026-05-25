"""DecisionRecord audit schema (A3).

A single structured record assembled by ContextManager.resolve_and_formulate
and emitted as an ``audit.decision_record`` trace event. The whole point of
the corporate-harness framing is that auditors can ask "why did the system
respond this way to that query at that time?" and get one document back,
not a forensic reconstruction across scattered trace events.

Every stage of the cqs pipeline contributes a section. The record is
serializable to JSON without losing semantic content (`raw_data`, tool
outputs, etc. may contain non-JSON-serializable values; the recorder
coerces them to ``repr`` strings where necessary).

Compatibility note: this schema is part of the audit contract. Adding
fields is fine; renaming or removing fields breaks audit consumers and
must be done with care + version bumps.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RankedRuleEntry(BaseModel):
    """A single (rule_id, score) ranking result from the rule engine.
    Captured for ALL matching rules so auditors can see what else was
    considered, not just the winner.
    """
    rule_id: str
    score: float
    priority: Optional[int] = None


class HITLApprovalEntry(BaseModel):
    """One HITL approval gate that fired during this decision."""
    request_id: str
    approver_id: str
    status: str  # "approved", "denied", "error", "auto_approved", etc.
    reason: Optional[str] = None
    decided_at: float


class ToolCallEntry(BaseModel):
    """One tool execution that fired during this decision."""
    tool_id: str
    params: Dict[str, Any] = Field(default_factory=dict)
    result_preview: Optional[str] = None  # repr-truncated, full value lives in traces
    error: Optional[str] = None
    started_at: float
    duration_ms: float
    caller_chain: List[str] = Field(default_factory=list)
    """Components that requested this tool execution, outermost first. E.g.
    ``["cqs.context_manager", "GenericAgentDerivationPlugin",
       "llm_assisted_tool_selection_processor_v1"]`` — read top-down."""


class DecisionRecord(BaseModel):
    """The full audit document for a single ``resolve_and_formulate`` call.

    Designed for direct serialization (e.g. into a SIEM / log warehouse).
    Every field that could be missing is Optional with an explicit default,
    so a record where the pipeline failed mid-stage is still well-formed
    JSON.
    """

    # --- identity ---
    decision_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: str = "1"
    session_id: Optional[str] = None
    user_identity: Optional[Dict[str, Any]] = None
    """Free-form caller-supplied identity (corporate SSO uid, role, tenant
    id, etc.). The framework doesn't interpret this; auditors do."""

    # --- inputs ---
    query: str = ""

    # --- stage 1: context loading ---
    profile: Optional[Dict[str, Any]] = None
    history_length: Optional[int] = None

    # --- stage 2: inference ---
    inferred_context: Dict[str, Any] = Field(default_factory=dict)

    # --- stage 3: predicate extraction ---
    predicate: Optional[str] = None
    predicate_extractor_id: Optional[str] = None

    # --- stage 4: rule evaluation ---
    rule_engine_id: Optional[str] = None
    ranked_rules: List[RankedRuleEntry] = Field(default_factory=list)
    winning_rule_id: Optional[str] = None

    # --- stage 5: constraint aggregation ---
    c_d: Dict[str, Any] = Field(default_factory=dict)
    c_f: Dict[str, Any] = Field(default_factory=dict)

    # --- stage 6: derivation ---
    derivation_strategy_id: Optional[str] = None
    derivation_status: Optional[str] = None  # "success" | "error"
    derivation_result_preview: Optional[str] = None
    derivation_error: Optional[str] = None

    # --- stage 7: formulation ---
    formulation_strategy_id: Optional[str] = None
    formulation_template_id: Optional[str] = None
    formulation_constraints_text: Optional[str] = None
    """The instruction block produced by the C_F translator (A1). This is
    the EXACT text prepended to the rendered template before the LLM saw
    it — auditors can reproduce the LLM input from this + the template."""
    final_response: Optional[str] = None

    # --- correlated subordinate events ---
    tool_calls: List[ToolCallEntry] = Field(default_factory=list)
    hitl_approvals: List[HITLApprovalEntry] = Field(default_factory=list)

    # --- timing ---
    started_at: float = Field(default_factory=time.time)
    completed_at: Optional[float] = None
    stage_timings_ms: Dict[str, float] = Field(default_factory=dict)

    # --- error envelope ---
    error: Optional[str] = None
    """Set if the pipeline aborted before completing all stages. The record
    is still emitted; downstream alerts watch for non-null error."""

    def to_jsonable(self) -> Dict[str, Any]:
        """Convenience for trace_event payloads."""
        return self.model_dump(mode="json")


class _DecisionRecorder:
    """Internal helper used by ContextManager to assemble a DecisionRecord
    incrementally. Not part of the public audit surface — only the
    DecisionRecord schema is.

    Usage::

        rec = _DecisionRecorder(session_id="x", query="...")
        with rec.stage("inference"):
            ...
        rec.record.inferred_context = ...
        ...
        rec.finalise()
        await genie.observability.trace_event(
            "audit.decision_record", rec.record.to_jsonable(), "cqs", corr_id
        )
    """

    def __init__(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_identity: Optional[Dict[str, Any]] = None,
    ):
        self.record = DecisionRecord(
            query=query,
            session_id=session_id,
            user_identity=user_identity,
        )
        self._stage_starts: Dict[str, float] = {}

    def stage(self, name: str) -> "_StageTimer":
        return _StageTimer(self, name)

    def _stage_started(self, name: str) -> None:
        self._stage_starts[name] = time.perf_counter()

    def _stage_ended(self, name: str) -> None:
        start = self._stage_starts.pop(name, None)
        if start is not None:
            self.record.stage_timings_ms[name] = (time.perf_counter() - start) * 1000.0

    def finalise(self) -> None:
        self.record.completed_at = time.time()


class _StageTimer:
    """Context-managerish helper. Used as ``with recorder.stage('x'): ...``
    Async usage is supported via async with."""

    def __init__(self, recorder: _DecisionRecorder, name: str):
        self._recorder = recorder
        self._name = name

    def __enter__(self) -> "_StageTimer":
        self._recorder._stage_started(self._name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._recorder._stage_ended(self._name)

    async def __aenter__(self) -> "_StageTimer":
        self._recorder._stage_started(self._name)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self._recorder._stage_ended(self._name)


def _preview(value: Any, max_chars: int = 500) -> str:
    """Truncated repr for audit previews. Never raises."""
    try:
        s = repr(value)
    except Exception as e:
        s = f"<unrepr-able: {type(value).__name__}: {e}>"
    if len(s) > max_chars:
        s = s[:max_chars] + f"...<truncated {len(s) - max_chars} chars>"
    return s


__all__ = [
    "DecisionRecord",
    "HITLApprovalEntry",
    "RankedRuleEntry",
    "ToolCallEntry",
    "_DecisionRecorder",
    "_StageTimer",
    "_preview",
]
