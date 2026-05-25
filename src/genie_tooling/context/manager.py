import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from .audit import DecisionRecord, RankedRuleEntry, _DecisionRecorder, _preview
from .protocols import (
    ContextInferencePlugin,
    ContextSourcePlugin,
    DerivationStrategyPlugin,
    FormulationStrategyPlugin,
    PredicateExtractorPlugin,
    RuleEnginePlugin,
)

if TYPE_CHECKING:
    from genie_tooling.genie import Genie

logger = logging.getLogger(__name__)


class ContextManager:
    """Internal orchestrator for the CQS-Engine pipeline."""

    def __init__(self, genie: "Genie", config: Dict[str, Any]):
        self._genie = genie
        self._config = config
        self._context_source: Optional[ContextSourcePlugin] = None
        self._inference_engine: Optional[ContextInferencePlugin] = None
        self._rule_engine: Optional[RuleEnginePlugin] = None
        self._predicate_extractor: Optional[PredicateExtractorPlugin] = None
        self._formulation_strategy: Optional[FormulationStrategyPlugin] = None
        # The DecisionRecord emitted by the most recent resolve_and_formulate
        # call. Exposed via ContextInterface.last_decision for testing,
        # debugging, and audit reconstruction. Long-lived audit consumers
        # should subscribe to the `audit.decision_record` tracer event
        # instead — this in-memory slot is a convenience, not the contract.
        self._last_decision: Optional[DecisionRecord] = None

    async def setup(self):
        """
        Loads and initializes all configured plugins for the CQS pipeline.
        This method is called by the bootstrap plugin.
        """
        async def _load_plugin(key: str, default_id: str, protocol: type) -> Optional[Any]:
            plugin_id = self._config.get(f"{key}_plugin_id", default_id)
            if not plugin_id:
                logger.error(f"ContextManager: No plugin ID configured for '{key}'.")
                return None

            config = self._config.get(f"{key}_config", {})

            instance = await self._genie.plugins.get_instance(plugin_id, config=config)
            if instance and isinstance(instance, protocol):
                logger.info(f"ContextManager: Loaded {protocol.__name__} '{plugin_id}'.")
                return instance

            logger.error(f"ContextManager: Failed to load {protocol.__name__} '{plugin_id}'.")
            return None

        self._context_source = await _load_plugin("context_source", "configurable_context_source_v1", ContextSourcePlugin)
        self._inference_engine = await _load_plugin("context_inference", "llm_context_inference_v1", ContextInferencePlugin)
        self._rule_engine = await _load_plugin("rule_engine", "filesystem_rule_engine_v1", RuleEnginePlugin)
        self._predicate_extractor = await _load_plugin("predicate_extractor", "llm_predicate_extractor_v1", PredicateExtractorPlugin)
        self._formulation_strategy = await _load_plugin("formulation_strategy", "llm_prompt_formulation_v1", FormulationStrategyPlugin)

        if self._rule_engine:
            # RuleEnginePlugin protocol declares load_rules(genie); both
            # bundled engines accept it. Plugins that don't need the facade
            # should accept and ignore it.
            await self._rule_engine.load_rules(genie=self._genie)

    async def _aggregate_constraints(self, ranked_rules: list[Tuple[Dict, float]]) -> Tuple[Dict, Dict]:
        C_D: Dict[str, Any] = {}
        C_F: Dict[str, Any] = {}
        if not ranked_rules:
            return C_D, C_F

        top_rule = ranked_rules[0][0]
        logger.info(f"Applying highest-ranked rule '{top_rule.get('rule_id')}' with score {ranked_rules[0][1]:.4f}.")

        for target, op, key, value in top_rule.get("actions", []):
            target_dict = C_D if target == "C_D" else C_F
            if op == "set":
                target_dict[key] = value
            elif op == "default" and key not in target_dict:
                target_dict[key] = value
            elif op == "add":
                if key not in target_dict:
                    target_dict[key] = []
                if isinstance(target_dict[key], list):
                    target_dict[key].append(value)

        return C_D, C_F

    async def _derivation_step(self, query: str, C_D: Dict[str, Any]) -> Dict[str, Any]:
        derivation_strategy_id = C_D.get("derivation_strategy_id", "generic_agent_derivation_v1")

        derivation_plugin = await self._genie.plugins.get_instance(derivation_strategy_id) # type: ignore

        if not derivation_plugin or not isinstance(derivation_plugin, DerivationStrategyPlugin):
            return {"status": "error", "error": f"Failed to load configured Derivation Strategy '{derivation_strategy_id}'."}

        return await derivation_plugin.derive(query=query, constraints=C_D, genie=self._genie)

    @property
    def last_decision(self) -> Optional[DecisionRecord]:
        """The DecisionRecord assembled by the most recent
        resolve_and_formulate. Test/debug convenience; production consumers
        should subscribe to the ``audit.decision_record`` trace event."""
        return self._last_decision

    async def resolve_and_formulate(
        self,
        query: str,
        session_id: Optional[str],
        user_identity: Optional[Dict[str, Any]] = None,
    ) -> Any:
        recorder = _DecisionRecorder(
            query=query, session_id=session_id, user_identity=user_identity
        )
        try:
            return await self._resolve_and_formulate_inner(query, session_id, recorder)
        except Exception as e:  # pragma: no cover - belt-and-suspenders
            recorder.record.error = f"{type(e).__name__}: {e}"
            raise
        finally:
            recorder.finalise()
            self._last_decision = recorder.record
            await self._emit_audit_event(recorder.record)

    async def _emit_audit_event(self, record: DecisionRecord) -> None:
        """Best-effort audit emission. Never fail the call because the
        tracer is unavailable — the in-memory `last_decision` is the
        backup for tests."""
        try:
            await self._genie.observability.trace_event(
                "audit.decision_record",
                record.to_jsonable(),
                "cqs.context_manager",
                record.decision_id,
            )
        except Exception:  # pragma: no cover
            logger.warning(
                "ContextManager: failed to emit audit.decision_record trace event",
                exc_info=True,
            )

    async def _resolve_and_formulate_inner(
        self,
        query: str,
        session_id: Optional[str],
        recorder: "_DecisionRecorder",
    ) -> Any:
        rec = recorder.record

        if not all(
            [
                self._context_source,
                self._inference_engine,
                self._rule_engine,
                self._predicate_extractor,
                self._formulation_strategy,
            ]
        ):
            logger.error("CQS-Engine is not fully configured. A required plugin is missing.")
            rec.error = "CQS-Engine is not fully configured. A required plugin is missing."
            return {"error": rec.error}

        # Stage 1: load context (history + profile)
        async with recorder.stage("context_load"):
            history_state = (
                await self._genie.conversation.load_state(session_id)
                if session_id
                else None
            )
            profile = await self._context_source.get_profile(session_id, self._genie)
        rec.profile = profile if isinstance(profile, dict) else {"value": profile}
        history_list = history_state.get("history", []) if history_state else []
        rec.history_length = len(history_list)
        raw_context = {"history": history_list, "profile": profile}

        # Stage 2: inference
        async with recorder.stage("inference"):
            inferred_context = await self._inference_engine.infer_context_properties(
                raw_context, self._genie
            )
        rec.inferred_context = inferred_context if isinstance(inferred_context, dict) else {}

        # Stage 3: predicate extraction
        rec.predicate_extractor_id = getattr(self._predicate_extractor, "plugin_id", None)
        async with recorder.stage("predicate_extract"):
            predicate = await self._predicate_extractor.extract(query, self._genie)
        rec.predicate = predicate

        # Stage 4: rule evaluation
        rec.rule_engine_id = getattr(self._rule_engine, "plugin_id", None)
        async with recorder.stage("rule_evaluate"):
            ranked_rules = await self._rule_engine.evaluate(
                inferred_context, predicate, genie=self._genie
            )
        rec.ranked_rules = [
            RankedRuleEntry(
                rule_id=str(rule.get("rule_id", "<no id>")),
                score=float(score),
                priority=rule.get("priority"),
            )
            for rule, score in ranked_rules
        ]
        if rec.ranked_rules:
            rec.winning_rule_id = rec.ranked_rules[0].rule_id

        # Stage 5: constraint aggregation
        async with recorder.stage("aggregate"):
            C_D, C_F = await self._aggregate_constraints(ranked_rules)
        rec.c_d = dict(C_D)
        rec.c_f = dict(C_F)
        rec.derivation_strategy_id = C_D.get(
            "derivation_strategy_id", "generic_agent_derivation_v1"
        )

        # Stage 6: derivation
        async with recorder.stage("derivation"):
            raw_data = await self._derivation_step(query, C_D)
        logger.debug(f"Derivation Result (Raw Data): {raw_data}")
        if isinstance(raw_data, dict):
            rec.derivation_status = raw_data.get("status")
            if raw_data.get("status") == "error":
                rec.derivation_error = str(raw_data.get("error", ""))
            else:
                rec.derivation_result_preview = _preview(raw_data.get("result"))
        else:
            rec.derivation_result_preview = _preview(raw_data)

        # Stage 7: formulation
        rec.formulation_strategy_id = getattr(self._formulation_strategy, "plugin_id", None)
        rec.formulation_template_id = C_F.get("prompt_template_id", "default_formulation_prompt")
        # The translator is also called inside the formulation plugin; we
        # call it here too so the audit record contains the exact
        # instruction text the LLM saw, without depending on plugin
        # internals.
        from .constraints import formulation_constraints_to_instructions
        rec.formulation_constraints_text = formulation_constraints_to_instructions(C_F)
        async with recorder.stage("formulation"):
            final_response = await self._formulation_strategy.formulate(
                query=query, raw_data=raw_data, constraints=C_F, genie=self._genie
            )
        rec.final_response = (
            final_response if isinstance(final_response, str) else _preview(final_response)
        )

        return final_response
