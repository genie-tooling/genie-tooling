"""
End-to-end test of the cqs pipeline against REAL plugins. No mocks for the
deterministic path (heuristic predicate extraction, filesystem rule engine,
generic tool derivation, calculator tool). Only the LLM-only stages (context
inference, formulation) are mocked with minimal fakes — because they require
an LLM and we don't want this test to depend on a live LLM provider.

This complements the unit tests by exercising the actual code paths of the
deterministic core, catching the class of bugs the mock-heavy tests can't see.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from genie_tooling.context.manager import ContextManager
from genie_tooling.context.plugins.context_sources.configurable_source import (
    ConfigurableContextSourcePlugin,
)
from genie_tooling.context.plugins.derivation.generic_tool_derivation import (
    GenericToolDerivationPlugin,
)
from genie_tooling.context.plugins.predicate_extractors.heuristic_extractor import (
    HeuristicPredicateExtractorPlugin,
)
from genie_tooling.context.plugins.rule_engines.filesystem_engine import (
    FileSystemRuleEnginePlugin,
)
from genie_tooling.context.protocols import (
    ContextInferencePlugin,
    FormulationStrategyPlugin,
)


class _FakeInferencePlugin(ContextInferencePlugin):
    """A deterministic stand-in for the LLM-based inference plugin.

    Returns whatever profile was passed in as the AudienceProfile, plus
    a trivial DiscourseTopic. Lets us drive rule conditions deterministically.
    """

    plugin_id: str = "fake_inference_v1"
    description: str = "test-only fake inference"

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        pass

    async def teardown(self) -> None:
        pass

    async def infer_context_properties(
        self, raw_context: Dict[str, Any], genie: Any
    ) -> Dict[str, Any]:
        return {
            "AudienceProfile": raw_context.get("profile", {}),
            "DiscourseTopic": {"primary": 1.0},
        }


class _FakeFormulationPlugin(FormulationStrategyPlugin):
    """Passthrough formulation that returns the raw derivation result as-is."""

    plugin_id: str = "fake_formulation_v1"
    description: str = "test-only passthrough formulation"

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        pass

    async def teardown(self) -> None:
        pass

    async def formulate(
        self, query: str, raw_data: Any, constraints: Dict[str, Any], genie: Any
    ) -> str:
        return f"answer={raw_data!r} constraints={constraints!r}"


def _write_test_rule(tmp_path, **overrides) -> None:
    """Write a single test rule that calls calculator_tool deterministically."""
    rule = {
        "rule_id": "TEST_CALC",
        "predicate": "predicate_calculate",
        "priority": 1,
        "conditions": [["AudienceProfile.intent", "==", "computation"]],
        "actions": [
            ["C_D", "set", "derivation_strategy_id", "generic_tool_derivation_v1"],
            ["C_D", "set", "tool_id", "calculator_tool"],
            ["C_D", "set", "params", {"num1": 2.0, "num2": 3.0, "operation": "add"}],
            ["C_F", "set", "tone", "concise"],
        ],
    }
    rule.update(overrides)
    (tmp_path / "test_calc.yml").write_text(yaml.dump(rule))


@pytest.mark.asyncio()
async def test_e2e_real_pipeline_routes_calc_query_to_calculator_tool(tmp_path):
    """
    Full deterministic pipeline:

        heuristic predicate extractor -> filesystem rule engine (test rule) ->
        generic_tool_derivation -> calculator_tool -> fake formulation

    Asserts the calculator actually ran and the constraint set propagated.
    """
    _write_test_rule(tmp_path)

    # Real plugins
    source = ConfigurableContextSourcePlugin()
    await source.setup(config={"default_profile": {"intent": "computation"}})

    inference = _FakeInferencePlugin()
    await inference.setup()

    predicate_extractor = HeuristicPredicateExtractorPlugin()
    await predicate_extractor.setup()

    rule_engine = FileSystemRuleEnginePlugin()
    await rule_engine.setup(config={"rules_path": str(tmp_path)})

    derivation = GenericToolDerivationPlugin()
    await derivation.setup()

    formulation = _FakeFormulationPlugin()
    await formulation.setup()

    # Mock genie whose execute_tool runs a real calculator.
    mock_genie = MagicMock()

    # Plug calculator into mock_genie.execute_tool by composing a real calculator
    # via the registered plugin without spinning up a Genie.create().
    from genie_tooling.tools.impl.calculator import CalculatorTool

    calc = CalculatorTool()
    await calc.setup()

    async def _execute_tool(tool_id: str, **params: Any) -> Any:
        if tool_id != "calculator_tool":
            raise AssertionError(f"unexpected tool_id {tool_id}")
        return await calc.execute(params=params, key_provider=MagicMock(), context=None)

    mock_genie.execute_tool = AsyncMock(side_effect=_execute_tool)
    mock_genie.conversation = MagicMock()
    mock_genie.conversation.load_state = AsyncMock(return_value={"history": []})

    # Plugin resolution: only one plugin id is needed at derivation time —
    # generic_tool_derivation_v1. ContextManager calls genie.plugins.get_instance
    # for the derivation strategy. Everything else is plumbed via slot assignment.
    mock_genie.plugins = MagicMock()

    async def _get_instance(plugin_id: str, config: Optional[Dict] = None) -> Any:
        if plugin_id == "generic_tool_derivation_v1":
            return derivation
        return None

    mock_genie.plugins.get_instance = AsyncMock(side_effect=_get_instance)

    manager = ContextManager(genie=mock_genie, config={})
    # Bypass setup() (which would try to load all the plugins via the plugin
    # manager) and inject the real instances directly.
    manager._context_source = source
    manager._inference_engine = inference
    manager._rule_engine = rule_engine
    manager._predicate_extractor = predicate_extractor
    manager._formulation_strategy = formulation
    await rule_engine.load_rules()

    # ACT
    final_response = await manager.resolve_and_formulate(
        query="calculate 2 plus 3", session_id="session-e2e"
    )

    # ASSERT — the calculator was actually invoked with the rule's params.
    mock_genie.execute_tool.assert_awaited_once()
    call = mock_genie.execute_tool.await_args
    assert call.args[0] == "calculator_tool"
    assert call.kwargs == {"num1": 2.0, "num2": 3.0, "operation": "add"}

    # The fake formulation echoes the derivation result; calculator returns
    # {"result": 5.0} per its contract.
    assert "5.0" in final_response or "5" in final_response, (
        f"expected calculator result in final response, got: {final_response!r}"
    )
    # C_F constraints should have made it through.
    assert "concise" in final_response


@pytest.mark.asyncio()
async def test_e2e_pipeline_returns_error_when_tool_missing(tmp_path):
    """If a rule references a tool that doesn't exist, derivation reports an
    error instead of crashing the pipeline."""
    _write_test_rule(
        tmp_path,
        actions=[
            ["C_D", "set", "derivation_strategy_id", "generic_tool_derivation_v1"],
            ["C_D", "set", "tool_id", "does_not_exist_tool"],
            ["C_D", "set", "params", {}],
        ],
    )

    source = ConfigurableContextSourcePlugin()
    await source.setup(config={"default_profile": {"intent": "computation"}})

    rule_engine = FileSystemRuleEnginePlugin()
    # This test deliberately routes to a non-existent tool to exercise the
    # error path inside the derivation. The A2 load-time validator would
    # reject the rule before that — disable strict_validation here so the
    # rule survives load and the test can verify runtime error handling.
    await rule_engine.setup(
        config={"rules_path": str(tmp_path), "strict_validation": False}
    )
    await rule_engine.load_rules()

    derivation = GenericToolDerivationPlugin()
    await derivation.setup()

    mock_genie = MagicMock()
    mock_genie.execute_tool = AsyncMock(
        side_effect=RuntimeError("tool 'does_not_exist_tool' not registered")
    )
    mock_genie.conversation = MagicMock()
    mock_genie.conversation.load_state = AsyncMock(return_value={"history": []})

    inference = _FakeInferencePlugin()
    await inference.setup()
    predicate_extractor = HeuristicPredicateExtractorPlugin()
    await predicate_extractor.setup()
    formulation = _FakeFormulationPlugin()
    await formulation.setup()

    mock_genie.plugins = MagicMock()
    mock_genie.plugins.get_instance = AsyncMock(
        side_effect=lambda pid, **kw: derivation if pid == "generic_tool_derivation_v1" else None
    )

    manager = ContextManager(genie=mock_genie, config={})
    manager._context_source = source
    manager._inference_engine = inference
    manager._rule_engine = rule_engine
    manager._predicate_extractor = predicate_extractor
    manager._formulation_strategy = formulation

    final_response = await manager.resolve_and_formulate(
        query="calculate something", session_id="session-e2e-err"
    )

    # Derivation returns {"status": "error", ...}; fake formulation echoes it.
    assert "error" in final_response.lower()
    assert "does_not_exist_tool" in final_response
