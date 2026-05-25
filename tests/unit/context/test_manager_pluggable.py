from unittest.mock import AsyncMock, MagicMock

import pytest
from genie_tooling.context.manager import ContextManager
from genie_tooling.context.protocols import (
    ContextInferencePlugin,
    ContextSourcePlugin,
    DerivationStrategyPlugin,
    FormulationStrategyPlugin,
    PredicateExtractorPlugin,
    RuleEnginePlugin,
)


@pytest.mark.asyncio()
async def test_manager_uses_pluggable_pipeline(monkeypatch):
    # ARRANGE
    mock_genie = MagicMock()
    mock_genie.conversation.load_state = AsyncMock(return_value={"history": []})

    # spec= makes isinstance(mock, Protocol) succeed inside ContextManager.setup.
    mock_source = MagicMock(spec=ContextSourcePlugin)
    mock_source.get_profile = AsyncMock(return_value={"expertise": "expert"})

    mock_inference = MagicMock(spec=ContextInferencePlugin)
    mock_inference.infer_context_properties = AsyncMock(
        return_value={"AudienceProfile": {"expertise": "expert"}}
    )

    mock_predicate_extractor = MagicMock(spec=PredicateExtractorPlugin)
    mock_predicate_extractor.extract = AsyncMock(return_value="predicate_what")

    mock_rule_engine = MagicMock(spec=RuleEnginePlugin)
    mock_rule_engine.load_rules = AsyncMock(return_value=True)
    mock_rule_engine.evaluate = AsyncMock(
        return_value=[
            (
                {
                    "rule_id": "EXPERT_RULE",
                    "actions": [
                        ["C_D", "set", "derivation_strategy_id", "karta_lookup_derivation_v1"]
                    ],
                },
                1.0,
            )
        ]
    )

    mock_derivation_strategy = MagicMock(spec=DerivationStrategyPlugin)
    mock_derivation_strategy.derive = AsyncMock(
        return_value={"status": "success", "result": "Paris"}
    )

    mock_formulation_strategy = MagicMock(spec=FormulationStrategyPlugin)
    mock_formulation_strategy.formulate = AsyncMock(
        return_value="The final answer is Paris."
    )

    # Create a map of plugin IDs to their mock instances
    plugin_map = {
        "in_memory_user_profile_source_v1": mock_source,
        "llm_context_inference_v1": mock_inference,
        "llm_predicate_extractor_v1": mock_predicate_extractor,
        "filesystem_rule_engine_v1": mock_rule_engine,
        "karta_lookup_derivation_v1": mock_derivation_strategy,
        "llm_prompt_formulation_v1": mock_formulation_strategy,
    }

    # ContextManager resolves plugins via genie.plugins.get_instance (public facade).
    mock_genie.plugins = MagicMock()
    mock_genie.plugins.get_instance = AsyncMock(
        side_effect=lambda plugin_id, **kwargs: plugin_map.get(plugin_id)
    )

    # Instantiate the manager with a config that points to our mocks
    manager_config = {
        "context_source_plugin_id": "in_memory_user_profile_source_v1",
        "context_inference_plugin_id": "llm_context_inference_v1",
        "predicate_extractor_plugin_id": "llm_predicate_extractor_v1",
        "rule_engine_plugin_id": "filesystem_rule_engine_v1",
        "formulation_strategy_plugin_id": "llm_prompt_formulation_v1"
    }
    manager = ContextManager(genie=mock_genie, config=manager_config)

    # --- THIS IS THE FIX ---
    # Call the setup method, which will use the mocked plugin manager to load our plugins.
    await manager.setup()

    # ACT
    final_response = await manager.resolve_and_formulate(
        "what is the capital of france", "session123"
    )

    # ASSERT
    mock_source.get_profile.assert_called_once() # This will now pass
    mock_inference.infer_context_properties.assert_called_once()
    mock_predicate_extractor.extract.assert_called_once()
    mock_rule_engine.load_rules.assert_called_once()
    mock_rule_engine.evaluate.assert_called_once()

    # We now assert on the derivation strategy that should be loaded by the manager
    # based on the rule evaluation.
    mock_derivation_strategy.derive.assert_called_once()
    derive_call_args = mock_derivation_strategy.derive.call_args
    assert derive_call_args.kwargs["query"] == "what is the capital of france"
    assert derive_call_args.kwargs["constraints"] == {
        "derivation_strategy_id": "karta_lookup_derivation_v1"
    }

    mock_formulation_strategy.formulate.assert_called_once()
    assert final_response == "The final answer is Paris."
