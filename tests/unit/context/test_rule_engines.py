from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from genie_tooling.context.plugins.rule_engines.filesystem_engine import (
    FileSystemRuleEnginePlugin,
    _default_rules_path,
)
from genie_tooling.context.plugins.rule_engines.vectordb_engine import (
    VectorDBRuleEnginePlugin,
)


@pytest.mark.asyncio()
async def test_filesystem_engine_default_path_resolves_to_bundled_rules():
    """When no rules_path is configured, the default points at the bundled rules
    shipped inside the installed package (via importlib.resources), not the CWD."""
    engine = FileSystemRuleEnginePlugin()
    await engine.setup(config=None)
    assert engine._rules_path == _default_rules_path()
    assert engine._rules_path.is_dir(), (
        f"Default rules path doesn't exist on disk: {engine._rules_path}"
    )
    # The bundled rules dir must actually contain rules.
    yml_files = list(engine._rules_path.glob("*.yml"))
    assert yml_files, f"No bundled rules found at {engine._rules_path}"

@pytest.mark.asyncio()
async def test_filesystem_engine_loads_and_evaluates(tmp_path):
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    rule_content = {
        "rule_id": "TEST_FS_RULE", "predicate": "predicate_is", "priority": 10,
        "conditions": [["AudienceProfile.expertise", "==", "layperson"]],
        "actions": [["C_F", "set", "tone", "casual"]]
    }
    (rules_dir / "test_rule.yml").write_text(yaml.dump(rule_content))
    engine = FileSystemRuleEnginePlugin()
    await engine.setup(config={"rules_path": str(rules_dir)})
    await engine.load_rules()
    matching_context = {"AudienceProfile": {"expertise": "layperson"}}
    result = await engine.evaluate(matching_context, "predicate_is")
    non_matching_context = {"AudienceProfile": {"expertise": "expert"}}
    result_none = await engine.evaluate(non_matching_context, "predicate_is")
    assert len(result) == 1
    assert result[0][0]["rule_id"] == "TEST_FS_RULE"
    assert result[0][1] == 1.0
    assert result[0][0]["actions"][0][3] == "casual"
    assert len(result_none) == 0

@pytest.mark.asyncio()
async def test_vectordb_engine_evaluation_flow():
    mock_genie = MagicMock()
    mock_genie.rag.search = AsyncMock()
    mock_retrieved_chunk = MagicMock()
    mock_retrieved_chunk.metadata = {"rule_id": "SIMILAR_RULE_1"}
    mock_retrieved_chunk.score = 0.85
    mock_genie.rag.search.return_value = [mock_retrieved_chunk]
    engine = VectorDBRuleEnginePlugin()
    await engine.setup(config={"genie_facade": mock_genie, "rules_path": "./dummy_path"})
    engine._rules_in_memory = {
        "SIMILAR_RULE_1": {
            "rule_id": "SIMILAR_RULE_1", "predicate": "*", "priority": 1,
            "conditions": [], "actions": [["C_D", "set", "precision", "high"]]
        }
    }
    result = await engine.evaluate({"some": "context"}, "some_predicate", genie=mock_genie)
    mock_genie.rag.search.assert_called_once()
    call_args = mock_genie.rag.search.call_args
    assert "The user's action is about 'some_predicate'." in call_args.kwargs["query"]
    assert len(result) == 1
    assert result[0][0]["rule_id"] == "SIMILAR_RULE_1"
