from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock

import pytest

from genie_tooling.core.types import Plugin
from genie_tooling.context.bootstrap import CqsEngineBootstrapPlugin
from genie_tooling.context.protocols import (
    ContextInferencePlugin,
    ContextSourcePlugin,
    RuleEnginePlugin,
)

# Mock implementations of core genie-tooling plugins our extension depends on
class MockEmbeddingGenerator(Plugin):
    plugin_id = "mock_embedder_v1"
    async def embed(self, *args, **kwargs) -> List[Tuple[Any, List[float]]]:
        return [("mock_chunk", [0.1, 0.2, 0.3])]
    async def setup(self, *args, **kwargs): pass
    async def teardown(self, *args, **kwargs): pass

class MockVectorStore(Plugin):
    plugin_id = "mock_vector_store_v1"
    search = AsyncMock(return_value=[])
    add = AsyncMock(return_value={"added_count": 1})
    async def setup(self, *args, **kwargs): pass
    async def teardown(self, *args, **kwargs): pass

# Mock implementations of our new CQS plugin types for testing the manager
class MockContextSource(ContextSourcePlugin):
    plugin_id = "mock_context_source_v1"
    async def get_profile(self, session_id: str, genie: Any) -> Dict[str, Any]:
        return {"expertise": "layperson", "location": "office"}

class MockContextInference(ContextInferencePlugin):
    plugin_id = "mock_context_inference_v1"
    async def infer_context_properties(self, raw_context: Dict[str, Any], genie: Any) -> Dict[str, Any]:
        return {
            "AudienceProfile": {"expertise": "layperson"},
            "DiscourseTopic": {"primary": "general_knowledge"}
        }

class MockRuleEngine(RuleEnginePlugin):
    plugin_id = "mock_rule_engine_v1"
    load_rules = AsyncMock(return_value=True)
    async def evaluate(self, inferred_context: Dict[str, Any], query_predicate: str) -> List[Tuple[Dict, float]]:
        mock_rule = {
            "rule_id": "TEST_RULE", "predicate": "*", "priority": 1,
            "conditions": [],
            "actions": [["C_F", "set", "formality", "informal"]]
        }
        return [(mock_rule, 1.0)]

# Mock PluginManager to control plugin instantiation
class MockPluginManager:
    def __init__(self, plugins: Dict[str, Plugin]):
        self._plugins = plugins

    async def get_plugin_instance(self, plugin_id: str, **kwargs: Any) -> Optional[Plugin]:
        return self._plugins.get(plugin_id)

@pytest.fixture
def mock_plugin_manager_fixture():
    """Provides a pytest fixture for a pre-populated MockPluginManager."""
    plugins = {
        "cqs_engine_bootstrap_v1": CqsEngineBootstrapPlugin(),
        "mock_embedder_v1": MockEmbeddingGenerator(),
        "mock_vector_store_v1": MockVectorStore(),
        "mock_context_source_v1": MockContextSource(),
        "mock_context_inference_v1": MockContextInference(),
        "mock_rule_engine_v1": MockRuleEngine(),
    }
    return MockPluginManager(plugins)
