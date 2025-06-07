from typing import Any, Dict, Optional
from unittest.mock import AsyncMock

import pytest

try:
    import numpy as np
except ImportError:
    np = None

from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import Plugin
from genie_tooling.embedding_generators.abc import EmbeddingGeneratorPlugin
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tool_lookup_providers.impl.embedding_similarity import (
    EmbeddingSimilarityLookupProvider,
)

PROVIDER_LOGGER_NAME = "genie_tooling.tool_lookup_providers.impl.embedding_similarity"

class MockEmbedderForLookup(EmbeddingGeneratorPlugin, Plugin):
    _plugin_id_value: str = "mock_embedder_for_lookup_v1"
    description: str = "Mock embedder for testing lookup"
    setup_config_received: Optional[Dict[str, Any]] = None
    @property
    def plugin_id(self) -> str: return self._plugin_id_value
    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.setup_config_received = config
    async def embed(self, chunks, config=None):
        if False: yield

@pytest.fixture
def mock_plugin_manager_for_es_lookup(mocker) -> PluginManager:
    pm = mocker.MagicMock(spec=PluginManager)
    pm.get_plugin_instance = AsyncMock()
    return pm

@pytest.fixture
def mock_key_provider_for_es_lookup(mocker) -> KeyProvider:
    kp = mocker.AsyncMock(spec=KeyProvider)
    return kp

@pytest.mark.asyncio
async def test_es_setup_direct_vs_config_params(
    mock_plugin_manager_for_es_lookup: PluginManager,
    mock_key_provider_for_es_lookup: KeyProvider
):
    provider = EmbeddingSimilarityLookupProvider()
    mock_embedder = MockEmbedderForLookup()
    mock_vs = AsyncMock()
    type(mock_vs).plugin_id = "mock_vs_for_lookup_v1"

    async def get_instance_side_effect_vs(plugin_id_req, config=None, **kwargs):
        if plugin_id_req == "custom_embed_id":
            await mock_embedder.setup(config)
            return mock_embedder
        if plugin_id_req == "mock_vs_for_lookup_v1":
            await mock_vs.setup(config)
            return mock_vs
        return None
    mock_plugin_manager_for_es_lookup.get_plugin_instance.side_effect = get_instance_side_effect_vs

    await provider.setup(config={
        "plugin_manager": mock_plugin_manager_for_es_lookup,
        "key_provider": mock_key_provider_for_es_lookup,
        "embedder_id": "custom_embed_id",
        "embedder_config": {"model": "embed_model"},
        "vector_store_id": "mock_vs_for_lookup_v1",
        "tool_embeddings_collection_name": "my_tools_collection",
        "tool_embeddings_path": "/custom/tools/db_path",
        "vector_store_config": {"specific_vs_param": "value", "path": "ignored_path_due_to_direct"}
    })

    assert provider._embedder is mock_embedder
    assert mock_embedder.setup_config_received is not None
    assert mock_embedder.setup_config_received.get("model") == "embed_model"
    assert mock_embedder.setup_config_received.get("key_provider") is mock_key_provider_for_es_lookup
