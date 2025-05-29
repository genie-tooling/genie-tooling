import logging
from typing import Any, AsyncIterable, Dict, List, Optional, cast
from unittest.mock import AsyncMock, patch

import pytest

try:
    import numpy as np
except ImportError:
    np = None

from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import Chunk, EmbeddingVector, Plugin
from genie_tooling.tool_lookup_providers.impl.embedding_similarity import (
    EmbeddingSimilarityLookupProvider,
)
from genie_tooling.tool_lookup_providers.impl.embedding_similarity import logger as es_logger
from genie_tooling.embedding_generators.abc import EmbeddingGeneratorPlugin
from genie_tooling.lookup.types import RankedToolResult


class MockEmbedderForLookup(EmbeddingGeneratorPlugin, Plugin):
    _plugin_id_value: str
    _fixed_embedding: Optional[List[float]] = None
    _embeddings_map: Optional[Dict[str, List[float]]] = None
    _fail_on_embed: bool = False
    teardown_called: bool = False

    def __init__(self, plugin_id_val: str = "mock_embedder_for_lookup_v1"):
        self._plugin_id_value = plugin_id_val
        self.description: str = "Mock embedder for testing lookup"

    @property
    def plugin_id(self) -> str: return self._plugin_id_value

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.teardown_called = False; pass
    def set_fixed_embedding(self, embedding: List[float]): self._fixed_embedding = embedding; self._embeddings_map = None
    def set_embeddings_map(self, embeddings_map: Dict[str, List[float]]): self._embeddings_map = embeddings_map; self._fixed_embedding = None
    def set_fail_on_embed(self, fail: bool): self._fail_on_embed = fail
    async def embed(self, chunks: AsyncIterable[Chunk], config: Optional[Dict[str, Any]] = None) -> AsyncIterable[tuple[Chunk, EmbeddingVector]]:
        if self._fail_on_embed: raise RuntimeError("Simulated embedder failure")
        async for chunk_item in chunks:
            if self._embeddings_map and chunk_item.content in self._embeddings_map:
                yield chunk_item, self._embeddings_map[chunk_item.content]
            elif self._fixed_embedding:
                yield chunk_item, self._fixed_embedding
            else:
                dim = (config or {}).get("expected_dim", 3)
                yield chunk_item, [0.1] * dim
    async def teardown(self) -> None: self.teardown_called = True

@pytest.fixture
def mock_plugin_manager_for_es_lookup(mocker) -> PluginManager:
    pm = mocker.MagicMock(spec=PluginManager); pm.get_plugin_instance = AsyncMock()
    return pm

@pytest.fixture
async def es_lookup_provider(mock_plugin_manager_for_es_lookup: PluginManager) -> EmbeddingSimilarityLookupProvider:
    provider = EmbeddingSimilarityLookupProvider()
    mock_embedder = MockEmbedderForLookup()
    mock_plugin_manager_for_es_lookup.get_plugin_instance.return_value = mock_embedder
    await provider.setup(config={"plugin_manager": mock_plugin_manager_for_es_lookup})
    return provider

@pytest.mark.asyncio
async def test_es_setup_success_default_embedder(mock_plugin_manager_for_es_lookup: PluginManager):
    es_lookup_provider_inst = EmbeddingSimilarityLookupProvider()
    mock_embedder = MockEmbedderForLookup(plugin_id_val=EmbeddingSimilarityLookupProvider.DEFAULT_EMBEDDER_ID)
    mock_plugin_manager_for_es_lookup.get_plugin_instance.return_value = mock_embedder
    await es_lookup_provider_inst.setup(config={"plugin_manager": mock_plugin_manager_for_es_lookup})
    assert es_lookup_provider_inst._embedder is mock_embedder
    mock_plugin_manager_for_es_lookup.get_plugin_instance.assert_awaited_once_with(
        EmbeddingSimilarityLookupProvider.DEFAULT_EMBEDDER_ID, config={}
    )

@pytest.mark.asyncio
async def test_es_setup_success_custom_embedder(mock_plugin_manager_for_es_lookup: PluginManager):
    es_lookup_provider_inst = EmbeddingSimilarityLookupProvider()
    custom_embedder_id = "custom_embed_v1"
    mock_embedder = MockEmbedderForLookup(plugin_id_val=custom_embedder_id)
    mock_plugin_manager_for_es_lookup.get_plugin_instance.return_value = mock_embedder
    custom_embedder_config = {"model": "test_model"}
    await es_lookup_provider_inst.setup(config={"plugin_manager": mock_plugin_manager_for_es_lookup, "embedder_id": custom_embedder_id, "embedder_config": custom_embedder_config})
    assert es_lookup_provider_inst._embedder is mock_embedder
    mock_plugin_manager_for_es_lookup.get_plugin_instance.assert_awaited_once_with(custom_embedder_id, config=custom_embedder_config)

@pytest.mark.skipif(np is None, reason="NumPy not available")
@pytest.mark.asyncio
async def test_es_index_tools_successful(es_lookup_provider: EmbeddingSimilarityLookupProvider):
    actual_provider = await es_lookup_provider
    cast(MockEmbedderForLookup, actual_provider._embedder).set_embeddings_map({"Tool A desc": [1.0, 0.0, 0.0], "Tool B desc": [0.0, 1.0, 0.0]})
    tools_data = [{"identifier": "tool_a", "lookup_text_representation": "Tool A desc"}, {"identifier": "tool_b", "lookup_text_representation": "Tool B desc"}]
    await actual_provider.index_tools(tools_data)
    assert actual_provider._indexed_tool_embeddings is not None
    assert actual_provider._indexed_tool_embeddings.shape == (2, 3)
    assert len(actual_provider._indexed_tool_data_list) == 2
    assert actual_provider._indexed_tool_data_list[0]["identifier"] == "tool_a"

@pytest.mark.asyncio
async def test_es_find_tools_empty_query(es_lookup_provider: EmbeddingSimilarityLookupProvider, caplog: pytest.LogCaptureFixture):
    actual_provider = await es_lookup_provider
    caplog.set_level(logging.DEBUG, logger="genie_tooling.tool_lookup_providers.impl.embedding_similarity")
    cast(MockEmbedderForLookup, actual_provider._embedder).set_fixed_embedding([0.1,0.2,0.3])
    await actual_provider.index_tools([{"identifier":"t1", "lookup_text_representation":"some text"}])
    results = await actual_provider.find_tools("")
    assert results == []
    assert "Empty query provided" in caplog.text and actual_provider.plugin_id in caplog.text
    caplog.clear()
    results_whitespace = await actual_provider.find_tools("   ")
    assert results_whitespace == []
    assert "Empty query provided" in caplog.text and actual_provider.plugin_id in caplog.text
