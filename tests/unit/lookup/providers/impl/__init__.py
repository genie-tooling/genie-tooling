### tests/unit/lookup/providers/impl/test_embedding_similarity_lookup.py
"""Unit tests for EmbeddingSimilarityLookupProvider."""
import logging
from typing import Any, AsyncIterable, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

try:
    import numpy as np
except ImportError:
    np = None

from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import Chunk, EmbeddingVector, Plugin
from genie_tooling.lookup.providers.impl.embedding_similarity import (
    EmbeddingSimilarityLookupProvider,
)
from genie_tooling.rag.plugins.abc import EmbeddingGeneratorPlugin


# --- Mocks ---
class MockEmbedderForLookup(EmbeddingGeneratorPlugin, Plugin):
    plugin_id = "mock_embedder_for_lookup_v1"
    description = "Mock embedder for testing lookup"
    _fixed_embedding: Optional[List[float]] = None
    _embeddings_map: Optional[Dict[str, List[float]]] = None # content -> embedding
    _fail_on_embed: bool = False
    teardown_called: bool = False

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.teardown_called = False # Reset for each setup
        pass

    def set_fixed_embedding(self, embedding: List[float]):
        self._fixed_embedding = embedding
        self._embeddings_map = None

    def set_embeddings_map(self, embeddings_map: Dict[str, List[float]]):
        self._embeddings_map = embeddings_map
        self._fixed_embedding = None

    def set_fail_on_embed(self, fail: bool):
        self._fail_on_embed = fail

    async def embed(self, chunks: AsyncIterable[Chunk], config: Optional[Dict[str, Any]] = None) -> AsyncIterable[tuple[Chunk, EmbeddingVector]]:
        if self._fail_on_embed:
            raise RuntimeError("Simulated embedder failure")

        async for chunk in chunks:
            if self._embeddings_map and chunk.content in self._embeddings_map:
                yield chunk, self._embeddings_map[chunk.content]
            elif self._fixed_embedding:
                yield chunk, self._fixed_embedding
            else: # Default behavior if no specific map or fixed embedding
                dim = (config or {}).get("expected_dim", 3) # Allow test to hint dimension
                yield chunk, [0.1] * dim # Generic embedding

    async def teardown(self) -> None:
        self.teardown_called = True


@pytest.fixture
def mock_plugin_manager_for_es_lookup(mocker) -> PluginManager:
    pm = mocker.MagicMock(spec=PluginManager)
    pm.get_plugin_instance = AsyncMock()
    return pm

@pytest.fixture
async def es_lookup_provider(mock_plugin_manager_for_es_lookup: PluginManager) -> EmbeddingSimilarityLookupProvider:
    provider = EmbeddingSimilarityLookupProvider()
    # Default setup behavior relies on _plugin_manager being set by the provider's setup
    return provider

# --- Test Cases ---

@pytest.mark.asyncio
async def test_es_setup_success_default_embedder(
    es_lookup_provider: EmbeddingSimilarityLookupProvider,
    mock_plugin_manager_for_es_lookup: PluginManager
):
    mock_embedder = MockEmbedderForLookup()
    mock_plugin_manager_for_es_lookup.get_plugin_instance.return_value = mock_embedder

    await es_lookup_provider.setup(config={"plugin_manager": mock_plugin_manager_for_es_lookup})

    assert es_lookup_provider._embedder is mock_embedder
    mock_plugin_manager_for_es_lookup.get_plugin_instance.assert_awaited_once_with(
        EmbeddingSimilarityLookupProvider.DEFAULT_EMBEDDER_ID, config={}
    )

@pytest.mark.asyncio
async def test_es_setup_success_custom_embedder(
    es_lookup_provider: EmbeddingSimilarityLookupProvider,
    mock_plugin_manager_for_es_lookup: PluginManager
):
    mock_embedder = MockEmbedderForLookup()
    mock_plugin_manager_for_es_lookup.get_plugin_instance.return_value = mock_embedder
    custom_embedder_id = "custom_embed_v1"
    custom_embedder_config = {"model": "test_model"}

    await es_lookup_provider.setup(config={
        "plugin_manager": mock_plugin_manager_for_es_lookup,
        "embedder_id": custom_embedder_id,
        "embedder_config": custom_embedder_config
    })

    assert es_lookup_provider._embedder is mock_embedder
    mock_plugin_manager_for_es_lookup.get_plugin_instance.assert_awaited_once_with(
        custom_embedder_id, config=custom_embedder_config
    )

@pytest.mark.asyncio
async def test_es_setup_no_plugin_manager(
    es_lookup_provider: EmbeddingSimilarityLookupProvider, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR)
    await es_lookup_provider.setup(config={}) # Missing plugin_manager
    assert es_lookup_provider._embedder is None
    assert f"{es_lookup_provider.plugin_id} Error: PluginManager not provided in config." in caplog.text

@pytest.mark.asyncio
async def test_es_setup_embedder_load_failure(
    es_lookup_provider: EmbeddingSimilarityLookupProvider,
    mock_plugin_manager_for_es_lookup: PluginManager, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR)
    mock_plugin_manager_for_es_lookup.get_plugin_instance.return_value = None # Simulate embedder not found

    await es_lookup_provider.setup(config={"plugin_manager": mock_plugin_manager_for_es_lookup})
    assert es_lookup_provider._embedder is None
    assert f"{es_lookup_provider.plugin_id} Error: Embedder plugin '{EmbeddingSimilarityLookupProvider.DEFAULT_EMBEDDER_ID}' not found" in caplog.text

@pytest.mark.skipif(np is None, reason="NumPy not available")
@pytest.mark.asyncio
async def test_es_index_tools_successful(
    es_lookup_provider: EmbeddingSimilarityLookupProvider,
    mock_plugin_manager_for_es_lookup: PluginManager
):
    mock_embedder = MockEmbedderForLookup()
    mock_embedder.set_embeddings_map({
        "Tool A desc": [1.0, 0.0, 0.0],
        "Tool B desc": [0.0, 1.0, 0.0]
    })
    mock_plugin_manager_for_es_lookup.get_plugin_instance.return_value = mock_embedder
    await es_lookup_provider.setup(config={"plugin_manager": mock_plugin_manager_for_es_lookup})

    tools_data = [
        {"identifier": "tool_a", "lookup_text_representation": "Tool A desc"},
        {"identifier": "tool_b", "lookup_text_representation": "Tool B desc"}
    ]
    await es_lookup_provider.index_tools(tools_data)

    assert es_lookup_provider._indexed_tool_embeddings is not None
    assert es_lookup_provider._indexed_tool_embeddings.shape == (2, 3)
    assert len(es_lookup_provider._indexed_tool_data_list) == 2
    assert es_lookup_provider._indexed_tool_data_list[0]["identifier"] == "tool_a"

@pytest.mark.asyncio
async def test_es_index_tools_no_embedder(
    es_lookup_provider: EmbeddingSimilarityLookupProvider, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR)
    # Setup will fail to load embedder if PM returns None
    mock_plugin_manager_for_es_lookup.get_plugin_instance.return_value = None
    await es_lookup_provider.setup(config={"plugin_manager": mock_plugin_manager_for_es_lookup})

    await es_lookup_provider.index_tools([{"id": "t1", "lookup_text_representation": "desc"}])
    assert es_lookup_provider._indexed_tool_embeddings is None
    assert len(es_lookup_provider._indexed_tool_data_list) == 0
    assert f"{es_lookup_provider.plugin_id}: Embedder not available. Cannot index tools." in caplog.text

@pytest.mark.asyncio
async def test_es_index_tools_embed_failure(
    es_lookup_provider: EmbeddingSimilarityLookupProvider,
    mock_plugin_manager_for_es_lookup: PluginManager, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR)
    mock_embedder = MockEmbedderForLookup()
    mock_embedder.set_fail_on_embed(True)
    mock_plugin_manager_for_es_lookup.get_plugin_instance.return_value = mock_embedder
    await es_lookup_provider.setup(config={"plugin_manager": mock_plugin_manager_for_es_lookup})

    await es_lookup_provider.index_tools([{"id": "t1", "lookup_text_representation": "desc"}])
    assert es_lookup_provider._indexed_tool_embeddings is not None # Should be empty array
    if np:
        assert es_lookup_provider._indexed_tool_embeddings.size == 0
    assert len(es_lookup_provider._indexed_tool_data_list) == 0
    assert "Error during tool text embedding for index: Simulated embedder failure" in caplog.text

@pytest.mark.skipif(np is None, reason="NumPy not available")
@pytest.mark.asyncio
async def test_es_find_tools_successful(
    es_lookup_provider: EmbeddingSimilarityLookupProvider,
    mock_plugin_manager_for_es_lookup: PluginManager
):
    mock_embedder = MockEmbedderForLookup()
    mock_embedder.set_embeddings_map({
        "Query text for tool A": [0.9, 0.1, 0.0], # Query embedding
        "Tool A description for index": [1.0, 0.0, 0.0],
        "Tool B description for index": [0.0, 1.0, 0.0]
    })
    mock_plugin_manager_for_es_lookup.get_plugin_instance.return_value = mock_embedder
    await es_lookup_provider.setup(config={"plugin_manager": mock_plugin_manager_for_es_lookup})

    tools_data = [
        {"identifier": "tool_a", "lookup_text_representation": "Tool A description for index"},
        {"identifier": "tool_b", "lookup_text_representation": "Tool B description for index"}
    ]
    await es_lookup_provider.index_tools(tools_data)

    results = await es_lookup_provider.find_tools("Query text for tool A", top_k=1)
    assert len(results) == 1
    assert results[0].tool_identifier == "tool_a"
    assert results[0].score > 0.9 # High similarity due to embedding map design

@pytest.mark.asyncio
async def test_es_find_tools_index_not_built(
    es_lookup_provider: EmbeddingSimilarityLookupProvider,
    mock_plugin_manager_for_es_lookup: PluginManager, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.WARNING)
    # Setup with embedder, but don't call index_tools
    mock_embedder = MockEmbedderForLookup()
    mock_plugin_manager_for_es_lookup.get_plugin_instance.return_value = mock_embedder
    await es_lookup_provider.setup(config={"plugin_manager": mock_plugin_manager_for_es_lookup})

    results = await es_lookup_provider.find_tools("any query")
    assert results == []
    assert f"{es_lookup_provider.plugin_id}: Index not built" in caplog.text

@pytest.mark.asyncio
async def test_es_find_tools_query_embed_failure(
    es_lookup_provider: EmbeddingSimilarityLookupProvider,
    mock_plugin_manager_for_es_lookup: PluginManager, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR)
    mock_embedder = MockEmbedderForLookup()
    mock_embedder.set_fail_on_embed(True) # Make query embedding fail
    mock_plugin_manager_for_es_lookup.get_plugin_instance.return_value = mock_embedder
    await es_lookup_provider.setup(config={"plugin_manager": mock_plugin_manager_for_es_lookup})

    # Index successfully first
    mock_embedder.set_fail_on_embed(False)
    mock_embedder.set_fixed_embedding([0.1, 0.2, 0.3])
    await es_lookup_provider.index_tools([{"id":"t1", "lookup_text_representation":"text"}])
    assert es_lookup_provider._indexed_tool_embeddings is not None

    mock_embedder.set_fail_on_embed(True) # Now make it fail for the query
    results = await es_lookup_provider.find_tools("any query")
    assert results == []
    assert "Error embedding lookup query: Simulated embedder failure" in caplog.text

@pytest.mark.asyncio
async def test_es_find_tools_empty_query(
    es_lookup_provider: EmbeddingSimilarityLookupProvider,
    mock_plugin_manager_for_es_lookup: PluginManager
):
    mock_embedder = MockEmbedderForLookup() # Ensure embedder is set up
    mock_plugin_manager_for_es_lookup.get_plugin_instance.return_value = mock_embedder
    await es_lookup_provider.setup(config={"plugin_manager": mock_plugin_manager_for_es_lookup})

    # Index some data so index is not empty
    mock_embedder.set_fixed_embedding([0.1,0.2,0.3])
    await es_lookup_provider.index_tools([{"identifier":"t1", "lookup_text_representation":"some text"}])

    results = await es_lookup_provider.find_tools("") # Empty query
    assert results == []
    # No specific log for empty query in this provider, it relies on embedder to handle.
    # The embedder might yield nothing or an empty vector, leading to no results.

@pytest.mark.skipif(np is None, reason="NumPy not available")
@pytest.mark.asyncio
async def test_es_index_tools_mismatched_embedding_dimensions(
    es_lookup_provider: EmbeddingSimilarityLookupProvider,
    mock_plugin_manager_for_es_lookup: PluginManager,
    caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR)
    mock_embedder = MockEmbedderForLookup()
    # Simulate embedder returning embeddings of different dimensions
    async def inconsistent_embed(chunks: AsyncIterable[Chunk], config=None):
        count = 0
        async for chunk_item in chunks:
            if count == 0:
                yield chunk_item, [1.0, 2.0, 3.0]
            else:
                yield chunk_item, [4.0, 5.0] # Different dimension
            count += 1
    mock_embedder.embed = inconsistent_embed # type: ignore
    mock_plugin_manager_for_es_lookup.get_plugin_instance.return_value = mock_embedder
    await es_lookup_provider.setup(config={"plugin_manager": mock_plugin_manager_for_es_lookup})

    tools_data = [
        {"identifier": "tool_a", "lookup_text_representation": "Desc A"},
        {"identifier": "tool_b", "lookup_text_representation": "Desc B"}
    ]
    await es_lookup_provider.index_tools(tools_data)
    assert "Failed to convert embeddings to NumPy array" in caplog.text
    assert "Likely inconsistent embedding dimensions" in caplog.text
    if np:
        assert es_lookup_provider._indexed_tool_embeddings.size == 0
    assert len(es_lookup_provider._indexed_tool_data_list) == 0


@pytest.mark.asyncio
async def test_es_teardown(
    es_lookup_provider: EmbeddingSimilarityLookupProvider,
    mock_plugin_manager_for_es_lookup: PluginManager
):
    mock_embedder = MockEmbedderForLookup()
    mock_plugin_manager_for_es_lookup.get_plugin_instance.return_value = mock_embedder
    await es_lookup_provider.setup(config={"plugin_manager": mock_plugin_manager_for_es_lookup})

    # Index something to populate internal state
    mock_embedder.set_fixed_embedding([0.1,0.2,0.3])
    await es_lookup_provider.index_tools([{"identifier":"t1", "lookup_text_representation":"text"}])
    assert es_lookup_provider._embedder is not None
    assert es_lookup_provider._indexed_tool_embeddings is not None

    await es_lookup_provider.teardown()
    assert es_lookup_provider._embedder is None
    assert es_lookup_provider._indexed_tool_embeddings is None
    assert len(es_lookup_provider._indexed_tool_data_list) == 0
    assert mock_embedder.teardown_called is True
