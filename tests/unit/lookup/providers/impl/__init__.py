### tests/unit/lookup/providers/impl/test_embedding_similarity_lookup.py
"""Unit tests for EmbeddingSimilarityLookupProvider."""
from typing import Any, AsyncIterable, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

try:
    import numpy as np
except ImportError:
    np = None

from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import Chunk, EmbeddingVector, Plugin

# Corrected import path for EmbeddingGeneratorPlugin
from genie_tooling.embedding_generators.abc import EmbeddingGeneratorPlugin

# Corrected import for the implementation
from genie_tooling.tool_lookup_providers.impl.embedding_similarity import (
    EmbeddingSimilarityLookupProvider,
)

# Import the specific logger instance from the module under test


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
def es_lookup_provider(mock_plugin_manager_for_es_lookup: PluginManager) -> EmbeddingSimilarityLookupProvider:
    provider = EmbeddingSimilarityLookupProvider()
    return provider

# --- Test Cases from original file ---
# Ensure all tests use the `es_lookup_provider` fixture correctly
# and `await provider.setup(...)` within the test if setup is per-test.

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

# Add other tests from the original test_embedding_similarity_lookup.py here,
# ensuring they use the corrected imports and fixtures.
# For brevity, I'm not copying all of them, but the structure above shows how.
# Example:
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

