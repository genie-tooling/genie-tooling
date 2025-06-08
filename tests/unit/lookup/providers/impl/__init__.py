# tests/unit/lookup/providers/impl/test_embedding_similarity_lookup.py
# (Content from the original tests/unit/lookup/providers/impl/__init__.py will be moved here and augmented)
from typing import Any, AsyncIterable, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch
import logging
import pytest

try:
    import numpy as np
except ImportError:
    np = None # type: ignore

from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import Chunk, EmbeddingVector, Plugin, RetrievedChunk
from genie_tooling.embedding_generators.abc import EmbeddingGeneratorPlugin
from genie_tooling.lookup.types import RankedToolResult
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tool_lookup_providers.impl.embedding_similarity import (
    EmbeddingSimilarityLookupProvider,
)
from genie_tooling.vector_stores.abc import VectorStorePlugin

PROVIDER_LOGGER_NAME = "genie_tooling.tool_lookup_providers.impl.embedding_similarity"


# --- Mocks ---
class MockEmbedderForLookup(EmbeddingGeneratorPlugin, Plugin):
    _plugin_id_value: str = "mock_embedder_for_lookup_v1"
    description: str = "Mock embedder for testing lookup"
    setup_config_received: Optional[Dict[str, Any]] = None
    _fixed_embedding: Optional[List[float]] = None
    _embeddings_map: Optional[Dict[str, List[float]]] = None
    _fail_on_embed: bool = False
    teardown_called: bool = False

    @property
    def plugin_id(self) -> str:
        return self._plugin_id_value

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.setup_config_received = config
        self.teardown_called = False

    def set_fixed_embedding(self, embedding: List[float]):
        self._fixed_embedding = embedding
        self._embeddings_map = None

    def set_embeddings_map(self, embeddings_map: Dict[str, List[float]]):
        self._embeddings_map = embeddings_map
        self._fixed_embedding = None

    def set_fail_on_embed(self, fail: bool):
        self._fail_on_embed = fail

    async def embed(
        self, chunks: AsyncIterable[Chunk], config: Optional[Dict[str, Any]] = None
    ) -> AsyncIterable[tuple[Chunk, EmbeddingVector]]:
        if self._fail_on_embed:
            raise RuntimeError("Simulated embedder failure")

        async for chunk in chunks:
            if self._embeddings_map and chunk.content in self._embeddings_map:
                yield chunk, self._embeddings_map[chunk.content]
            elif self._fixed_embedding:
                yield chunk, self._fixed_embedding
            else:
                dim = (config or {}).get("expected_dim", 3)
                yield chunk, [0.1] * dim

    async def teardown(self) -> None:
        self.teardown_called = True


class MockVectorStoreForLookup(VectorStorePlugin, Plugin):
    _plugin_id_value: str = "mock_vs_for_lookup_v1"
    description: str = "Mock Vector Store for lookup tests"
    setup_config_received: Optional[Dict[str, Any]] = None
    add_should_fail: bool = False
    search_should_fail: bool = False
    search_results: List[RetrievedChunk] = []
    teardown_called: bool = False

    @property
    def plugin_id(self) -> str:
        return self._plugin_id_value

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.setup_config_received = config
        self.teardown_called = False

    async def add(
        self,
        embeddings: AsyncIterable[Tuple[Chunk, EmbeddingVector]],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.add_should_fail:
            raise RuntimeError("Simulated VS add failure")
        count = 0
        async for _ in embeddings:
            count += 1
        return {"added_count": count, "errors": []}

    async def search(
        self,
        query_embedding: EmbeddingVector,
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        if self.search_should_fail:
            raise RuntimeError("Simulated VS search failure")
        return self.search_results[:top_k]

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        delete_all: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return True # Not focus of these tests

    async def teardown(self) -> None:
        self.teardown_called = True


@pytest.fixture
def mock_plugin_manager_for_es_lookup(mocker) -> PluginManager:
    pm = mocker.MagicMock(spec=PluginManager)
    pm.get_plugin_instance = AsyncMock()
    return pm


@pytest.fixture
def mock_key_provider_for_es_lookup(mocker) -> KeyProvider:
    kp = mocker.AsyncMock(spec=KeyProvider)
    return kp


@pytest.fixture
def es_lookup_provider(
    mock_plugin_manager_for_es_lookup: PluginManager,
) -> EmbeddingSimilarityLookupProvider:
    provider = EmbeddingSimilarityLookupProvider()
    # Setup will be called within tests with specific configurations
    return provider


# --- Test Cases ---
@pytest.mark.asyncio
async def test_es_setup_direct_vs_config_params(
    es_lookup_provider: EmbeddingSimilarityLookupProvider,
    mock_plugin_manager_for_es_lookup: PluginManager,
    mock_key_provider_for_es_lookup: KeyProvider,
):
    mock_embedder = MockEmbedderForLookup()
    mock_vs = MockVectorStoreForLookup()

    async def get_instance_side_effect_vs(plugin_id_req, config=None, **kwargs):
        if plugin_id_req == "custom_embed_id":
            await mock_embedder.setup(config)
            return mock_embedder
        if plugin_id_req == "mock_vs_for_lookup_v1":
            await mock_vs.setup(config)
            return mock_vs
        return None

    mock_plugin_manager_for_es_lookup.get_plugin_instance.side_effect = (
        get_instance_side_effect_vs
    )

    await es_lookup_provider.setup(
        config={
            "plugin_manager": mock_plugin_manager_for_es_lookup,
            "key_provider": mock_key_provider_for_es_lookup,
            "embedder_id": "custom_embed_id",
            "embedder_config": {"model": "embed_model"},
            "vector_store_id": "mock_vs_for_lookup_v1",
            "tool_embeddings_collection_name": "my_tools_collection",
            "tool_embeddings_path": "/custom/tools/db_path",
            "vector_store_config": {
                "specific_vs_param": "value",
                "path": "ignored_path_due_to_direct",
            },
        }
    )

    assert es_lookup_provider._embedder is mock_embedder
    assert mock_embedder.setup_config_received is not None
    assert mock_embedder.setup_config_received.get("model") == "embed_model"
    assert (
        mock_embedder.setup_config_received.get("key_provider")
        is mock_key_provider_for_es_lookup
    )
    assert es_lookup_provider._tool_vector_store is mock_vs
    assert mock_vs.setup_config_received is not None
    assert mock_vs.setup_config_received.get("specific_vs_param") == "value"
    assert mock_vs.setup_config_received.get("collection_name") == "my_tools_collection"
    # Path is passed to VS config, so it should be there
    assert mock_vs.setup_config_received.get("path") == "/custom/tools/db_path"


@pytest.mark.asyncio
async def test_es_setup_fail_plugin_manager_missing(
    es_lookup_provider: EmbeddingSimilarityLookupProvider, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
    await es_lookup_provider.setup(config={})  # No plugin_manager
    assert es_lookup_provider._embedder is None
    assert es_lookup_provider._tool_vector_store is None
    assert "PluginManager not provided. Cannot load sub-plugins." in caplog.text


@pytest.mark.asyncio
async def test_es_setup_fail_embedder_load(
    es_lookup_provider: EmbeddingSimilarityLookupProvider,
    mock_plugin_manager_for_es_lookup: PluginManager,
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
    mock_plugin_manager_for_es_lookup.get_plugin_instance.return_value = None
    await es_lookup_provider.setup(
        config={"plugin_manager": mock_plugin_manager_for_es_lookup}
    )
    assert es_lookup_provider._embedder is None
    assert (
        f"Embedder '{EmbeddingSimilarityLookupProvider.DEFAULT_EMBEDDER_ID}' not found/invalid."
        in caplog.text
    )


@pytest.mark.skipif(np is None, reason="NumPy not available for this test variant")
@pytest.mark.asyncio
async def test_es_index_tools_success_in_memory(
    es_lookup_provider: EmbeddingSimilarityLookupProvider,
    mock_plugin_manager_for_es_lookup: PluginManager,
    mock_key_provider_for_es_lookup: KeyProvider,
):
    mock_embedder = MockEmbedderForLookup()
    mock_embedder.set_embeddings_map(
        {
            "Tool A desc": [1.0, 0.0, 0.0],
            "Tool B desc": [0.0, 1.0, 0.0],
            "Tool C text": [0.0, 0.0, 1.0],
        }
    )
    mock_plugin_manager_for_es_lookup.get_plugin_instance.return_value = mock_embedder

    await es_lookup_provider.setup(
        config={
            "plugin_manager": mock_plugin_manager_for_es_lookup,
            "key_provider": mock_key_provider_for_es_lookup,
            "vector_store_id": None,  # Force in-memory
        }
    )

    tools_data = [
        {"identifier": "tool_a", "lookup_text_representation": "Tool A desc"},
        {"identifier": "tool_b", "lookup_text_representation": "Tool B desc"},
        {"identifier": "tool_c", "lookup_text_representation": "Tool C text"},
    ]
    await es_lookup_provider.index_tools(tools_data)

    assert es_lookup_provider._indexed_tool_embeddings_np is not None
    assert es_lookup_provider._indexed_tool_embeddings_np.shape == (3, 3)
    assert len(es_lookup_provider._indexed_tool_data_list_np) == 3
    assert es_lookup_provider._indexed_tool_data_list_np[0]["identifier"] == "tool_a"


@pytest.mark.asyncio
async def test_es_index_tools_success_with_vector_store(
    es_lookup_provider: EmbeddingSimilarityLookupProvider,
    mock_plugin_manager_for_es_lookup: PluginManager,
    mock_key_provider_for_es_lookup: KeyProvider,
):
    mock_embedder = MockEmbedderForLookup()
    mock_embedder.set_fixed_embedding([0.5, 0.5])
    mock_vs = MockVectorStoreForLookup()

    async def get_instance_side_effect(plugin_id_req, config=None, **kwargs):
        if plugin_id_req == EmbeddingSimilarityLookupProvider.DEFAULT_EMBEDDER_ID:
            return mock_embedder
        if plugin_id_req == "vs_for_index_test":
            return mock_vs
        return None

    mock_plugin_manager_for_es_lookup.get_plugin_instance.side_effect = (
        get_instance_side_effect
    )

    await es_lookup_provider.setup(
        config={
            "plugin_manager": mock_plugin_manager_for_es_lookup,
            "key_provider": mock_key_provider_for_es_lookup,
            "vector_store_id": "vs_for_index_test",
        }
    )

    tools_data = [
        {"identifier": "tool_vs1", "lookup_text_representation": "VS Text 1"},
    ]
    mock_vs.add = AsyncMock(return_value={"added_count": 1, "errors": []})
    await es_lookup_provider.index_tools(tools_data)
    mock_vs.add.assert_awaited_once()


@pytest.mark.asyncio
async def test_es_index_tools_embedder_fails(
    es_lookup_provider: EmbeddingSimilarityLookupProvider,
    mock_plugin_manager_for_es_lookup: PluginManager,
    mock_key_provider_for_es_lookup: KeyProvider,
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
    mock_embedder = MockEmbedderForLookup()
    mock_embedder.set_fail_on_embed(True)
    mock_plugin_manager_for_es_lookup.get_plugin_instance.return_value = mock_embedder
    await es_lookup_provider.setup(
        config={
            "plugin_manager": mock_plugin_manager_for_es_lookup,
            "key_provider": mock_key_provider_for_es_lookup,
            "vector_store_id": None,
        }
    )
    await es_lookup_provider.index_tools(
        [{"identifier": "t1", "lookup_text_representation": "text"}]
    )
    assert "Error during embedding tool texts: Simulated embedder failure" in caplog.text


@pytest.mark.skipif(np is None, reason="NumPy not available for this test variant")
@pytest.mark.asyncio
async def test_es_find_tools_success_in_memory(
    es_lookup_provider: EmbeddingSimilarityLookupProvider,
    mock_plugin_manager_for_es_lookup: PluginManager,
    mock_key_provider_for_es_lookup: KeyProvider,
):
    mock_embedder = MockEmbedderForLookup()
    # Query embedding will be [0.1, 0.1, 0.1]
    mock_embedder.set_fixed_embedding([0.1, 0.1, 0.1])

    mock_plugin_manager_for_es_lookup.get_plugin_instance.return_value = mock_embedder
    await es_lookup_provider.setup(
        config={
            "plugin_manager": mock_plugin_manager_for_es_lookup,
            "key_provider": mock_key_provider_for_es_lookup,
            "vector_store_id": None,
        }
    )
    # Manually set up the in-memory index for testing find_tools
    es_lookup_provider._indexed_tool_embeddings_np = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.1, 0.1, 0.09]], dtype=np.float32
    )
    es_lookup_provider._indexed_tool_data_list_np = [
        {"identifier": "tool_A", "lookup_text_representation": "A"},
        {"identifier": "tool_B", "lookup_text_representation": "B"},
        {"identifier": "tool_C", "lookup_text_representation": "C"}, # Most similar
    ]

    results = await es_lookup_provider.find_tools("query for C", top_k=1)
    assert len(results) == 1
    assert results[0].tool_identifier == "tool_C"
    assert results[0].score > 0.9 # Expect high similarity


@pytest.mark.asyncio
async def test_es_find_tools_success_with_vector_store(
    es_lookup_provider: EmbeddingSimilarityLookupProvider,
    mock_plugin_manager_for_es_lookup: PluginManager,
    mock_key_provider_for_es_lookup: KeyProvider,
):
    mock_embedder = MockEmbedderForLookup()
    mock_embedder.set_fixed_embedding([0.7, 0.7]) # Query embedding
    mock_vs = MockVectorStoreForLookup()
    mock_vs.search_results = [
        RetrievedChunk(id="vs_tool1", content="VS Tool 1 Content", score=0.95, metadata={"id": "vs_tool1"}) # type: ignore
    ]

    async def get_instance_side_effect(plugin_id_req, config=None, **kwargs):
        if plugin_id_req == EmbeddingSimilarityLookupProvider.DEFAULT_EMBEDDER_ID:
            return mock_embedder
        if plugin_id_req == "vs_for_find_test":
            return mock_vs
        return None

    mock_plugin_manager_for_es_lookup.get_plugin_instance.side_effect = (
        get_instance_side_effect
    )
    await es_lookup_provider.setup(
        config={
            "plugin_manager": mock_plugin_manager_for_es_lookup,
            "key_provider": mock_key_provider_for_es_lookup,
            "vector_store_id": "vs_for_find_test",
        }
    )

    results = await es_lookup_provider.find_tools("query for vs_tool1", top_k=1)
    assert len(results) == 1
    assert results[0].tool_identifier == "vs_tool1"
    assert results[0].score == 0.95
    mock_vs.search.assert_awaited_once()


@pytest.mark.asyncio
async def test_es_teardown_clears_resources(
    es_lookup_provider: EmbeddingSimilarityLookupProvider,
    mock_plugin_manager_for_es_lookup: PluginManager,
    mock_key_provider_for_es_lookup: KeyProvider,
):
    mock_embedder = MockEmbedderForLookup()
    mock_vs = MockVectorStoreForLookup()
    async def get_instance_side_effect(plugin_id_req, config=None, **kwargs):
        if plugin_id_req == EmbeddingSimilarityLookupProvider.DEFAULT_EMBEDDER_ID: return mock_embedder
        if plugin_id_req == "vs_for_teardown": return mock_vs
        return None
    mock_plugin_manager_for_es_lookup.get_plugin_instance.side_effect = get_instance_side_effect

    await es_lookup_provider.setup(config={
        "plugin_manager": mock_plugin_manager_for_es_lookup,
        "key_provider": mock_key_provider_for_es_lookup,
        "vector_store_id": "vs_for_teardown"
    })
    es_lookup_provider._indexed_tool_embeddings_np = "dummy_np_array" # type: ignore
    es_lookup_provider._indexed_tool_data_list_np = [{"data": "dummy"}]

    await es_lookup_provider.teardown()

    assert es_lookup_provider._embedder is None
    assert es_lookup_provider._tool_vector_store is None
    assert es_lookup_provider._plugin_manager is None
    assert es_lookup_provider._key_provider is None
    assert es_lookup_provider._indexed_tool_embeddings_np is None
    assert es_lookup_provider._indexed_tool_data_list_np == []