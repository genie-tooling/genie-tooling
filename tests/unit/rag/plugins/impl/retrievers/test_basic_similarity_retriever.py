### tests/unit/rag/plugins/impl/retrievers/test_basic_similarity_retriever.py
"""Unit tests for BasicSimilarityRetriever."""
import logging
from typing import Any, AsyncIterable, ClassVar, Dict, List, Optional, Tuple, cast
from unittest.mock import AsyncMock

import pytest
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import Chunk, EmbeddingVector, Plugin, RetrievedChunk
from genie_tooling.embedding_generators.abc import EmbeddingGeneratorPlugin
from genie_tooling.retrievers.impl.basic_similarity import (
    BasicSimilarityRetriever,
    _QueryChunkForEmbedding,
)
from genie_tooling.vector_stores.abc import VectorStorePlugin

RETRIEVER_LOGGER_NAME = "genie_tooling.retrievers.impl.basic_similarity"

class MockRetrieverEmbedder(EmbeddingGeneratorPlugin):
    plugin_id: str = "mock_retriever_embedder_v1"
    description: str = "Mock embedder for retriever tests"
    _embeddings_to_yield: ClassVar[List[Tuple[Chunk, EmbeddingVector]]] = []
    _embed_should_raise: Optional[Exception] = None

    async def embed(self, chunks: AsyncIterable[Chunk], config: Optional[Dict[str, Any]] = None) -> AsyncIterable[Tuple[Chunk, EmbeddingVector]]:
        if self._embed_should_raise:
            async for _ in chunks:
                pass # Consume input if erroring
            raise self._embed_should_raise

        chunk_list = [c async for c in chunks]
        if not chunk_list:
            if False:
                yield # Make it an async generator
            return

        if len(chunk_list) == 1 and isinstance(chunk_list[0], _QueryChunkForEmbedding):
            yield chunk_list[0], [0.1, 0.2, 0.3] # Default query embedding
        else:
            for item in self._embeddings_to_yield:
                yield item

    def set_embeddings_to_yield(self, embeddings: List[Tuple[Chunk, EmbeddingVector]]):
        self._embeddings_to_yield = embeddings

    def set_embed_should_raise(self, error: Exception):
        self._embed_should_raise = error

    async def teardown(self) -> None:
        pass
    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass


class MockRetrieverVectorStore(VectorStorePlugin):
    plugin_id: str = "mock_retriever_vector_store_v1"
    description: str = "Mock vector store for retriever tests"
    _search_results: ClassVar[List[RetrievedChunk]] = []
    _search_should_raise: Optional[Exception] = None

    async def search(self, query_embedding: EmbeddingVector, top_k: int, filter_metadata: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None) -> List[RetrievedChunk]:
        if self._search_should_raise:
            raise self._search_should_raise
        return self._search_results[:top_k]

    def set_search_results(self, results: List[RetrievedChunk]):
        self._search_results = results

    def set_search_should_raise(self, error: Exception):
        self._search_should_raise = error

    async def add(self, embeddings: AsyncIterable[Tuple[Chunk, EmbeddingVector]], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: return {"added_count": 0}
    async def delete(self, ids: Optional[List[str]] = None, filter_metadata: Optional[Dict[str, Any]] = None, delete_all: bool = False, config: Optional[Dict[str, Any]] = None) -> bool: return True
    async def teardown(self) -> None:
        pass
    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass


class MockRetrievedChunkImpl(RetrievedChunk, Chunk): # type: ignore
    def __init__(self, content: str, metadata: Dict[str, Any], score: float, id: Optional[str] = None, rank: Optional[int] = None):
        self.content: str = content
        self.metadata: Dict[str, Any] = metadata
        self.id: Optional[str] = id
        self.score: float = score
        self.rank: Optional[int] = rank

@pytest.fixture()
def mock_plugin_manager_for_retriever() -> PluginManager:
    pm = AsyncMock(spec=PluginManager)
    return pm

@pytest.fixture()
async def basic_retriever(mock_plugin_manager_for_retriever: PluginManager) -> BasicSimilarityRetriever:
    retriever_instance = BasicSimilarityRetriever()
    mock_embedder = MockRetrieverEmbedder()
    mock_store = MockRetrieverVectorStore()

    async def default_get_instance(plugin_id_req: str, config=None):
        if plugin_id_req == retriever_instance._default_embedder_id:
            await mock_embedder.setup(config)
            return mock_embedder
        if plugin_id_req == retriever_instance._default_vector_store_id:
            await mock_store.setup(config)
            return mock_store
        # Fallback for other plugins if needed by tests, though not directly by retriever
        generic_mock = AsyncMock(spec=Plugin)
        generic_mock.plugin_id = plugin_id_req
        if hasattr(generic_mock, "setup"):
            await generic_mock.setup(config)
        return generic_mock

    mock_plugin_manager_for_retriever.get_plugin_instance.side_effect = default_get_instance

    await retriever_instance.setup(config={"plugin_manager": mock_plugin_manager_for_retriever})
    return retriever_instance

@pytest.mark.asyncio()
async def test_setup_success(mock_plugin_manager_for_retriever: PluginManager):
    retriever_instance = BasicSimilarityRetriever()
    mock_embedder = MockRetrieverEmbedder()
    mock_store = MockRetrieverVectorStore()

    async def get_instance(plugin_id_req: str, config=None):
        if plugin_id_req == "custom_embed_id":
            await mock_embedder.setup(config)
            return mock_embedder
        if plugin_id_req == "custom_store_id":
            await mock_store.setup(config)
            return mock_store
        return None
    mock_plugin_manager_for_retriever.get_plugin_instance.side_effect = get_instance

    await retriever_instance.setup(config={
        "plugin_manager": mock_plugin_manager_for_retriever,
        "embedder_id": "custom_embed_id",
        "vector_store_id": "custom_store_id"
    })
    assert retriever_instance._embedder is mock_embedder
    assert retriever_instance._vector_store is mock_store
    assert retriever_instance._embedder_id_used == "custom_embed_id"
    assert retriever_instance._vector_store_id_used == "custom_store_id"
    await retriever_instance.teardown()

@pytest.mark.asyncio()
async def test_setup_no_plugin_manager(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=RETRIEVER_LOGGER_NAME)
    retriever_instance = BasicSimilarityRetriever()
    await retriever_instance.setup(config={}) # No plugin_manager
    assert any(
        "PluginManager not provided or invalid" in rec.message and rec.name == RETRIEVER_LOGGER_NAME
        for rec in caplog.records
    )
    assert retriever_instance._embedder is None
    assert retriever_instance._vector_store is None
    await retriever_instance.teardown()

@pytest.mark.asyncio()
async def test_setup_embedder_load_fail(mock_plugin_manager_for_retriever: PluginManager, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=RETRIEVER_LOGGER_NAME)
    retriever_instance = BasicSimilarityRetriever()
    mock_store = MockRetrieverVectorStore()

    async def get_instance_fail_embed(plugin_id_req: str, config=None):
        if plugin_id_req == retriever_instance._default_embedder_id:
            return None # Simulate load fail
        if plugin_id_req == retriever_instance._default_vector_store_id:
            await mock_store.setup(config)
            return mock_store
        return None
    mock_plugin_manager_for_retriever.get_plugin_instance.side_effect = get_instance_fail_embed

    await retriever_instance.setup(config={"plugin_manager": mock_plugin_manager_for_retriever})
    assert any(
        f"EmbeddingGeneratorPlugin '{retriever_instance._default_embedder_id}' not found or invalid" in rec.message and rec.name == RETRIEVER_LOGGER_NAME
        for rec in caplog.records
    )
    assert retriever_instance._embedder is None
    assert retriever_instance._vector_store is mock_store
    await retriever_instance.teardown()

@pytest.mark.asyncio()
async def test_setup_vector_store_load_fail(mock_plugin_manager_for_retriever: PluginManager, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=RETRIEVER_LOGGER_NAME)
    retriever_instance = BasicSimilarityRetriever()
    mock_embedder = MockRetrieverEmbedder()

    async def get_instance_fail_store(plugin_id_req: str, config=None):
        if plugin_id_req == retriever_instance._default_embedder_id:
            await mock_embedder.setup(config)
            return mock_embedder
        if plugin_id_req == retriever_instance._default_vector_store_id:
            return None # Simulate load fail
        return None
    mock_plugin_manager_for_retriever.get_plugin_instance.side_effect = get_instance_fail_store

    await retriever_instance.setup(config={"plugin_manager": mock_plugin_manager_for_retriever})
    assert any(
        f"VectorStorePlugin '{retriever_instance._default_vector_store_id}' not found or invalid" in rec.message and rec.name == RETRIEVER_LOGGER_NAME
        for rec in caplog.records
    )
    assert retriever_instance._vector_store is None
    assert retriever_instance._embedder is mock_embedder
    await retriever_instance.teardown()

@pytest.mark.asyncio()
async def test_retrieve_success(basic_retriever: BasicSimilarityRetriever):
    actual_retriever = await basic_retriever
    retrieved_doc = MockRetrievedChunkImpl("content", {}, 0.9, "id1")
    cast(MockRetrieverVectorStore, actual_retriever._vector_store).set_search_results([retrieved_doc])

    results = await actual_retriever.retrieve("test query", top_k=1)
    assert len(results) == 1
    assert results[0].id == "id1"
    await actual_retriever.teardown()

@pytest.mark.asyncio()
async def test_retrieve_not_setup(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=RETRIEVER_LOGGER_NAME)
    retriever_instance = BasicSimilarityRetriever() # Not calling setup
    results = await retriever_instance.retrieve("query", 1)
    assert results == []
    assert any(
        "Embedder or VectorStore not initialized" in rec.message and rec.name == RETRIEVER_LOGGER_NAME
        for rec in caplog.records
    )
    await retriever_instance.teardown()

@pytest.mark.asyncio()
async def test_retrieve_empty_query(basic_retriever: BasicSimilarityRetriever, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING, logger=RETRIEVER_LOGGER_NAME)
    actual_retriever = await basic_retriever
    results = await actual_retriever.retrieve("", top_k=1)
    assert results == []
    assert any(
        "Empty query provided" in rec.message and rec.name == RETRIEVER_LOGGER_NAME
        for rec in caplog.records
    )

    results_whitespace = await actual_retriever.retrieve("   ", top_k=1)
    assert results_whitespace == []
    assert any(
        "Empty query provided" in rec.message and rec.name == RETRIEVER_LOGGER_NAME
        for rec in caplog.records
    ) # Check again for the second call
    await actual_retriever.teardown()

@pytest.mark.asyncio()
async def test_retrieve_embed_query_fail(basic_retriever: BasicSimilarityRetriever, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=RETRIEVER_LOGGER_NAME)
    actual_retriever = await basic_retriever
    cast(MockRetrieverEmbedder, actual_retriever._embedder).set_embed_should_raise(RuntimeError("Embedding failed"))
    results = await actual_retriever.retrieve("query", 1)
    assert results == []
    assert any(
        "Error embedding query" in rec.message and "Embedding failed" in rec.message and rec.name == RETRIEVER_LOGGER_NAME
        for rec in caplog.records
    )
    await actual_retriever.teardown()

@pytest.mark.asyncio()
async def test_retrieve_embed_query_returns_no_vector(basic_retriever: BasicSimilarityRetriever, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=RETRIEVER_LOGGER_NAME)
    actual_retriever = await basic_retriever
    class EmptyVecEmbedder(EmbeddingGeneratorPlugin, Plugin): # type: ignore
        plugin_id="empty_vec_embed"
        description=""
        async def embed(self, chunks: AsyncIterable[Chunk], config=None) -> AsyncIterable[Tuple[Chunk, EmbeddingVector]]:
            async for chunk_item in chunks:
                yield chunk_item, [] # Return empty vector
        async def teardown(self) -> None: pass
        async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass


    actual_retriever._embedder = EmptyVecEmbedder()

    results = await actual_retriever.retrieve("query", 1)
    assert results == []
    assert any(
        "Failed to generate embedding for query. Embedder returned no vector." in rec.message and rec.name == RETRIEVER_LOGGER_NAME
        for rec in caplog.records
    )
    await actual_retriever.teardown()

@pytest.mark.asyncio()
async def test_retrieve_vector_store_search_fail(basic_retriever: BasicSimilarityRetriever, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=RETRIEVER_LOGGER_NAME)
    actual_retriever = await basic_retriever
    cast(MockRetrieverVectorStore, actual_retriever._vector_store).set_search_should_raise(RuntimeError("Search failed"))
    results = await actual_retriever.retrieve("query", 1)
    assert results == []
    assert any(
        "Error searching vector store" in rec.message and "Search failed" in rec.message and rec.name == RETRIEVER_LOGGER_NAME
        for rec in caplog.records
    )
    await actual_retriever.teardown()

@pytest.mark.asyncio()
async def test_retrieve_no_results_from_store(basic_retriever: BasicSimilarityRetriever):
    actual_retriever = await basic_retriever
    cast(MockRetrieverVectorStore, actual_retriever._vector_store).set_search_results([])
    results = await actual_retriever.retrieve("query", 5)
    assert results == []
    await actual_retriever.teardown()

@pytest.mark.asyncio()
async def test_teardown_nullifies_refs(basic_retriever: BasicSimilarityRetriever, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG, logger=RETRIEVER_LOGGER_NAME)
    actual_retriever = await basic_retriever

    embedder_teardown_mock = AsyncMock()
    vector_store_teardown_mock = AsyncMock()

    if actual_retriever._embedder:
        actual_retriever._embedder.teardown = embedder_teardown_mock # type: ignore
    if actual_retriever._vector_store:
        actual_retriever._vector_store.teardown = vector_store_teardown_mock # type: ignore

    assert actual_retriever._embedder is not None
    assert actual_retriever._vector_store is not None
    assert actual_retriever._plugin_manager is not None

    await actual_retriever.teardown()

    embedder_teardown_mock.assert_not_called()
    vector_store_teardown_mock.assert_not_called()

    assert actual_retriever._embedder is None
    assert actual_retriever._vector_store is None
    assert actual_retriever._plugin_manager is None
    assert any(
        "Teardown complete (references released)" in rec.message and rec.name == RETRIEVER_LOGGER_NAME
        for rec in caplog.records
    )
