"""Unit tests for the RAGManager."""
import logging  # Added logging import
from typing import Any, AsyncIterable, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import (  # Added Plugin
    Chunk,
    Document,
    EmbeddingVector,
    Plugin,
    RetrievedChunk,
)
from genie_tooling.rag.manager import RAGManager
# Updated import paths for RAG plugin ABCs
from genie_tooling.document_loaders.abc import DocumentLoaderPlugin
from genie_tooling.embedding_generators.abc import EmbeddingGeneratorPlugin
from genie_tooling.retrievers.abc import RetrieverPlugin
from genie_tooling.text_splitters.abc import TextSplitterPlugin
from genie_tooling.vector_stores.abc import VectorStorePlugin

# --- Mock RAG Component Implementations ---

class MockDocument(Document):
    def __init__(self, content: str, metadata: Dict[str, Any], id: Optional[str] = None):
        self.content = content
        self.metadata = metadata
        self.id = id

class MockChunk(Chunk):
    def __init__(self, content: str, metadata: Dict[str, Any], id: Optional[str] = None):
        self.content = content
        self.metadata = metadata
        self.id = id

class MockRetrievedChunk(RetrievedChunk, MockChunk):
    def __init__(self, content: str, metadata: Dict[str, Any], score: float, id: Optional[str] = None, rank: Optional[int] = None):
        super().__init__(content, metadata, id)
        self.score = score
        self.rank = rank


class MockRAGLoader(DocumentLoaderPlugin):
    plugin_id: str = "mock_rag_loader_v1"
    description: str = "Mock RAG Loader"

    def __init__(self):
        self._documents_to_yield: List[Document] = []
        self.load_should_raise: Optional[Exception] = None

    async def load(self, source_uri: str, config: Optional[Dict[str, Any]] = None) -> AsyncIterable[Document]:
        if self.load_should_raise:
            raise self.load_should_raise
        for doc in self._documents_to_yield:
            yield doc

    def set_documents(self, docs: List[Document]):
        self._documents_to_yield = docs

    def set_load_error(self, error: Exception):
        self.load_should_raise = error


class MockRAGSplitter(TextSplitterPlugin):
    plugin_id: str = "mock_rag_splitter_v1"
    description: str = "Mock RAG Splitter"

    def __init__(self):
        self._chunks_to_yield: List[Chunk] = []
        self.split_should_raise: Optional[Exception] = None

    async def split(self, documents: AsyncIterable[Document], config: Optional[Dict[str, Any]] = None) -> AsyncIterable[Chunk]:
        async for _doc in documents:
            if self.split_should_raise:
                raise self.split_should_raise

        for chunk_item in self._chunks_to_yield:
            yield chunk_item

    def set_chunks(self, chunks: List[Chunk]):
        self._chunks_to_yield = chunks

    def set_split_error(self, error: Exception):
        self.split_should_raise = error


class MockRAGEmbedder(EmbeddingGeneratorPlugin):
    plugin_id: str = "mock_rag_embedder_v1"
    description: str = "Mock RAG Embedder"

    def __init__(self):
        self._embeddings_to_yield: List[tuple[Chunk, EmbeddingVector]] = []
        self.embed_should_raise: Optional[Exception] = None

    async def embed(self, chunks: AsyncIterable[Chunk], config: Optional[Dict[str, Any]] = None) -> AsyncIterable[tuple[Chunk, EmbeddingVector]]:
        async for _chunk in chunks:
            if self.embed_should_raise:
                raise self.embed_should_raise
        for item in self._embeddings_to_yield:
            yield item

    def set_embeddings(self, embeddings: List[tuple[Chunk, EmbeddingVector]]):
        self._embeddings_to_yield = embeddings

    def set_embed_error(self, error: Exception):
        self.embed_should_raise = error

class MockRAGVectorStore(VectorStorePlugin):
    plugin_id: str = "mock_rag_vector_store_v1"
    description: str = "Mock RAG Vector Store"

    def __init__(self):
        self._add_result: Dict[str, Any] = {"added_count": 0, "errors": []}
        self._search_results: List[RetrievedChunk] = []
        self.add_should_raise: Optional[Exception] = None

    async def add(self, embeddings: AsyncIterable[tuple[Chunk, EmbeddingVector]], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.add_should_raise:
            async for _emb_item in embeddings: pass
            raise self.add_should_raise

        async for _emb_item in embeddings:
            pass
        return self._add_result

    async def search(self, query_embedding: EmbeddingVector, top_k: int, filter_metadata: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None) -> List[RetrievedChunk]:
        return self._search_results[:top_k]

    async def delete(self, ids: Optional[List[str]] = None, filter_metadata: Optional[Dict[str, Any]] = None, delete_all: bool = False, config: Optional[Dict[str, Any]] = None) -> bool:
        return True

    def set_add_result(self, result: Dict[str, Any]):
        self._add_result = result

    def set_search_results(self, results: List[RetrievedChunk]):
        self._search_results = results

    def set_add_error(self, error: Exception):
        self.add_should_raise = error


class MockRAGRetriever(RetrieverPlugin):
    plugin_id: str = "mock_rag_retriever_v1"
    description: str = "Mock RAG Retriever"

    def __init__(self):
        self._retrieve_results: List[RetrievedChunk] = []

    async def retrieve(self, query: str, top_k: int, config: Optional[Dict[str, Any]] = None) -> List[RetrievedChunk]:
        return self._retrieve_results[:top_k]

    def set_retrieve_results(self, results: List[RetrievedChunk]):
        self._retrieve_results = results


@pytest.fixture
def mock_plugin_manager_for_rag(mocker) -> PluginManager:
    pm = mocker.MagicMock(spec=PluginManager)
    pm.get_plugin_instance = AsyncMock()
    return pm

@pytest.fixture
def rag_manager_fixture(mock_plugin_manager_for_rag: PluginManager) -> RAGManager:
    return RAGManager(plugin_manager=mock_plugin_manager_for_rag)

# --- Test Cases ---

@pytest.mark.asyncio
async def test_rag_manager_index_data_source_success(
    rag_manager_fixture: RAGManager,
    mock_plugin_manager_for_rag: PluginManager
):
    loader_id, splitter_id, embedder_id, store_id = "loader1", "splitter1", "embedder1", "store1"
    source_uri = "/path/to/data"

    mock_loader = MockRAGLoader()
    mock_splitter = MockRAGSplitter()
    mock_embedder = MockRAGEmbedder()
    mock_store = MockRAGVectorStore()

    mock_docs = [MockDocument("doc1 content", {"s": "d1"}, "doc1")]
    mock_chunks = [MockChunk("chunk1 content", {"s": "c1"}, "chunk1")]
    mock_embeddings = [(mock_chunks[0], [0.1, 0.2])]

    mock_loader.set_documents(mock_docs)
    mock_splitter.set_chunks(mock_chunks)
    mock_embedder.set_embeddings(mock_embeddings)
    mock_store.set_add_result({"added_count": 1, "errors": []})

    async def get_plugin_side_effect(plugin_id_requested: str, config: Optional[Dict[str, Any]] = None):
        if plugin_id_requested == loader_id: return mock_loader
        if plugin_id_requested == splitter_id: return mock_splitter
        if plugin_id_requested == embedder_id: return mock_embedder
        if plugin_id_requested == store_id: return mock_store
        return None
    mock_plugin_manager_for_rag.get_plugin_instance.side_effect = get_plugin_side_effect

    result = await rag_manager_fixture.index_data_source(
        loader_id=loader_id, loader_source_uri=source_uri,
        splitter_id=splitter_id, embedder_id=embedder_id, vector_store_id=store_id
    )

    assert result["status"] == "success"
    assert result["added_count"] == 1
    assert not result["store_errors"]

    mock_plugin_manager_for_rag.get_plugin_instance.assert_any_call(loader_id, config=None)
    mock_plugin_manager_for_rag.get_plugin_instance.assert_any_call(splitter_id, config=None)
    mock_plugin_manager_for_rag.get_plugin_instance.assert_any_call(embedder_id, config=None)
    mock_plugin_manager_for_rag.get_plugin_instance.assert_any_call(store_id, config=None)


@pytest.mark.asyncio
async def test_rag_manager_index_data_source_component_load_failure(
    rag_manager_fixture: RAGManager,
    mock_plugin_manager_for_rag: PluginManager
):
    loader_id, splitter_id, embedder_id, store_id = "loader1", "splitter1", "embedder1", "store1"

    mock_plugin_manager_for_rag.get_plugin_instance.side_effect = lambda plugin_id, config=None: (
        MockRAGSplitter() if plugin_id == splitter_id else
        MockRAGEmbedder() if plugin_id == embedder_id else
        MockRAGVectorStore() if plugin_id == store_id else
        None
    )

    result = await rag_manager_fixture.index_data_source(
        loader_id=loader_id, loader_source_uri="/path",
        splitter_id=splitter_id, embedder_id=embedder_id, vector_store_id=store_id
    )
    assert result["status"] == "error"
    assert "Loader" in result["message"]


@pytest.mark.asyncio
async def test_rag_manager_index_data_source_loader_pipeline_error(
    rag_manager_fixture: RAGManager,
    mock_plugin_manager_for_rag: PluginManager
):
    loader_id, splitter_id, embedder_id, store_id = "loader1", "splitter1", "embedder1", "store1"

    error_raising_loader = MockRAGLoader()
    error_raising_loader.set_load_error(RuntimeError("Failed to load documents"))

    mock_plugin_manager_for_rag.get_plugin_instance.side_effect = lambda plugin_id, config=None: (
        error_raising_loader if plugin_id == loader_id else
        MockRAGSplitter() if plugin_id == splitter_id else
        MockRAGEmbedder() if plugin_id == embedder_id else
        MockRAGVectorStore() if plugin_id == store_id else
        None
    )

    result = await rag_manager_fixture.index_data_source(
        loader_id=loader_id, loader_source_uri="/path",
        splitter_id=splitter_id, embedder_id=embedder_id, vector_store_id=store_id
    )
    assert result["status"] == "error", f"Expected 'error', got '{result['status']}'. Message: {result.get('message')}"
    assert "Failed to load documents" in result["message"]


@pytest.mark.asyncio
async def test_rag_manager_index_data_source_splitter_pipeline_error(
    rag_manager_fixture: RAGManager,
    mock_plugin_manager_for_rag: PluginManager
):
    loader_id, splitter_id, embedder_id, store_id = "loader1", "splitter1", "embedder1", "store1"
    mock_loader = MockRAGLoader()
    mock_loader.set_documents([MockDocument("content", {}, "id1")])

    error_raising_splitter = MockRAGSplitter()
    error_raising_splitter.set_split_error(RuntimeError("Failed to split documents"))

    mock_plugin_manager_for_rag.get_plugin_instance.side_effect = lambda plugin_id, config=None: (
        mock_loader if plugin_id == loader_id else
        error_raising_splitter if plugin_id == splitter_id else
        MockRAGEmbedder() if plugin_id == embedder_id else
        MockRAGVectorStore() if plugin_id == store_id else
        None
    )
    result = await rag_manager_fixture.index_data_source(
        loader_id=loader_id, loader_source_uri="/path",
        splitter_id=splitter_id, embedder_id=embedder_id, vector_store_id=store_id
    )
    assert result["status"] == "error"
    assert "Failed to split documents" in result["message"]

@pytest.mark.asyncio
async def test_rag_manager_index_data_source_embedder_pipeline_error(
    rag_manager_fixture: RAGManager,
    mock_plugin_manager_for_rag: PluginManager
):
    loader_id, splitter_id, embedder_id, store_id = "loader1", "splitter1", "embedder1", "store1"
    mock_loader = MockRAGLoader()
    mock_loader.set_documents([MockDocument("content", {}, "id1")])
    mock_splitter = MockRAGSplitter()
    mock_splitter.set_chunks([MockChunk("chunk_content", {}, "cid1")])

    error_raising_embedder = MockRAGEmbedder()
    error_raising_embedder.set_embed_error(RuntimeError("Failed to embed chunks"))

    mock_plugin_manager_for_rag.get_plugin_instance.side_effect = lambda plugin_id, config=None: (
        mock_loader if plugin_id == loader_id else
        mock_splitter if plugin_id == splitter_id else
        error_raising_embedder if plugin_id == embedder_id else
        MockRAGVectorStore() if plugin_id == store_id else
        None
    )
    result = await rag_manager_fixture.index_data_source(
        loader_id=loader_id, loader_source_uri="/path",
        splitter_id=splitter_id, embedder_id=embedder_id, vector_store_id=store_id
    )
    assert result["status"] == "error"
    assert "Failed to embed chunks" in result["message"]

@pytest.mark.asyncio
async def test_rag_manager_index_data_source_vector_store_pipeline_error(
    rag_manager_fixture: RAGManager,
    mock_plugin_manager_for_rag: PluginManager
):
    loader_id, splitter_id, embedder_id, store_id = "loader1", "splitter1", "embedder1", "store1"
    mock_loader = MockRAGLoader()
    mock_loader.set_documents([MockDocument("content", {}, "id1")])
    mock_splitter = MockRAGSplitter()
    mock_splitter.set_chunks([MockChunk("c", {}, "c1")])
    mock_embedder = MockRAGEmbedder()
    mock_embedder.set_embeddings([(MockChunk("c",{},"c1"), [0.1])])

    error_raising_store = MockRAGVectorStore()
    error_raising_store.set_add_error(RuntimeError("Failed to add to vector store"))

    mock_plugin_manager_for_rag.get_plugin_instance.side_effect = lambda plugin_id, config=None: (
        mock_loader if plugin_id == loader_id else
        mock_splitter if plugin_id == splitter_id else
        mock_embedder if plugin_id == embedder_id else
        error_raising_store if plugin_id == store_id else
        None
    )
    result = await rag_manager_fixture.index_data_source(
        loader_id=loader_id, loader_source_uri="/path",
        splitter_id=splitter_id, embedder_id=embedder_id, vector_store_id=store_id
    )
    assert result["status"] == "error"
    assert "Failed to add to vector store" in result["message"]


@pytest.mark.asyncio
async def test_rag_manager_retrieve_from_query_success(
    rag_manager_fixture: RAGManager,
    mock_plugin_manager_for_rag: PluginManager
):
    retriever_id = "retriever1"
    query = "What is RAG?"

    mock_retrieved_chunks = [MockRetrievedChunk("RAG is...", {}, 0.9, "rc1")]
    mock_retriever = MockRAGRetriever()
    mock_retriever.set_retrieve_results(mock_retrieved_chunks)

    mock_plugin_manager_for_rag.get_plugin_instance.return_value = mock_retriever

    retrieved_results = await rag_manager_fixture.retrieve_from_query(
        query_text=query,
        retriever_id=retriever_id,
        top_k=1
    )

    assert len(retrieved_results) == 1
    assert retrieved_results[0].content == "RAG is..."
    mock_plugin_manager_for_rag.get_plugin_instance.assert_called_once_with(
        retriever_id,
        config={"plugin_manager": mock_plugin_manager_for_rag}
    )

@pytest.mark.asyncio
async def test_rag_manager_retrieve_from_query_retriever_load_failure(
    rag_manager_fixture: RAGManager,
    mock_plugin_manager_for_rag: PluginManager
):
    mock_plugin_manager_for_rag.get_plugin_instance.return_value = None

    results = await rag_manager_fixture.retrieve_from_query("query", "non_existent_retriever")
    assert results == []

@pytest.mark.asyncio
async def test_rag_manager_retrieve_from_query_retriever_error(
    rag_manager_fixture: RAGManager,
    mock_plugin_manager_for_rag: PluginManager
):
    error_raising_retriever = AsyncMock(spec=RetrieverPlugin)
    error_raising_retriever.retrieve = AsyncMock(side_effect=RuntimeError("Retrieval failed"))
    mock_plugin_manager_for_rag.get_plugin_instance.return_value = error_raising_retriever

    results = await rag_manager_fixture.retrieve_from_query("query", "error_retriever")
    assert results == []

@pytest.mark.asyncio
async def test_rag_manager_index_data_source_splitter_plugin_load_failure(
    rag_manager_fixture: RAGManager,
    mock_plugin_manager_for_rag: PluginManager
):
    """Test RAGManager indexing when the TextSplitter plugin fails to load."""
    loader_id, splitter_id, embedder_id, store_id = "loader1", "bad_splitter", "embedder1", "store1"
    source_uri = "/path/to/data"

    mock_loader = MockRAGLoader()
    mock_embedder = MockRAGEmbedder()
    mock_store = MockRAGVectorStore()

    async def get_plugin_side_effect(plugin_id_requested: str, config: Optional[Dict[str, Any]] = None):
        if plugin_id_requested == loader_id:
            return mock_loader
        if plugin_id_requested == splitter_id: # bad_splitter
            return None
        if plugin_id_requested == embedder_id:
            return mock_embedder
        if plugin_id_requested == store_id:
            return mock_store
        return None
    mock_plugin_manager_for_rag.get_plugin_instance.side_effect = get_plugin_side_effect

    result = await rag_manager_fixture.index_data_source(
        loader_id=loader_id,
        loader_source_uri=source_uri,
        splitter_id=splitter_id,
        embedder_id=embedder_id,
        vector_store_id=store_id
    )

    assert result["status"] == "error"
    assert "Splitter" in result["message"]
    assert "failed to load" in result["message"]

    mock_plugin_manager_for_rag.get_plugin_instance.assert_any_call(loader_id, config=None)
    mock_plugin_manager_for_rag.get_plugin_instance.assert_any_call(splitter_id, config=None)
    mock_plugin_manager_for_rag.get_plugin_instance.assert_any_call(embedder_id, config=None)
    mock_plugin_manager_for_rag.get_plugin_instance.assert_any_call(store_id, config=None)

@pytest.mark.asyncio
async def test_get_configured_plugin_wrong_type(
    rag_manager_fixture: RAGManager,
    mock_plugin_manager_for_rag: PluginManager,
    caplog
):
    """Test _get_configured_plugin when a plugin is loaded but is of an unexpected type."""
    plugin_id = "wrong_type_plugin"

    class NotADocLoader(Plugin): # Define here or import if it's a common mock
        plugin_id: str = "not_a_doc_loader_v1"
        description: str = "Not a doc loader"
        async def setup(self, config=None): pass
        async def teardown(self): pass

    mock_wrong_plugin = NotADocLoader()
    mock_plugin_manager_for_rag.get_plugin_instance.return_value = mock_wrong_plugin

    with caplog.at_level(logging.ERROR):
        loaded_plugin = await rag_manager_fixture._get_configured_plugin(
            plugin_id, DocumentLoaderPlugin, "DocumentLoader" # Specify the expected protocol
        )

    assert loaded_plugin is None
    assert f"Plugin '{plugin_id}' loaded but is not a valid DocumentLoader" in caplog.text
    assert f"expected type compatible with DocumentLoaderPlugin, got {type(mock_wrong_plugin).__name__}" in caplog.text
