from typing import Any, AsyncIterable, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import (
    Chunk,
    Document,
    EmbeddingVector,
    Plugin,
    RetrievedChunk,
)
from genie_tooling.document_loaders.abc import DocumentLoaderPlugin
from genie_tooling.embedding_generators.abc import EmbeddingGeneratorPlugin
from genie_tooling.rag.manager import RAGManager
from genie_tooling.retrievers.abc import RetrieverPlugin
from genie_tooling.text_splitters.abc import TextSplitterPlugin
from genie_tooling.vector_stores.abc import VectorStorePlugin


# --- Concrete Implementations for Testing (to avoid Protocol instantiation errors) ---
class _ConcreteDocument:
    def __init__(self, content: str, metadata: Dict, id: Optional[str] = None):
        self.content = content
        self.metadata = metadata
        self.id = id
class _ConcreteChunk:
    def __init__(self, content: str, metadata: Dict, id: Optional[str] = None):
        self.content = content
        self.metadata = metadata
        self.id = id
class _ConcreteRetrievedChunk(_ConcreteChunk):
    def __init__(self, content: str, metadata: Dict, score: float, id: Optional[str] = None, rank: Optional[int] = None):
        super().__init__(content, metadata, id)
        self.score = score
        self.rank = rank

# --- Mock RAG Components ---
class MockRAGComponent(Plugin):
    def __init__(self, plugin_id, should_fail=False, fail_msg="Simulated failure"):
        self._plugin_id = plugin_id
        self._should_fail = should_fail
        self._fail_msg = fail_msg
    @property
    def plugin_id(self) -> str:
        return self._plugin_id
    async def setup(self, config=None):
        if self._should_fail:
            raise RuntimeError(self._fail_msg)
    async def teardown(self):
        pass

class MockLoader(MockRAGComponent, DocumentLoaderPlugin):
    async def load(self, source_uri: str, config=None) -> AsyncIterable[Document]:
        if self._should_fail:
            raise RuntimeError(self._fail_msg)
        yield _ConcreteDocument(content="doc content", metadata={"source": source_uri}, id="doc1")

class MockSplitter(MockRAGComponent, TextSplitterPlugin):
    async def split(self, documents: AsyncIterable[Document], config=None) -> AsyncIterable[Chunk]:
        if self._should_fail:
            raise RuntimeError(self._fail_msg)
        async for _ in documents:
            pass
        yield _ConcreteChunk(content="chunk content", metadata={}, id="chunk1")

class MockEmbedder(MockRAGComponent, EmbeddingGeneratorPlugin):
    async def embed(self, chunks: AsyncIterable[Chunk], config=None) -> AsyncIterable[Tuple[Chunk, EmbeddingVector]]:
        if self._should_fail:
            raise RuntimeError(self._fail_msg)
        async for chunk in chunks:
            yield chunk, [0.1, 0.2]

class MockVectorStore(MockRAGComponent, VectorStorePlugin):
    async def add(self, embeddings: AsyncIterable[Tuple[Chunk, EmbeddingVector]], config=None) -> Dict[str, Any]:
        if self._should_fail:
            raise RuntimeError(self._fail_msg)
        count = 0
        async for _ in embeddings:
            count += 1
        return {"added_count": count, "errors": []}
    async def search(self, query_embedding, top_k, filter_metadata=None, config=None) -> List[RetrievedChunk]: return []
    async def delete(self, ids=None, filter_metadata=None, delete_all=False, config=None) -> bool: return True

class MockRetriever(MockRAGComponent, RetrieverPlugin):
    async def retrieve(self, query: str, top_k: int, config=None) -> List[RetrievedChunk]:
        if self._should_fail:
            raise RuntimeError(self._fail_msg)
        return [_ConcreteRetrievedChunk(id="retrieved1", content="retrieved content", score=0.9, metadata={})]

# --- Fixtures ---
@pytest.fixture
def mock_plugin_manager() -> MagicMock:
    return MagicMock(spec=PluginManager)

@pytest.fixture
def rag_manager(mock_plugin_manager: MagicMock) -> RAGManager:
    return RAGManager(plugin_manager=mock_plugin_manager)

# --- Tests ---
@pytest.mark.asyncio
async def test_index_data_source_success(rag_manager: RAGManager, mock_plugin_manager: MagicMock):
    async def get_instance_side_effect(plugin_id, config=None):
        if plugin_id == "loader1":
            return MockLoader(plugin_id)
        if plugin_id == "splitter1":
            return MockSplitter(plugin_id)
        if plugin_id == "embedder1":
            return MockEmbedder(plugin_id)
        if plugin_id == "store1":
            return MockVectorStore(plugin_id)
        return None
    mock_plugin_manager.get_plugin_instance.side_effect = get_instance_side_effect
    result = await rag_manager.index_data_source("loader1", "/path", "splitter1", "embedder1", "store1")
    assert result["status"] == "success"
    assert result["added_count"] == 1

@pytest.mark.asyncio
async def test_index_data_source_component_load_failure(rag_manager: RAGManager, mock_plugin_manager: MagicMock):
    async def get_instance_side_effect(plugin_id, config=None):
        if plugin_id == "loader1":
            return MockLoader(plugin_id)
        if plugin_id == "splitter1":
            return None
        if plugin_id == "embedder1":
            return MockEmbedder(plugin_id)
        if plugin_id == "store1":
            return MockVectorStore(plugin_id)
        return None
    mock_plugin_manager.get_plugin_instance.side_effect = get_instance_side_effect
    result = await rag_manager.index_data_source("loader1", "/path", "splitter1", "embedder1", "store1")
    assert result["status"] == "error"
    assert "One or more RAG components failed to load: TextSplitter" in result["message"]

@pytest.mark.asyncio
async def test_index_data_source_pipeline_error(rag_manager: RAGManager, mock_plugin_manager: MagicMock):
    async def get_instance_side_effect(plugin_id, config=None):
        if plugin_id == "loader1":
            return MockLoader(plugin_id, should_fail=True, fail_msg="Disk read error")
        if plugin_id == "splitter1":
            return MockSplitter(plugin_id)
        if plugin_id == "embedder1":
            return MockEmbedder(plugin_id)
        if plugin_id == "store1":
            return MockVectorStore(plugin_id)
        return None
    mock_plugin_manager.get_plugin_instance.side_effect = get_instance_side_effect
    result = await rag_manager.index_data_source("loader1", "/path", "splitter1", "embedder1", "store1")
    assert result["status"] == "error"
    assert "Indexing failed: Disk read error" in result["message"]

@pytest.mark.asyncio
async def test_retrieve_from_query_success(rag_manager: RAGManager, mock_plugin_manager: MagicMock):
    mock_retriever = MockRetriever("retriever1")
    mock_plugin_manager.get_plugin_instance.return_value = mock_retriever
    results = await rag_manager.retrieve_from_query("query", "retriever1")
    assert len(results) == 1
    assert results[0].id == "retrieved1"

@pytest.mark.asyncio
async def test_retrieve_from_query_retriever_load_failure(rag_manager: RAGManager, mock_plugin_manager: MagicMock):
    mock_plugin_manager.get_plugin_instance.return_value = None
    results = await rag_manager.retrieve_from_query("query", "non_existent_retriever")
    assert results == []

@pytest.mark.asyncio
async def test_retrieve_from_query_retriever_error(rag_manager: RAGManager, mock_plugin_manager: MagicMock):
    mock_retriever = MockRetriever("retriever1", should_fail=True, fail_msg="Retrieve failed")
    mock_plugin_manager.get_plugin_instance.return_value = mock_retriever
    results = await rag_manager.retrieve_from_query("query", "retriever1")
    assert results == []
