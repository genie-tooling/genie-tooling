"""Unit tests for the RAGManager."""
import logging
from typing import Any, AsyncIterable, Dict, List, Optional

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


# --- Mock RAG Component Implementations (Updated setup for data/failure priming) ---
class MockDocument(Document):
    def __init__(self, content: str, metadata: Dict[str, Any], id: Optional[str] = None):
        self.content = content; self.metadata = metadata; self.id = id
class MockChunk(Chunk):
    def __init__(self, content: str, metadata: Dict[str, Any], id: Optional[str] = None):
        self.content = content; self.metadata = metadata; self.id = id
class MockRetrievedChunk(RetrievedChunk, MockChunk):
    def __init__(self, content: str, metadata: Dict[str, Any], score: float, id: Optional[str] = None, rank: Optional[int] = None):
        super().__init__(content, metadata, id); self.score = score; self.rank = rank

class MockRAGLoader(DocumentLoaderPlugin):
    plugin_id: str = "mock_rag_loader_v1"; description: str = "Mock RAG Loader"
    def __init__(self):
        self._documents_to_yield: List[Document] = []
        self._fail_on_load_msg: Optional[str] = None
    async def setup(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        if cfg.get("test_fail_load"):
            self._fail_on_load_msg = cfg.get("test_fail_load_msg", "Configured load failure in MockRAGLoader")
        if "_test_documents_to_yield" in cfg:
            self.set_documents(cfg["_test_documents_to_yield"])
    async def load(self, source_uri: str, config: Optional[Dict[str, Any]]=None) -> AsyncIterable[Document]:
        if self._fail_on_load_msg: raise RuntimeError(self._fail_on_load_msg)
        for doc in self._documents_to_yield: yield doc
    def set_documents(self, docs: List[Document]): self._documents_to_yield = docs
    async def teardown(self): pass

class MockRAGSplitter(TextSplitterPlugin):
    plugin_id: str = "mock_rag_splitter_v1"; description: str = "Mock RAG Splitter"
    def __init__(self):
        self._chunks_to_yield: List[Chunk] = []
        self._fail_on_split_msg: Optional[str] = None
    async def setup(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        if cfg.get("test_fail_split"):
            self._fail_on_split_msg = cfg.get("test_fail_split_msg", "Configured split failure in MockRAGSplitter")
        if "_test_chunks_to_yield" in cfg:
            self.set_chunks(cfg["_test_chunks_to_yield"])
    async def split(self, documents: AsyncIterable[Document], config: Optional[Dict[str, Any]]=None) -> AsyncIterable[Chunk]:
        async for _d in documents: pass
        if self._fail_on_split_msg: raise RuntimeError(self._fail_on_split_msg)
        for ch in self._chunks_to_yield: yield ch
    def set_chunks(self, chunks: List[Chunk]): self._chunks_to_yield = chunks
    async def teardown(self): pass

class MockRAGEmbedder(EmbeddingGeneratorPlugin):
    plugin_id: str = "mock_rag_embedder_v1"; description: str = "Mock RAG Embedder"
    def __init__(self):
        self._embeddings: List[tuple[Chunk, EmbeddingVector]] = []
        self._fail_on_embed_msg: Optional[str] = None
    async def setup(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        if cfg.get("test_fail_embed"):
            self._fail_on_embed_msg = cfg.get("test_fail_embed_msg", "Configured embed failure in MockRAGEmbedder")
        if "_test_embeddings_to_yield" in cfg:
            self.set_embeddings(cfg["_test_embeddings_to_yield"])
    async def embed(self, chunks: AsyncIterable[Chunk], config: Optional[Dict[str, Any]]=None) -> AsyncIterable[tuple[Chunk, EmbeddingVector]]:
        async for _ch in chunks: pass
        if self._fail_on_embed_msg: raise RuntimeError(self._fail_on_embed_msg)
        for item in self._embeddings: yield item
    def set_embeddings(self, embs: List[tuple[Chunk, EmbeddingVector]]): self._embeddings = embs
    async def teardown(self): pass

class MockRAGVectorStore(VectorStorePlugin):
    plugin_id: str = "mock_rag_vector_store_v1"; description: str = "Mock RAG Vector Store"
    def __init__(self):
        self._add_result_val: Dict[str, Any]={"added_count":0,"errors":[]}
        self._search_res: List[RetrievedChunk]=[]
        self._fail_on_add_msg: Optional[str] = None
        self._return_preset_add_result = False
    async def setup(self, config: Optional[Dict[str, Any]] = None):
        self._return_preset_add_result = False
        cfg = config or {}
        if cfg.get("test_fail_add"):
            self._fail_on_add_msg = cfg.get("test_fail_add_msg", "Configured add failure")
        if "_test_preset_add_result_config" in cfg: # For success test
            self.set_add_result(cfg["_test_preset_add_result_config"])
    async def add(self, embeddings: AsyncIterable[tuple[Chunk, EmbeddingVector]], config: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        if self._fail_on_add_msg:
            async for _e in embeddings: pass
            raise RuntimeError(self._fail_on_add_msg)

        num_items_in_stream = 0
        async for _ in embeddings:
            num_items_in_stream += 1

        if self._return_preset_add_result:
            if num_items_in_stream > 0:
                # Return the preset count only if actual items were processed
                return {"added_count": self._add_result_val.get("added_count",0) if num_items_in_stream > 0 else 0,
                        "errors": self._add_result_val.get("errors", [])}
            else:
                return {"added_count": 0, "errors": self._add_result_val.get("errors", [])}

        return {"added_count": num_items_in_stream, "errors": []}

    async def search(self, query_embedding: EmbeddingVector, top_k: int, filter_metadata: Optional[Dict[str, Any]]=None, config: Optional[Dict[str, Any]]=None) -> List[RetrievedChunk]: return self._search_res[:top_k]
    async def delete(self, ids: Optional[List[str]]=None, filter_metadata: Optional[Dict[str, Any]]=None, delete_all: bool=False, config: Optional[Dict[str, Any]]=None) -> bool: return True
    def set_add_result(self, res: Dict[str, Any]):
        self._add_result_val = res
        self._return_preset_add_result = True
    def set_search_results(self, res: List[RetrievedChunk]): self._search_res = res
    async def teardown(self): pass

class MockRAGRetriever(RetrieverPlugin):
    plugin_id: str = "mock_rag_retriever_v1"; description: str = "Mock RAG Retriever"
    def __init__(self):
        self._retrieve_res: List[RetrievedChunk] = []
        self._fail_on_retrieve_msg: Optional[str] = None
    async def setup(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        if cfg.get("test_fail_retrieve"):
            self._fail_on_retrieve_msg = cfg.get("test_fail_retrieve_msg", "Configured retrieve failure")
        if "_test_set_retrieve_results" in cfg:
            self.set_retrieve_results(cfg["_test_set_retrieve_results"])
    async def retrieve(self, query: str, top_k: int, config: Optional[Dict[str, Any]]=None) -> List[RetrievedChunk]:
        if self._fail_on_retrieve_msg: raise RuntimeError(self._fail_on_retrieve_msg)
        return self._retrieve_res[:top_k]
    def set_retrieve_results(self, res: List[RetrievedChunk]): self._retrieve_res = res
    async def teardown(self): pass

@pytest.fixture
def mock_plugin_manager_for_rag(mocker) -> PluginManager:
    pm = mocker.MagicMock(spec=PluginManager)
    pm.list_discovered_plugin_classes = mocker.MagicMock(return_value={})
    return pm

@pytest.fixture
def rag_manager_fixture(mock_plugin_manager_for_rag: PluginManager) -> RAGManager:
    return RAGManager(plugin_manager=mock_plugin_manager_for_rag)

@pytest.mark.asyncio
async def test_rag_manager_index_data_source_success(rag_manager_fixture: RAGManager, mock_plugin_manager_for_rag: PluginManager):
    loader_id, splitter_id, embedder_id, store_id = "loader1", "splitter1", "embedder1", "store1"
    source_uri = "/path/to/data"

    mock_docs = [MockDocument("doc1 content", {"s": "d1"}, "doc1")]
    mock_chunks = [MockChunk("chunk1 content", {"s": "c1"}, "chunk1")]
    mock_embeddings = [(mock_chunks[0], [0.1, 0.2])]
    expected_add_result = {"added_count": 1, "errors": []}

    mock_plugin_manager_for_rag.list_discovered_plugin_classes.return_value = {
        loader_id: MockRAGLoader, splitter_id: MockRAGSplitter,
        embedder_id: MockRAGEmbedder, store_id: MockRAGVectorStore
    }

    result = await rag_manager_fixture.index_data_source(
        loader_id, source_uri, splitter_id, embedder_id, store_id,
        loader_config={"_test_documents_to_yield": mock_docs},
        splitter_config={"_test_chunks_to_yield": mock_chunks},
        embedder_config={"_test_embeddings_to_yield": mock_embeddings},
        vector_store_config={"_test_preset_add_result_config": expected_add_result}
    )
    assert result["status"] == "success", f"Result: {result}"
    assert result["added_count"] == 1
    assert not result["store_errors"]

@pytest.mark.asyncio
async def test_rag_manager_index_data_source_component_load_failure(rag_manager_fixture: RAGManager, mock_plugin_manager_for_rag: PluginManager):
    mock_plugin_manager_for_rag.list_discovered_plugin_classes.return_value = {
        "splitter1": MockRAGSplitter, "embedder1": MockRAGEmbedder, "store1": MockRAGVectorStore
    }
    result = await rag_manager_fixture.index_data_source("loader1", "/path", "splitter1", "embedder1", "store1")
    assert result["status"] == "error"
    assert "One or more RAG components failed to load: DocumentLoader." in result["message"]

@pytest.mark.asyncio
async def test_rag_manager_index_data_source_loader_pipeline_error(rag_manager_fixture: RAGManager, mock_plugin_manager_for_rag: PluginManager):
    mock_plugin_manager_for_rag.list_discovered_plugin_classes.return_value = {
        "loader1": MockRAGLoader, "splitter1": MockRAGSplitter,
        "embedder1": MockRAGEmbedder, "store1": MockRAGVectorStore
    }
    result = await rag_manager_fixture.index_data_source(
        "loader1", "/path", "splitter1", "embedder1", "store1",
        loader_config={"test_fail_load": True, "test_fail_load_msg": "Failed to load documents"}
    )
    assert result["status"] == "error"
    assert "Indexing failed: Failed to load documents" in result["message"]

@pytest.mark.asyncio
async def test_rag_manager_index_data_source_splitter_pipeline_error(rag_manager_fixture: RAGManager, mock_plugin_manager_for_rag: PluginManager):
    mock_loader_config = {"_test_documents_to_yield": [MockDocument("content", {}, "id1")]}
    mock_plugin_manager_for_rag.list_discovered_plugin_classes.return_value = {
        "loader1": MockRAGLoader, "splitter1": MockRAGSplitter,
        "embedder1": MockRAGEmbedder, "store1": MockRAGVectorStore
    }
    result = await rag_manager_fixture.index_data_source(
        "loader1", "/path", "splitter1", "embedder1", "store1",
        loader_config=mock_loader_config,
        splitter_config={"test_fail_split": True, "test_fail_split_msg": "Failed to split"}
    )
    assert result["status"] == "error"
    assert "Indexing failed: Failed to split" in result["message"]

@pytest.mark.asyncio
async def test_rag_manager_index_data_source_embedder_pipeline_error(rag_manager_fixture: RAGManager, mock_plugin_manager_for_rag: PluginManager):
    mock_loader_config = {"_test_documents_to_yield": [MockDocument("content", {}, "id1")]}
    mock_splitter_config = {"_test_chunks_to_yield": [MockChunk("chunk_content", {}, "cid1")]}
    mock_plugin_manager_for_rag.list_discovered_plugin_classes.return_value = {
        "loader1": MockRAGLoader, "splitter1": MockRAGSplitter,
        "embedder1": MockRAGEmbedder, "store1": MockRAGVectorStore
    }
    result = await rag_manager_fixture.index_data_source(
        "loader1", "/path", "splitter1", "embedder1", "store1",
        loader_config=mock_loader_config,
        splitter_config=mock_splitter_config,
        embedder_config={"test_fail_embed": True, "test_fail_embed_msg": "Failed to embed"}
    )
    assert result["status"] == "error"
    assert "Indexing failed: Failed to embed" in result["message"]

@pytest.mark.asyncio
async def test_rag_manager_index_data_source_vector_store_pipeline_error(rag_manager_fixture: RAGManager, mock_plugin_manager_for_rag: PluginManager):
    mock_loader_config = {"_test_documents_to_yield": [MockDocument("content", {}, "id1")]}
    mock_splitter_config = {"_test_chunks_to_yield": [MockChunk("c", {}, "c1")]}
    mock_embedder_config = {"_test_embeddings_to_yield": [(MockChunk("c",{},"c1"), [0.1])]}
    mock_plugin_manager_for_rag.list_discovered_plugin_classes.return_value = {
        "loader1": MockRAGLoader, "splitter1": MockRAGSplitter,
        "embedder1": MockRAGEmbedder, "store1": MockRAGVectorStore
    }
    result = await rag_manager_fixture.index_data_source(
        "loader1", "/path", "splitter1", "embedder1", "store1",
        loader_config=mock_loader_config,
        splitter_config=mock_splitter_config,
        embedder_config=mock_embedder_config,
        vector_store_config={"test_fail_add": True, "test_fail_add_msg": "Failed to store"}
    )
    assert result["status"] == "error"
    assert "Indexing failed: Failed to store" in result["message"]

@pytest.mark.asyncio
async def test_rag_manager_retrieve_from_query_success(rag_manager_fixture: RAGManager, mock_plugin_manager_for_rag: PluginManager):
    retriever_id = "retriever1"; query = "What is RAG?"
    mock_retrieved_chunks = [MockRetrievedChunk("RAG is...", {}, 0.9, "rc1")]

    mock_plugin_manager_for_rag.list_discovered_plugin_classes.return_value = {retriever_id: MockRAGRetriever}
    retrieved_results = await rag_manager_fixture.retrieve_from_query(
        query, retriever_id, top_k=1,
        retriever_config={"_test_set_retrieve_results": mock_retrieved_chunks}
    )
    assert len(retrieved_results) == 1
    assert retrieved_results[0].content == "RAG is..."

@pytest.mark.asyncio
async def test_rag_manager_retrieve_from_query_retriever_load_failure(rag_manager_fixture: RAGManager, mock_plugin_manager_for_rag: PluginManager):
    mock_plugin_manager_for_rag.list_discovered_plugin_classes.return_value = {}
    results = await rag_manager_fixture.retrieve_from_query("query", "non_existent_retriever")
    assert results == []

@pytest.mark.asyncio
async def test_rag_manager_retrieve_from_query_retriever_error(rag_manager_fixture: RAGManager, mock_plugin_manager_for_rag: PluginManager):
    mock_plugin_manager_for_rag.list_discovered_plugin_classes.return_value = {"error_retriever": MockRAGRetriever}
    results = await rag_manager_fixture.retrieve_from_query(
        "query", "error_retriever",
        retriever_config={"test_fail_retrieve": True, "test_fail_retrieve_msg": "Retrieval failed"}
    )
    assert results == []

@pytest.mark.asyncio
async def test_rag_manager_index_data_source_splitter_plugin_load_failure(rag_manager_fixture: RAGManager, mock_plugin_manager_for_rag: PluginManager):
    mock_plugin_manager_for_rag.list_discovered_plugin_classes.return_value = {
        "loader1": MockRAGLoader,
        "embedder1": MockRAGEmbedder, "store1": MockRAGVectorStore
    }
    result = await rag_manager_fixture.index_data_source("loader1", "/path", "bad_splitter", "embedder1", "store1")
    assert result["status"] == "error"
    assert "One or more RAG components failed to load: TextSplitter." in result["message"]

@pytest.mark.asyncio
async def test_get_plugin_instance_for_rag_wrong_type(rag_manager_fixture: RAGManager, mock_plugin_manager_for_rag: PluginManager, caplog):
    class NotADocLoader(Plugin):
        plugin_id: str = "not_a_doc_loader_v1"; description: str = "Not a doc loader"
        async def setup(self, config: Optional[Dict[str, Any]] = None): pass
        async def teardown(self): pass

    mock_plugin_manager_for_rag.list_discovered_plugin_classes.return_value = {"wrong_type_plugin": NotADocLoader}

    with caplog.at_level(logging.ERROR):
        loaded_plugin = await rag_manager_fixture._get_plugin_instance_for_rag(
            "wrong_type_plugin", DocumentLoaderPlugin, "DocumentLoader"
        )
    assert loaded_plugin is None
    assert "Instantiated plugin 'wrong_type_plugin' is not a valid DocumentLoader" in caplog.text
