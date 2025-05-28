"""Unit tests for the default implementations in rag.plugins.abc."""
import logging
from typing import Any, AsyncIterable, Dict, List, Optional, Tuple

import pytest
from genie_tooling.core.types import (
    Chunk,
    Document,
    EmbeddingVector,
    Plugin,
)
from genie_tooling.rag.plugins.abc import (
    DocumentLoaderPlugin,
    EmbeddingGeneratorPlugin,
    RetrieverPlugin,
    TextSplitterPlugin,
    VectorStorePlugin,
)

# Minimal concrete implementations for RAG plugin protocols

class DefaultImplDocLoader(DocumentLoaderPlugin, Plugin):
    plugin_id: str = "default_impl_doc_loader_v1"
    description: str = "Default doc loader."
    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass
    async def teardown(self) -> None: pass

class DefaultImplTextSplitter(TextSplitterPlugin, Plugin):
    plugin_id: str = "default_impl_text_splitter_v1"
    description: str = "Default text splitter."
    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass
    async def teardown(self) -> None: pass

class DefaultImplEmbedGenerator(EmbeddingGeneratorPlugin, Plugin):
    plugin_id: str = "default_impl_embed_generator_v1"
    description: str = "Default embed generator."
    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass
    async def teardown(self) -> None: pass

class DefaultImplVectorStore(VectorStorePlugin, Plugin):
    plugin_id: str = "default_impl_vector_store_v1"
    description: str = "Default vector store."
    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass
    async def teardown(self) -> None: pass

class DefaultImplRetriever(RetrieverPlugin, Plugin):
    plugin_id: str = "default_impl_retriever_v1"
    description: str = "Default retriever."
    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass
    async def teardown(self) -> None: pass

@pytest.fixture
async def default_doc_loader_fixture() -> DefaultImplDocLoader: return DefaultImplDocLoader() # Renamed
@pytest.fixture
async def default_text_splitter_fixture() -> DefaultImplTextSplitter: return DefaultImplTextSplitter() # Renamed
@pytest.fixture
async def default_embed_generator_fixture() -> DefaultImplEmbedGenerator: return DefaultImplEmbedGenerator() # Renamed
@pytest.fixture
async def default_vector_store_fixture() -> DefaultImplVectorStore: return DefaultImplVectorStore() # Renamed
@pytest.fixture
async def default_retriever_fixture() -> DefaultImplRetriever: return DefaultImplRetriever() # Renamed


async def collect_async_iterable(aiter: AsyncIterable[Any]) -> List[Any]:
    return [item async for item in aiter]

@pytest.mark.asyncio
async def test_doc_loader_default_load(default_doc_loader_fixture: DefaultImplDocLoader, caplog: pytest.LogCaptureFixture):
    default_doc_loader = await default_doc_loader_fixture # Await the fixture
    caplog.set_level(logging.WARNING)
    results = await collect_async_iterable(default_doc_loader.load("source_uri"))
    assert results == []
    assert any(f"DocumentLoaderPlugin '{default_doc_loader.plugin_id}' load method not fully implemented." in rec.message for rec in caplog.records)

@pytest.mark.asyncio
async def test_text_splitter_default_split(default_text_splitter_fixture: DefaultImplTextSplitter, caplog: pytest.LogCaptureFixture):
    default_text_splitter = await default_text_splitter_fixture # Await the fixture
    caplog.set_level(logging.WARNING)
    async def dummy_doc_stream() -> AsyncIterable[Document]:
        if False: yield # Make it an async generator
        return
    results = await collect_async_iterable(default_text_splitter.split(dummy_doc_stream()))
    assert results == []
    assert any(f"TextSplitterPlugin '{default_text_splitter.plugin_id}' split method not fully implemented." in rec.message for rec in caplog.records)

@pytest.mark.asyncio
async def test_embed_generator_default_embed(default_embed_generator_fixture: DefaultImplEmbedGenerator, caplog: pytest.LogCaptureFixture):
    default_embed_generator = await default_embed_generator_fixture # Await the fixture
    caplog.set_level(logging.WARNING)
    async def dummy_chunk_stream() -> AsyncIterable[Chunk]:
        if False: yield
        return
    results = await collect_async_iterable(default_embed_generator.embed(dummy_chunk_stream()))
    assert results == []
    assert any(f"EmbeddingGeneratorPlugin '{default_embed_generator.plugin_id}' embed method not fully implemented." in rec.message for rec in caplog.records)

@pytest.mark.asyncio
async def test_vector_store_default_add(default_vector_store_fixture: DefaultImplVectorStore, caplog: pytest.LogCaptureFixture):
    default_vector_store = await default_vector_store_fixture # Await the fixture
    caplog.set_level(logging.WARNING)
    async def dummy_embedding_stream() -> AsyncIterable[Tuple[Chunk, EmbeddingVector]]:
        if False: yield
        return
    result_dict = await default_vector_store.add(dummy_embedding_stream())
    assert result_dict == {"added_count": 0, "errors": ["Not implemented"]}
    assert any(f"VectorStorePlugin '{default_vector_store.plugin_id}' add method not fully implemented." in rec.message for rec in caplog.records)

@pytest.mark.asyncio
async def test_vector_store_default_search(default_vector_store_fixture: DefaultImplVectorStore, caplog: pytest.LogCaptureFixture):
    default_vector_store = await default_vector_store_fixture # Await the fixture
    caplog.set_level(logging.WARNING)
    results = await default_vector_store.search(query_embedding=[0.1, 0.2], top_k=5)
    assert results == []
    assert any(f"VectorStorePlugin '{default_vector_store.plugin_id}' search method not fully implemented." in rec.message for rec in caplog.records)

@pytest.mark.asyncio
async def test_vector_store_default_delete(default_vector_store_fixture: DefaultImplVectorStore, caplog: pytest.LogCaptureFixture):
    default_vector_store = await default_vector_store_fixture # Await the fixture
    caplog.set_level(logging.WARNING)
    result_bool = await default_vector_store.delete(ids=["id1"])
    assert result_bool is False
    assert any(f"VectorStorePlugin '{default_vector_store.plugin_id}' delete method not fully implemented." in rec.message for rec in caplog.records)

@pytest.mark.asyncio
async def test_retriever_default_retrieve(default_retriever_fixture: DefaultImplRetriever, caplog: pytest.LogCaptureFixture):
    default_retriever = await default_retriever_fixture # Await the fixture
    caplog.set_level(logging.WARNING)
    results = await default_retriever.retrieve(query="test query", top_k=3)
    assert results == []
    assert any(f"RetrieverPlugin '{default_retriever.plugin_id}' retrieve method not fully implemented." in rec.message for rec in caplog.records)

