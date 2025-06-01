import asyncio
import logging
from pathlib import Path
from typing import Any, AsyncIterable, Dict, Tuple
from unittest.mock import MagicMock, patch

import pytest

from genie_tooling.core.types import Chunk, EmbeddingVector
from genie_tooling.vector_stores.impl.chromadb_store import (
    ChromaDBVectorStore,
)

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMA_CLIENT_SPEC = chromadb.Client if chromadb else Any
    CHROMA_COLLECTION_SPEC = chromadb.api.models.Collection.Collection if chromadb else Any
except ImportError:
    chromadb = None; ChromaSettings = None; CHROMA_CLIENT_SPEC = Any; CHROMA_COLLECTION_SPEC = Any

@pytest.fixture
def mock_chroma_collection() -> MagicMock:
    collection = MagicMock(spec=CHROMA_COLLECTION_SPEC); collection.name = "mock_coll"
    collection.add = MagicMock(); collection.query = MagicMock(return_value={"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]})
    collection.delete = MagicMock(); collection.count = MagicMock(return_value=0)
    return collection

@pytest.fixture
def mock_chroma_client(mock_chroma_collection: MagicMock) -> MagicMock:
    client = MagicMock(); client.get_or_create_collection = MagicMock(return_value=mock_chroma_collection)
    client.delete_collection = MagicMock()
    return client

@pytest.fixture
def patch_chromadb_constructors(mock_chroma_client: MagicMock):
    with patch("genie_tooling.vector_stores.impl.chromadb_store.chromadb.Client", return_value=mock_chroma_client) as mock_eph, \
         patch("chromadb.PersistentClient", return_value=mock_chroma_client) as mock_per, \
         patch("chromadb.HttpClient", return_value=mock_chroma_client) as mock_http:
        yield {"ephemeral": mock_eph, "persistent": mock_per, "http": mock_http}

@pytest.fixture
async def chromadb_store_fixture(request) -> ChromaDBVectorStore:
    store = ChromaDBVectorStore()
    async def finalizer_async():
        if hasattr(store, "_client") and store._client is not None: await store.teardown()
    def finalizer_sync():
        try: loop = asyncio.get_event_loop_policy().get_event_loop()
        except RuntimeError: loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
        if loop.is_running(): asyncio.ensure_future(finalizer_async(), loop=loop)
        else: loop.run_until_complete(finalizer_async())
    request.addfinalizer(finalizer_sync)
    return store

class SimpleChunk(Chunk):
    def __init__(self, id: str, content: str, metadata: Dict[str, Any]):
        self.id=id; self.content=content; self.metadata=metadata
async def sample_embeddings_stream(count=2,dim=3,start_idx=0) -> AsyncIterable[Tuple[Chunk,EmbeddingVector]]:
    for i in range(count): yield SimpleChunk(f"c{start_idx+i}",f"Content{start_idx+i}",{"s":"t","idx":start_idx+i}), [float(start_idx+i+j*0.1) for j in range(dim)]

@pytest.mark.asyncio
async def test_chromadb_setup_default_path(chromadb_store_fixture: ChromaDBVectorStore, patch_chromadb_constructors, mock_chroma_client, caplog):
    store = await chromadb_store_fixture
    caplog.set_level(logging.INFO)
    collection_name = "my_default_path_collection"
    with patch.object(Path, "mkdir") as mock_mkdir:
        await store.setup(config={"collection_name": collection_name}) # 'path' is NOT in config
    expected_default_path = Path(f"./.genie_data/chromadb/{collection_name}")
    patch_chromadb_constructors["persistent"].assert_called_once()
    called_args, called_kwargs = patch_chromadb_constructors["persistent"].call_args
    assert called_kwargs.get("path") == str(expected_default_path)
    mock_mkdir.assert_any_call(parents=True, exist_ok=True)
    assert f"{store.plugin_id}: 'path' not provided in config, defaulting to persistent storage at '{str(expected_default_path)}'." in caplog.text
    assert store._path == str(expected_default_path)

@pytest.mark.asyncio
async def test_chromadb_setup_ephemeral_if_path_is_none_in_config(chromadb_store_fixture: ChromaDBVectorStore, patch_chromadb_constructors, caplog):
    store = await chromadb_store_fixture
    caplog.set_level(logging.INFO)
    await store.setup(config={"collection_name": "ephemeral_test", "path": None}) # 'path' IS in config, value is None
    patch_chromadb_constructors["ephemeral"].assert_called_once()
    assert f"{store.plugin_id}: 'path' explicitly set to None in config, using ephemeral client." in caplog.text
    assert store._path is None

@pytest.mark.asyncio
async def test_chromadb_setup_persistent(chromadb_store_fixture: ChromaDBVectorStore, patch_chromadb_constructors, mock_chroma_client, mock_chroma_collection, tmp_path: Path, caplog):
    chromadb_store = await chromadb_store_fixture; caplog.set_level(logging.INFO)
    db_path = tmp_path / "chroma_test_data"; collection_name = "persistent_test"; mock_chroma_collection.name = collection_name
    await chromadb_store.setup(config={"collection_name": collection_name, "path": str(db_path)})
    patch_chromadb_constructors["persistent"].assert_called_once()
    assert chromadb_store._client is mock_chroma_client; assert chromadb_store._collection is mock_chroma_collection
    assert f"{chromadb_store.plugin_id}: Using persistent ChromaDB at: {str(db_path)}" in caplog.text
    assert db_path.is_dir()

@pytest.mark.asyncio
async def test_chromadb_add_successful(chromadb_store_fixture: ChromaDBVectorStore, patch_chromadb_constructors, mock_chroma_collection):
    chromadb_store = await chromadb_store_fixture; await chromadb_store.setup(config={"collection_name": "add_test_coll", "path":None})
    mock_chroma_collection.add.return_value = None
    embeddings_data = [(SimpleChunk(f"c{i}",f"tc{i}",{"m_k":f"m_v{i}"}),[float(i)]*2) for i in range(3)]
    async def stream_data():
        for item in embeddings_data: yield item
    async def mock_run_in_executor(executor, func_partial): func_partial(); return None
    with patch("asyncio.BaseEventLoop.run_in_executor", side_effect=mock_run_in_executor):
        result = await chromadb_store.add(stream_data(), config={"batch_size": 2})
    assert result["added_count"] == 3; assert not result["errors"]; assert mock_chroma_collection.add.call_count == 2

@pytest.mark.asyncio
async def test_chromadb_setup_http(chromadb_store_fixture: ChromaDBVectorStore, patch_chromadb_constructors, mock_chroma_client, mock_chroma_collection, caplog):
    chromadb_store = await chromadb_store_fixture; caplog.set_level(logging.INFO)
    collection_name = "http_test"; mock_chroma_collection.name = collection_name
    await chromadb_store.setup(config={"collection_name": collection_name, "host": "localhost", "port": 8000})
    patch_chromadb_constructors["http"].assert_called_once()
    assert chromadb_store._client is mock_chroma_client; assert chromadb_store._collection is mock_chroma_collection
    assert f"{chromadb_store.plugin_id}: Connecting to remote ChromaDB: localhost:8000" in caplog.text

@pytest.mark.asyncio
async def test_chromadb_search_successful(chromadb_store_fixture: ChromaDBVectorStore, patch_chromadb_constructors, mock_chroma_collection):
    chromadb_store = await chromadb_store_fixture
    await chromadb_store.setup(config={"collection_name": "search_coll", "hnsw_space": "cosine", "path":None})
    mock_query_results = {"ids": [["id1", "id2"]],"documents": [["doc1", "doc2"]],"metadatas": [[{"id":"id1"}, {"id":"id2"}]],"distances": [[0.1, 0.2]]}
    mock_chroma_collection.query.return_value = mock_query_results; mock_chroma_collection.count.return_value = 2
    results = await chromadb_store.search([0.5]*3, top_k=2, filter_metadata={"src": "s1"})
    mock_chroma_collection.query.assert_called_once()
    call_kwargs = mock_chroma_collection.query.call_args.kwargs
    assert call_kwargs["where"] == {"src": "s1"}
    assert len(results) == 2; assert results[0].id == "id1"

@pytest.mark.asyncio
async def test_chromadb_delete_all(chromadb_store_fixture: ChromaDBVectorStore, patch_chromadb_constructors, mock_chroma_client):
    chromadb_store = await chromadb_store_fixture; await chromadb_store.setup(config={"collection_name": "delete_all_test", "path":None})
    async def mock_run_in_executor(executor, func_partial): func_partial(); return True
    with patch("asyncio.BaseEventLoop.run_in_executor", side_effect=mock_run_in_executor):
        success = await chromadb_store.delete(delete_all=True)
    assert success is True; mock_chroma_client.delete_collection.assert_called_once_with(name="delete_all_test")
    assert chromadb_store._collection is None
