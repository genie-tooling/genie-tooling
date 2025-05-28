###tests/unit/rag/plugins/impl/vector_stores/test_chromadb_store.py###
"""Unit tests for ChromaDBVectorStore."""
import asyncio
import logging
from pathlib import Path
from typing import Any, AsyncIterable, Dict, Tuple
from unittest.mock import MagicMock, patch

import pytest
from genie_tooling.core.types import Chunk, EmbeddingVector
from genie_tooling.rag.plugins.impl.vector_stores.chromadb_store import (
    ChromaDBVectorStore,
)

# Mock chromadb module and its components
try:
    import chromadb
    # Use actual types for spec if available, otherwise Any
    CHROMA_CLIENT_SPEC = chromadb.Client if chromadb else Any
    CHROMA_COLLECTION_SPEC = chromadb.api.models.Collection.Collection if chromadb else Any
except ImportError:
    chromadb = None
    CHROMA_CLIENT_SPEC = Any
    CHROMA_COLLECTION_SPEC = Any


@pytest.fixture
def mock_chroma_collection() -> MagicMock:
    collection = MagicMock(spec=CHROMA_COLLECTION_SPEC)
    collection.name = "mock_collection_name"
    collection.add = MagicMock(name="collection_add_method")
    # Make query return value more robust for cases where no results are found
    collection.query = MagicMock(name="collection_query_method", return_value={"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]})
    collection.delete = MagicMock(name="collection_delete_method")
    collection.count = MagicMock(name="collection_count_method", return_value=0)
    return collection

@pytest.fixture
def mock_chroma_client(mock_chroma_collection: MagicMock) -> MagicMock:
    client = MagicMock()
    client.get_or_create_collection = MagicMock(name="client_get_or_create_collection_method", return_value=mock_chroma_collection)
    client.delete_collection = MagicMock(name="client_delete_collection_method")
    return client

@pytest.fixture
def patch_chromadb_constructors(mock_chroma_client: MagicMock):
    with patch("chromadb.Client", return_value=mock_chroma_client) as mock_ephemeral_constructor, \
         patch("chromadb.PersistentClient", return_value=mock_chroma_client) as mock_persistent_constructor, \
         patch("chromadb.HttpClient", return_value=mock_chroma_client) as mock_http_constructor:
        yield {
            "ephemeral": mock_ephemeral_constructor,
            "persistent": mock_persistent_constructor,
            "http": mock_http_constructor,
        }


@pytest.fixture
async def chromadb_store_fixture(request) -> ChromaDBVectorStore:
    store = ChromaDBVectorStore()

    async def finalizer_async():
        if hasattr(store, "_client") and store._client is not None:
            await store.teardown()

    def finalizer_sync():
        try:
            loop = asyncio.get_event_loop_policy().get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
             #logger.warning("Async finalizer for chromadb_store_fixture called while event loop is running. Attempting to schedule.")
             asyncio.ensure_future(finalizer_async(), loop=loop)
        else:
            loop.run_until_complete(finalizer_async())

    request.addfinalizer(finalizer_sync)
    return store


class SimpleChunk(Chunk):
    def __init__(self, id: str, content: str, metadata: Dict[str, Any]):
        self.id = id
        self.content = content
        self.metadata = metadata

async def sample_embeddings_stream(count: int = 2, dim: int = 3, start_idx: int = 0) -> AsyncIterable[Tuple[Chunk, EmbeddingVector]]:
    for i in range(count):
        chunk = SimpleChunk(id=f"chunk{start_idx + i}", content=f"Content for chunk {start_idx + i}", metadata={"source": "test", "index": start_idx + i})
        vector = [float(start_idx + i + j * 0.1) for j in range(dim)]
        yield chunk, vector


@pytest.mark.asyncio
async def test_chromadb_setup_ephemeral(chromadb_store_fixture: ChromaDBVectorStore, patch_chromadb_constructors, mock_chroma_client, mock_chroma_collection, caplog):
    chromadb_store = await chromadb_store_fixture
    caplog.set_level(logging.INFO)
    collection_name = "ephemeral_test"
    mock_chroma_collection.name = collection_name

    await chromadb_store.setup(config={"collection_name": collection_name})

    patch_chromadb_constructors["ephemeral"].assert_called_once()
    assert chromadb_store._client is mock_chroma_client
    assert chromadb_store._collection is mock_chroma_collection
    assert chromadb_store._collection.name == collection_name # type: ignore
    assert "Using ephemeral in-memory ChromaDB client" in caplog.text


@pytest.mark.asyncio
async def test_chromadb_setup_persistent(chromadb_store_fixture: ChromaDBVectorStore, patch_chromadb_constructors, mock_chroma_client, mock_chroma_collection, tmp_path: Path, caplog):
    chromadb_store = await chromadb_store_fixture
    caplog.set_level(logging.INFO)
    db_path = tmp_path / "chroma_test_data"
    collection_name = "persistent_test"
    mock_chroma_collection.name = collection_name


    await chromadb_store.setup(config={"collection_name": collection_name, "path": str(db_path)})

    patch_chromadb_constructors["persistent"].assert_called_once()
    assert chromadb_store._client is mock_chroma_client
    assert chromadb_store._collection is mock_chroma_collection
    assert chromadb_store._collection.name == collection_name # type: ignore
    assert f"Using persistent ChromaDB at path '{db_path}'" in caplog.text
    assert db_path.is_dir()

@pytest.mark.asyncio
async def test_chromadb_setup_http(chromadb_store_fixture: ChromaDBVectorStore, patch_chromadb_constructors, mock_chroma_client, mock_chroma_collection, caplog):
    chromadb_store = await chromadb_store_fixture
    caplog.set_level(logging.INFO)
    collection_name = "http_test"
    mock_chroma_collection.name = collection_name

    await chromadb_store.setup(config={"collection_name": collection_name, "host": "localhost", "port": 8000})

    patch_chromadb_constructors["http"].assert_called_once()
    assert chromadb_store._client is mock_chroma_client
    assert chromadb_store._collection is mock_chroma_collection
    assert chromadb_store._collection.name == collection_name # type: ignore
    assert "Connecting to remote ChromaDB at localhost:8000" in caplog.text

@pytest.mark.asyncio
async def test_chromadb_setup_hnsw(chromadb_store_fixture: ChromaDBVectorStore, patch_chromadb_constructors, mock_chroma_client):
    chromadb_store = await chromadb_store_fixture
    await chromadb_store.setup(config={
        "collection_name": "hnsw_test",
        "use_hnsw_indexing": True,
        "hnsw_space": "cosine"
    })

    mock_chroma_client.get_or_create_collection.assert_called_once()
    args, kwargs = mock_chroma_client.get_or_create_collection.call_args
    assert kwargs["name"] == "hnsw_test"
    assert kwargs["metadata"] == {"hnsw:space": "cosine"}
    # ChromaDB client might not include embedding_function in kwargs if it's None by default.
    # So, checking .get() is safer.
    assert kwargs.get("embedding_function", None) is None


@pytest.mark.asyncio
async def test_chromadb_add_successful(chromadb_store_fixture: ChromaDBVectorStore, patch_chromadb_constructors, mock_chroma_collection):
    chromadb_store = await chromadb_store_fixture
    await chromadb_store.setup(config={"collection_name": "add_test_coll"})

    mock_chroma_collection.add.return_value = None # Simulate successful add (no return value needed)

    embeddings_data = [(SimpleChunk(id=f"c{i}", content=f"tc{i}", metadata={"m_k":f"m_v{i}", "m_num":i, "m_bool":bool(i%2)}), [float(i), float(i+1)]) for i in range(3)]
    async def stream_data():
        for item in embeddings_data: yield item

    async def mock_run_in_executor_direct_call(executor, func_partial):
        # Directly call the partial function, which wraps the mock
        func_partial()
        return None # Simulate what run_in_executor might return if func returns None

    with patch("asyncio.BaseEventLoop.run_in_executor", side_effect=mock_run_in_executor_direct_call):
        result = await chromadb_store.add(stream_data(), config={"batch_size": 2})

    assert result["added_count"] == 3
    assert not result["errors"]

    assert mock_chroma_collection.add.call_count == 2

    call1_args_tuple = mock_chroma_collection.add.call_args_list[0]
    call1_kwargs = call1_args_tuple.kwargs
    assert len(call1_kwargs["ids"]) == 2
    assert call1_kwargs["ids"] == ["c0", "c1"]
    assert call1_kwargs["metadatas"] == [{"m_k":"m_v0", "m_num":0, "m_bool":False}, {"m_k":"m_v1", "m_num":1, "m_bool":True}]

    call2_args_tuple = mock_chroma_collection.add.call_args_list[1]
    call2_kwargs = call2_args_tuple.kwargs
    assert len(call2_kwargs["ids"]) == 1
    assert call2_kwargs["ids"] == ["c2"]

@pytest.mark.asyncio
async def test_chromadb_add_empty_embedding_vector(chromadb_store_fixture: ChromaDBVectorStore, patch_chromadb_constructors, mock_chroma_collection, caplog):
    chromadb_store = await chromadb_store_fixture
    caplog.set_level(logging.WARNING)
    await chromadb_store.setup()

    mock_chroma_collection.add.return_value = None

    async def mock_run_in_executor_direct_call_empty(executor, func_partial):
        if func_partial.keywords.get("ids"): # Only call if there are IDs to add
            func_partial()
        return None

    with patch("asyncio.BaseEventLoop.run_in_executor", side_effect=mock_run_in_executor_direct_call_empty):
        async def stream_with_empty_vec():
            yield SimpleChunk(id="c1", content="good", metadata={}), [0.1, 0.2]
            yield SimpleChunk(id="c2_bad_empty", content="bad_empty", metadata={}), []
            yield SimpleChunk(id="c3", content="good2", metadata={}), [0.3, 0.4]

        result = await chromadb_store.add(stream_with_empty_vec())

    assert result["added_count"] == 2
    assert len(result["errors"]) == 1
    assert "Skipping chunk ID 'c2_bad_empty' due to empty embedding vector." in result["errors"][0]
    assert "Skipping chunk ID 'c2_bad_empty' due to empty embedding vector." in caplog.text

    assert mock_chroma_collection.add.call_count == 1
    called_kwargs = mock_chroma_collection.add.call_args.kwargs
    assert called_kwargs["ids"] == ["c1", "c3"]

@pytest.mark.asyncio
async def test_chromadb_add_fails_if_collection_not_init(chromadb_store_fixture: ChromaDBVectorStore):
    chromadb_store = await chromadb_store_fixture
    result = await chromadb_store.add(sample_embeddings_stream(1))
    assert result["added_count"] == 0
    assert "ChromaDB collection not initialized." in result["errors"][0]

@pytest.mark.asyncio
async def test_chromadb_search_successful(chromadb_store_fixture: ChromaDBVectorStore, patch_chromadb_constructors, mock_chroma_collection):
    chromadb_store = await chromadb_store_fixture
    await chromadb_store.setup(config={"collection_name": "search_coll", "hnsw_space": "cosine"})

    mock_query_results = {
        "ids": [["id1", "id2"]],
        "documents": [["doc content 1", "doc content 2"]],
        "metadatas": [[{"src": "s1", "id": "id1"}, {"src": "s2", "id":"id2"}]],
        "distances": [[0.1, 0.2]]
    }
    mock_chroma_collection.query.return_value = mock_query_results
    mock_chroma_collection.count.return_value = 2

    query_vec = [0.5, 0.5, 0.5]
    results = await chromadb_store.search(query_vec, top_k=2, filter_metadata={"src": "s1"})

    mock_chroma_collection.query.assert_called_once()
    call_args_tuple = mock_chroma_collection.query.call_args
    call_kwargs = call_args_tuple.kwargs

    assert call_kwargs["query_embeddings"] == [query_vec]
    assert call_kwargs["n_results"] == 2
    assert call_kwargs["where"] == {"src": "s1"}
    assert call_kwargs["include"] == ["metadatas", "documents", "distances"]

    assert len(results) == 2
    assert results[0].id == "id1"
    assert pytest.approx(results[0].score) == 1.0 - 0.1

@pytest.mark.asyncio
async def test_chromadb_search_empty_store_or_no_results(chromadb_store_fixture: ChromaDBVectorStore, patch_chromadb_constructors, mock_chroma_collection):
    chromadb_store = await chromadb_store_fixture
    await chromadb_store.setup()

    mock_chroma_collection.count.return_value = 0
    # Query method's mock return_value should reflect an empty result set
    mock_chroma_collection.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}


    results = await chromadb_store.search([0.1]*3, top_k=5)
    assert results == []
    # If count() is 0, _sync_search in ChromaDBVectorStore returns an empty-like dict *before* calling self._collection.query
    mock_chroma_collection.query.assert_not_called()


@pytest.mark.asyncio
async def test_chromadb_delete_by_ids(chromadb_store_fixture: ChromaDBVectorStore, patch_chromadb_constructors, mock_chroma_collection):
    chromadb_store = await chromadb_store_fixture
    await chromadb_store.setup()

    ids_to_delete = ["id1", "id2"]
    async def mock_run_in_executor_for_delete(executor, func_partial):
        func_partial()
        return True

    with patch("asyncio.BaseEventLoop.run_in_executor", side_effect=mock_run_in_executor_for_delete):
        success = await chromadb_store.delete(ids=ids_to_delete)

    assert success is True
    mock_chroma_collection.delete.assert_called_once_with(ids=ids_to_delete)

@pytest.mark.asyncio
async def test_chromadb_delete_all(chromadb_store_fixture: ChromaDBVectorStore, patch_chromadb_constructors, mock_chroma_client):
    chromadb_store = await chromadb_store_fixture
    await chromadb_store.setup(config={"collection_name": "delete_all_test"})

    async def mock_run_in_executor_for_delete_coll(executor, func_partial):
        func_partial()
        return True

    with patch("asyncio.BaseEventLoop.run_in_executor", side_effect=mock_run_in_executor_for_delete_coll):
        success = await chromadb_store.delete(delete_all=True)

    assert success is True
    mock_chroma_client.delete_collection.assert_called_once_with(name="delete_all_test")
    assert chromadb_store._collection is None # type: ignore

@pytest.mark.asyncio
async def test_chromadb_delete_filter_metadata(chromadb_store_fixture: ChromaDBVectorStore, patch_chromadb_constructors, mock_chroma_collection):
    chromadb_store = await chromadb_store_fixture
    await chromadb_store.setup()

    filter_meta = {"source": "test"}
    async def mock_run_in_executor_for_delete_where(executor, func_partial):
        func_partial()
        return True

    with patch("asyncio.BaseEventLoop.run_in_executor", side_effect=mock_run_in_executor_for_delete_where):
        success = await chromadb_store.delete(filter_metadata=filter_meta)

    assert success is True
    mock_chroma_collection.delete.assert_called_once_with(where=filter_meta)

@pytest.mark.asyncio
async def test_chromadb_library_not_installed(caplog):
    caplog.set_level(logging.ERROR)
    with patch("genie_tooling.rag.plugins.impl.vector_stores.chromadb_store.chromadb", None), \
         patch("genie_tooling.rag.plugins.impl.vector_stores.chromadb_store.ChromaSettings", None):

        store_no_libs = ChromaDBVectorStore()
        await store_no_libs.setup()

        assert store_no_libs._client is None # type: ignore
        assert store_no_libs._collection is None # type: ignore
        assert "ChromaDBVectorStore Error: 'chromadb-client' library not installed." in caplog.text

        result = await store_no_libs.add(sample_embeddings_stream(0))
        assert "ChromaDB collection not initialized." in result["errors"][0]

        search_results = await store_no_libs.search([0.1], 1)
        assert search_results == []

        delete_result = await store_no_libs.delete(ids=["test"])
        assert delete_result is False

@pytest.mark.asyncio
async def test_chromadb_search_score_calculation_ip_space(chromadb_store_fixture: ChromaDBVectorStore, patch_chromadb_constructors, mock_chroma_collection):
    chromadb_store = await chromadb_store_fixture
    await chromadb_store.setup(config={"collection_name": "ip_search_coll", "hnsw_space": "ip"})
    mock_chroma_collection.count.return_value = 1
    mock_query_results = {
        "ids": [["id1"]], "documents": [["doc1"]], "metadatas": [[{"id":"id1"}]] , "distances": [[-0.8]]
    }
    mock_chroma_collection.query.return_value = mock_query_results

    results = await chromadb_store.search([0.1]*3, top_k=1)
    assert len(results) == 1
    assert pytest.approx(results[0].score) == 0.8

@pytest.mark.asyncio
async def test_chromadb_search_score_calculation_l2_space(chromadb_store_fixture: ChromaDBVectorStore, patch_chromadb_constructors, mock_chroma_collection):
    chromadb_store = await chromadb_store_fixture
    await chromadb_store.setup(config={"collection_name": "l2_search_coll", "hnsw_space": "l2"})
    mock_chroma_collection.count.return_value = 1
    mock_query_results = {
        "ids": [["id1"]], "documents": [["doc1"]], "metadatas": [[{"id":"id1"}]] , "distances": [[0.5]]
    }
    mock_chroma_collection.query.return_value = mock_query_results

    results = await chromadb_store.search([0.1]*3, top_k=1)
    assert len(results) == 1
    assert pytest.approx(results[0].score) == 1.0 / (1.0 + 0.5)

@pytest.mark.asyncio
async def test_chromadb_add_metadata_sanitization(chromadb_store_fixture: ChromaDBVectorStore, patch_chromadb_constructors, mock_chroma_collection):
    chromadb_store = await chromadb_store_fixture
    await chromadb_store.setup()

    complex_metadata = {"normal_str": "abc", "num": 123, "floaty": 1.23, "truthy": True, "list_val": [1,2], "dict_val": {"a":1}}
    mock_chroma_collection.add = MagicMock(name="collection_add_method_for_test_add_meta_sanitization")

    async def mock_run_in_executor_direct_call_meta(executor, func_partial):
        if func_partial.func == mock_chroma_collection.add: # Ensure we are mocking the correct partial
            func_partial()
        else: # Should not happen in this specific test's add path
            return await asyncio.get_running_loop().run_in_executor(executor, func_partial)
        return None

    with patch("asyncio.BaseEventLoop.run_in_executor", side_effect=mock_run_in_executor_direct_call_meta):
        async def stream_complex_meta():
            yield SimpleChunk("c_complex", "content", complex_metadata), [0.1,0.2]
        await chromadb_store.add(stream_complex_meta())

    mock_chroma_collection.add.assert_called_once()
    called_args_tuple = mock_chroma_collection.add.call_args
    assert called_args_tuple is not None
    called_kwargs = called_args_tuple.kwargs

    called_metadata = called_kwargs["metadatas"][0]
    assert called_metadata["normal_str"] == "abc"
    assert called_metadata["num"] == 123
    assert called_metadata["floaty"] == 1.23
    assert called_metadata["truthy"] is True
    assert called_metadata["list_val"] == "[1, 2]"
    assert called_metadata["dict_val"] == "{'a': 1}"
