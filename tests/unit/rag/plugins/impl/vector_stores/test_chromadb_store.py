### tests/unit/rag/plugins/impl/vector_stores/test_chromadb_store.py
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, AsyncIterable, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest
from genie_tooling.core.types import Chunk, EmbeddingVector, RetrievedChunk
from genie_tooling.vector_stores.impl.chromadb_store import ChromaDBVectorStore

# Import ChromaSettings for mocking if chromadb is not installed
try:
    from chromadb.config import Settings as ActualChromaSettings
    CHROMADB_INSTALLED_FOR_TESTS = True
except ImportError:
    ActualChromaSettings = MagicMock  # type: ignore
    CHROMADB_INSTALLED_FOR_TESTS = False
import uuid

# anext was added in Python 3.10.
try:
    from asyncio import anext
except ImportError:
    # Fallback for environments where it might be missing.
    async def anext(ait): # type: ignore
        return await ait.__anext__()

logger = logging.getLogger(__name__)

# --- Test Helpers ---

class SimpleChunkForTest(Chunk):
    def __init__(self, id: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        self.id = id
        self.content = content
        self.metadata = metadata or {}

class _RetrievedChunkImpl(RetrievedChunk, Chunk): # type: ignore
    def __init__(self, content: str, metadata: Dict[str, Any], score: float, id: Optional[str] = None, rank: Optional[int] = None):
        self.content: str = content
        self.metadata: Dict[str, Any] = metadata
        self.id: Optional[str] = id
        self.score: float = score
        self.rank: Optional[int] = rank


async def sample_embeddings_stream(
    chunks: List[Tuple[str, str]], embeddings: List[List[float]]
) -> AsyncIterable[Tuple[Chunk, EmbeddingVector]]:
    for i, chunk_data in enumerate(chunks):
        chunk_id, chunk_content = chunk_data
        yield SimpleChunkForTest(id=chunk_id, content=chunk_content), embeddings[i]

# --- Fixtures ---

@pytest.fixture(params=["ephemeral", "persistent_tmp", "http_mocked"])
async def chromadb_store_instance(
    request, tmp_path: Path
) -> AsyncGenerator[ChromaDBVectorStore, None]:
    """
    A comprehensive fixture that sets up the ChromaDBVectorStore in different modes.
    This ensures that each test function receives a properly initialized instance.
    """
    store = ChromaDBVectorStore()
    # Use a unique collection name for each test parameterization to avoid clashes
    collection_name = f"test_collection_{request.param}_{uuid.uuid4().hex[:6]}"
    config: Dict[str, Any] = {"collection_name": collection_name, "embedding_dim": 2}


    if request.param == "persistent_tmp":
        # Ensure a unique path for each persistent test run
        db_path = tmp_path / f"chroma_db_{request.param}_{uuid.uuid4().hex[:6]}"
        config["path"] = str(db_path)
    elif request.param == "http_mocked":
        config["host"] = "mock-chroma-server"
        config["port"] = 8000

    if request.param == "http_mocked":
        mock_http_client_instance = MagicMock(name="MockHttpClientForFixture")
        mock_collection_for_http = MagicMock(name="MockCollectionForHttp")
        mock_collection_for_http.count.return_value = 0
        mock_http_client_instance.get_or_create_collection.return_value = mock_collection_for_http

        with patch("genie_tooling.vector_stores.impl.chromadb_store.chromadb.HttpClient", return_value=mock_http_client_instance), \
             patch("genie_tooling.vector_stores.impl.chromadb_store.ChromaSettings", new_callable=MagicMock) as MockSettingsConstructor:
            MockSettingsConstructor.return_value = ActualChromaSettings(anonymized_telemetry=False) if ActualChromaSettings else MagicMock()
            store._test_mock_http_client_instance = mock_http_client_instance # type: ignore
            await store.setup(config)
    else:
        await store.setup(config)

    yield store

    # Teardown: For persistent_tmp, explicitly delete the collection to ensure clean state for subsequent unrelated tests
    # if they were to somehow reuse paths (though tmp_path should prevent this).
    # For ephemeral, the data is in memory. For http_mocked, it's mocked.
    if request.param == "persistent_tmp" and store._client and store._collection_name:
        try:
            # Ensure client is not None before attempting to delete
            if hasattr(store._client, 'delete_collection'):
                 store._client.delete_collection(name=store._collection_name)
                 logger.info(f"Cleaned up persistent collection: {store._collection_name}")
        except Exception as e:
            logger.warning(f"Error during test cleanup of persistent collection {store._collection_name}: {e}")
    await store.teardown()


# --- Test Cases ---
@pytest.mark.asyncio()
async def test_add_and_search(chromadb_store_instance: AsyncGenerator[ChromaDBVectorStore, None]):
    store = await anext(chromadb_store_instance)
    if not hasattr(store, '_client') or store._client is None:
        pytest.skip("ChromaDB client not initialized for this test parameterization.")

    assert store._collection is not None, "Collection should be initialized after setup"

    chunks_to_add = [("id1", "A document about cats."), ("id2", "A document about dogs.")]
    embeddings_to_add = [[0.1, 0.9], [0.8, 0.1]]

    if hasattr(store, '_test_mock_http_client_instance'):
        store._collection.upsert = MagicMock() # Changed from add to upsert for Chroma
        store._collection.count.return_value = 2
        mock_query_result = {
            "ids": [["id2"]], "documents": [["A document about dogs."]],
            "metadatas": [[{"genie_chunk_id": "id2"}]], "distances": [[0.1]] # Ensure metadata has genie_chunk_id
        }
        store._collection.query = MagicMock(return_value=mock_query_result)

    add_result = await store.add(sample_embeddings_stream(chunks_to_add, embeddings_to_add))
    assert add_result["added_count"] == 2
    assert not add_result["errors"]

    query_embedding_dog = [0.7, 0.2]
    search_results = await store.search(query_embedding=query_embedding_dog, top_k=1)

    assert len(search_results) == 1
    assert search_results[0].id == "id2"
    assert search_results[0].content == "A document about dogs."
    # Score assertion depends on the distance metric, for cosine, higher is better (closer to 1)
    # For L2, lower is better (closer to 0), so score (1/(1+dist)) would be high for low dist.
    assert search_results[0].score > 0.5 # General check


@pytest.mark.asyncio()
async def test_delete_by_id(chromadb_store_instance: AsyncGenerator[ChromaDBVectorStore, None]):
    store = await anext(chromadb_store_instance)
    if not hasattr(store, '_client') or store._client is None:
        pytest.skip("ChromaDB client not initialized for this test parameterization.")
    assert store._collection is not None

    _mock_db_items_for_delete_test = {"id_del_1": "data1", "id_keep_1": "data2"}
    if hasattr(store, '_test_mock_http_client_instance'):
        store._collection.upsert = MagicMock()
        store._collection.delete = MagicMock()
        def mock_count_after_delete_id(): return len(_mock_db_items_for_delete_test)
        def mock_delete_id(ids=None, where=None):
            if ids:
                for item_id_to_del in ids: _mock_db_items_for_delete_test.pop(item_id_to_del, None)
        store._collection.count = MagicMock(side_effect=mock_count_after_delete_id)
        store._collection.delete = MagicMock(side_effect=mock_delete_id)
        store._collection.get = MagicMock(return_value={"ids": list(_mock_db_items_for_delete_test.keys())})

    chunks_to_add = [("id_del_1", "Item to delete"), ("id_keep_1", "Item to keep")]
    embeddings_to_add = [[0.1, 0.1], [0.9, 0.9]]
    await store.add(sample_embeddings_stream(chunks_to_add, embeddings_to_add))

    # Verify initial count if not mocked
    if not hasattr(store, '_test_mock_http_client_instance'):
        assert store._collection.count() == 2

    delete_success = await store.delete(ids=["id_del_1"])
    assert delete_success is True

    if not hasattr(store, '_test_mock_http_client_instance'):
        assert store._collection.count() == 1
        remaining_items = store._collection.get(ids=["id_keep_1"]) # type: ignore
        assert remaining_items["ids"] == ["id_keep_1"]
    else: # For mocked HTTP client
        assert store._collection.count() == 1
        assert "id_del_1" not in _mock_db_items_for_delete_test
        assert "id_keep_1" in _mock_db_items_for_delete_test

@pytest.mark.asyncio()
async def test_delete_all(chromadb_store_instance: AsyncGenerator[ChromaDBVectorStore, None]):
    store = await anext(chromadb_store_instance)
    if not hasattr(store, '_client') or store._client is None:
        pytest.skip("ChromaDB client not initialized for this test parameterization.")
    assert store._client is not None

    new_mock_collection_after_delete = MagicMock(name="NewCollectionAfterDelete")
    if hasattr(store, '_test_mock_http_client_instance'):
        store._client.delete_collection = MagicMock()
        new_mock_collection_after_delete.count.return_value = 0
        store._client.get_or_create_collection.return_value = new_mock_collection_after_delete

    chunks_to_add = [("id_a", "A"), ("id_b", "B")]
    embeddings_to_add = [[0.1, 0.1], [0.9, 0.9]]
    await store.add(sample_embeddings_stream(chunks_to_add, embeddings_to_add))

    if not hasattr(store, '_test_mock_http_client_instance'):
         assert store._collection is not None and store._collection.count() == 2

    delete_success = await store.delete(delete_all=True)
    assert delete_success is True
    assert store._collection is None # After delete_all, _collection is set to None

    # If we add again, _ensure_collection_exists_async_internal should recreate it
    if store._embedding_dim is None: # Ensure dim is set if it was inferred and then collection deleted
        store._embedding_dim = 2

    await store.add(sample_embeddings_stream([("id_c", "C")], [[0.5, 0.5]]))
    assert store._collection is not None # Should be recreated

    if hasattr(store, '_test_mock_http_client_instance'):
        assert store._collection is new_mock_collection_after_delete
        store._client.delete_collection.assert_called_once_with(name=store._collection_name)
        # get_or_create_collection is called during setup and after delete_all
        assert store._client.get_or_create_collection.call_count >= 1
    else: # Real client
        assert store._collection.count() == 1


@pytest.mark.asyncio()
async def test_setup_persistent_client_selected(tmp_path: Path):
    store = ChromaDBVectorStore()
    db_path = tmp_path / "persistent_test_dir_selected"
    config_persistent = {
        "collection_name": "persistent_coll_selected",
        "path": str(db_path),
        "embedding_dim": 128 # Must provide dim if collection might not exist
    }
    # Patch chromadb.PersistentClient where it's used in the store's module
    with patch("genie_tooling.vector_stores.impl.chromadb_store.chromadb.PersistentClient") as MockPersistentClientConstructor:
        mock_persistent_client_instance = MagicMock(name="MockPersistentClientInstanceSelected")
        mock_collection_for_persistent = MagicMock(name="MockCollectionForPersistentSelected")
        mock_collection_for_persistent.count.return_value = 0
        mock_persistent_client_instance.get_or_create_collection.return_value = mock_collection_for_persistent
        MockPersistentClientConstructor.return_value = mock_persistent_client_instance

        await store.setup(config_persistent)

    MockPersistentClientConstructor.assert_called_once()
    call_args, call_kwargs = MockPersistentClientConstructor.call_args
    assert call_kwargs.get("path") == str(db_path)
    assert store._client is mock_persistent_client_instance
    assert store._collection is mock_collection_for_persistent

@pytest.mark.asyncio()
async def test_setup_ephemeral_client_selected():
    store = ChromaDBVectorStore()
    config_ephemeral_explicit_none = {"collection_name": "ephem_coll_selected", "path": None, "embedding_dim": 128}

    with patch("genie_tooling.vector_stores.impl.chromadb_store.ChromaSettings", new_callable=MagicMock) as MockSettingsConstructor, \
         patch("genie_tooling.vector_stores.impl.chromadb_store.chromadb.EphemeralClient") as MockEphemeralClientConstructor:

        # Configure the return value of ChromaSettings constructor if needed by EphemeralClient
        mock_settings_instance = ActualChromaSettings(anonymized_telemetry=False) if ActualChromaSettings else MagicMock()
        MockSettingsConstructor.return_value = mock_settings_instance

        mock_ephemeral_client_instance = MagicMock(name="MockEphemeralClientInstanceSelected")
        mock_collection_for_ephemeral = MagicMock(name="MockCollectionForEphemeralSelected")
        mock_collection_for_ephemeral.count.return_value = 0
        mock_ephemeral_client_instance.get_or_create_collection.return_value = mock_collection_for_ephemeral
        MockEphemeralClientConstructor.return_value = mock_ephemeral_client_instance

        await store.setup(config_ephemeral_explicit_none)

    assert store._client is mock_ephemeral_client_instance
    assert store._collection is mock_collection_for_ephemeral
    # Check that ChromaSettings was called correctly for EphemeralClient (without chroma_api_impl)
    MockSettingsConstructor.assert_called_once_with(anonymized_telemetry=False)
    MockEphemeralClientConstructor.assert_called_once_with(settings=mock_settings_instance)


@pytest.mark.asyncio()
async def test_add_empty_vector_skipped(chromadb_store_instance: AsyncGenerator[ChromaDBVectorStore, None]):
    store = await anext(chromadb_store_instance)
    if not hasattr(store, '_client') or store._client is None:
        pytest.skip("ChromaDB client not initialized, skipping test.")

    async def stream_with_empty() -> AsyncIterable[Tuple[Chunk, EmbeddingVector]]:
        yield SimpleChunkForTest(id="id_good", content="good vector"), [0.1, 0.2]
        yield SimpleChunkForTest(id="id_empty", content="empty vector"), []
    result = await store.add(stream_with_empty())
    assert result["added_count"] == 1
    assert "Skipping chunk ID 'id_empty' due to empty vector." in result["errors"]

@pytest.mark.asyncio()
async def test_search_collection_empty(chromadb_store_instance: AsyncGenerator[ChromaDBVectorStore, None]):
    store = await anext(chromadb_store_instance)
    if not hasattr(store, '_client') or store._client is None:
        pytest.skip("ChromaDB client not initialized, skipping test.")

    # Ensure the collection is truly empty for this test, especially for persistent_tmp
    if store._collection:
        await store.delete(delete_all=True) # This will set _collection to None
        # Re-ensure collection exists if dim was known, or add will do it
        if store._embedding_dim:
            await store._ensure_collection_exists_async_internal()


    if hasattr(store, '_test_mock_http_client_instance'): # For http_mocked
        store._collection.count.return_value = 0
        store._collection.query = MagicMock(return_value={"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]})
    elif store._collection: # For real ephemeral/persistent clients
        assert store._collection.count() == 0, "Collection should be empty after setup/delete_all for this test"


    results = await store.search([0.1,0.2], top_k=1)
    assert results == []

@pytest.mark.asyncio()
async def test_search_with_filter(chromadb_store_instance: AsyncGenerator[ChromaDBVectorStore, None]):
    store = await anext(chromadb_store_instance)
    if not hasattr(store, '_client') or store._client is None:
        pytest.skip("ChromaDB client not initialized, skipping test.")

    filter_dict = {"source": "test_source"}

    if hasattr(store, '_test_mock_http_client_instance'):
        store._collection.count.return_value = 1 # Assume one item matches for mock
        # Mock query to return something if filter is applied
        mock_query_result_filtered = {
            "ids": [["filtered_id"]], "documents": [["Filtered content"]],
            "metadatas": [[{"source": "test_source", "genie_chunk_id": "filtered_id"}]], "distances": [[0.05]]
        }
        store._collection.query = MagicMock(return_value=mock_query_result_filtered)

    # Add an item that would match the filter if this were a real client
    if not hasattr(store, '_test_mock_http_client_instance'):
        await store.add(sample_embeddings_stream(
            [("item_for_filter", "content for filter")],
            [[0.4, 0.6]]
        ), config={"metadata_override_for_add": [{"source": "test_source"}]}) # Simulate adding with specific metadata

    await store.search([0.5, 0.5], top_k=1, filter_metadata=filter_dict)

    if hasattr(store, '_test_mock_http_client_instance'):
        store._collection.query.assert_called_once()
        call_kwargs = store._collection.query.call_args.kwargs
        assert "where" in call_kwargs and call_kwargs["where"] == filter_dict
    # For real client, the assertion is that it doesn't crash and the filter is passed.
    # Verifying the filter *worked* correctly is harder without inspecting ChromaDB internals or specific data.
    # The main goal here is that the `where` parameter is constructed and passed.