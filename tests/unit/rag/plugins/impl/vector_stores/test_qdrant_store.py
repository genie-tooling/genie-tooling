### tests/unit/rag/plugins/impl/vector_stores/test_qdrant_store.py
import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, AsyncIterable, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Attempt to import real Qdrant types for spec, but allow mocks if not installed
try:
    from qdrant_client import AsyncQdrantClient as ActualAsyncQdrantClient
    from qdrant_client.http import models as rest
    from qdrant_client.http.models import (
        Distance as ActualQdrantDistance,
    )
    from qdrant_client.http.models import (
        FieldCondition as ActualFieldCondition,
    )
    from qdrant_client.http.models import (
        Filter as ActualFilter,
    )
    from qdrant_client.http.models import (
        MatchValue as ActualMatchValue,
    )
    from qdrant_client.http.models import (
        PointStruct as ActualPointStruct,
    )
    from qdrant_client.http.models import (
        ScoredPoint as ActualScoredPoint,
    )
    from qdrant_client.http.models import (
        VectorParams as ActualVectorParams,
    )

    QDRANT_CLIENT_INSTALLED_FOR_TESTS = True
except ImportError:
    QDRANT_CLIENT_INSTALLED_FOR_TESTS = False
    ActualAsyncQdrantClient = AsyncMock # type: ignore
    rest = MagicMock()
    ActualQdrantDistance = MagicMock()
    ActualQdrantDistance.COSINE = "COSINE"
    ActualQdrantDistance.EUCLID = "EUCLIDEAN"
    ActualQdrantDistance.DOT = "DOT"
    ActualFilter = MagicMock
    ActualFieldCondition = MagicMock
    ActualMatchValue = MagicMock
    ActualPointStruct = MagicMock
    ActualVectorParams = MagicMock
    ActualScoredPoint = MagicMock


from genie_tooling.core.types import Chunk, EmbeddingVector
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.vector_stores.impl.qdrant_store import (
    QdrantVectorStorePlugin,
)

QDRANT_STORE_LOGGER_NAME = "genie_tooling.vector_stores.impl.qdrant_store"


class MockKeyProviderForQdrant(KeyProvider):
    _keys: Dict[str, Optional[str]]
    def __init__(self, api_key_value: Optional[str] = "test_qdrant_api_key"):
        self._keys = {"QDRANT_API_KEY_TEST": api_key_value}
    async def get_key(self, key_name: str) -> Optional[str]: return self._keys.get(key_name)
    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None: pass
    async def teardown(self) -> None: pass

class SimpleChunkForQdrant(Chunk):
    def __init__(self, id: Optional[str], content: str, metadata: Optional[Dict[str, Any]] = None):
        self.id = id
        self.content = content
        self.metadata = metadata or {}

async def sample_embeddings_stream_qdrant(
    count: int = 2, dim: int = 3, start_id_num: int = 0
) -> AsyncIterable[Tuple[Chunk, EmbeddingVector]]:
    for i in range(count):
        chunk_id = f"q_chunk_{start_id_num + i}"
        content = f"Content for {chunk_id}"
        metadata = {"source_doc_id": f"doc_{start_id_num + i}", "index": i}
        vector = [float(start_id_num + i + j * 0.1) for j in range(dim)]
        yield SimpleChunkForQdrant(id=chunk_id, content=content, metadata=metadata), vector

# Mock exception that qdrant_client might raise for "collection not found"
class MockCollectionNotFoundError(Exception):
    def __init__(self, message="Collection not found test mock"):
        super().__init__(message)
        self.status_code = 404 # Crucial for the logic in _ensure_collection_exists_async

@pytest.fixture()
async def qdrant_store(
    request: pytest.FixtureRequest
) -> AsyncGenerator[QdrantVectorStorePlugin, None]:
    store_instance = QdrantVectorStorePlugin()

    mock_client = AsyncMock(spec=ActualAsyncQdrantClient)
    mock_client.get_collection = AsyncMock()
    mock_client.create_collection = AsyncMock()
    mock_client.upsert = AsyncMock()
    mock_client.search = AsyncMock()
    mock_client.delete = AsyncMock()
    mock_client.delete_collection = AsyncMock()
    mock_client.close = AsyncMock()

    store_instance._test_mock_qdrant_client = mock_client # type: ignore

    with patch("genie_tooling.vector_stores.impl.qdrant_store.AsyncQdrantClient", return_value=mock_client) as constructor_patch, \
         patch("genie_tooling.vector_stores.impl.qdrant_store.QDRANT_CLIENT_AVAILABLE", True), \
         patch("genie_tooling.vector_stores.impl.qdrant_store.QdrantDistance", ActualQdrantDistance), \
         patch("genie_tooling.vector_stores.impl.qdrant_store.VectorParams", ActualVectorParams), \
         patch("genie_tooling.vector_stores.impl.qdrant_store.PointStruct", ActualPointStruct), \
         patch("genie_tooling.vector_stores.impl.qdrant_store.Filter", ActualFilter), \
         patch("genie_tooling.vector_stores.impl.qdrant_store.FieldCondition", ActualFieldCondition), \
         patch("genie_tooling.vector_stores.impl.qdrant_store.MatchValue", ActualMatchValue), \
         patch("genie_tooling.vector_stores.impl.qdrant_store.rest", rest if QDRANT_CLIENT_INSTALLED_FOR_TESTS else MagicMock()):

        store_instance._test_mock_qdrant_constructor = constructor_patch # type: ignore

        yield store_instance

        if hasattr(store_instance, "_client") and store_instance._client is not None:
            await store_instance.teardown()
        elif hasattr(store_instance, "teardown"):
            await store_instance.teardown()


async def acollect(async_iterable: AsyncIterable[Any]) -> List[Any]:
    return [item async for item in async_iterable]

@pytest.mark.asyncio()
class TestQdrantStoreSetup:
    async def test_setup_qdrant_library_not_available(self, caplog: pytest.LogCaptureFixture):
        caplog.set_level(logging.ERROR, logger=QDRANT_STORE_LOGGER_NAME)
        with patch("genie_tooling.vector_stores.impl.qdrant_store.QDRANT_CLIENT_AVAILABLE", False):
            store_no_lib = QdrantVectorStorePlugin()
            await store_no_lib.setup()
            assert store_no_lib._client is None
            assert "qdrant-client library not available" in caplog.text

    async def test_setup_in_memory_client(self, qdrant_store: AsyncGenerator[QdrantVectorStorePlugin, None]):
        store = await anext(qdrant_store)
        mock_constructor_patch = store._test_mock_qdrant_constructor # type: ignore
        await store.setup(config={})
        mock_constructor_patch.assert_called_once_with(
            prefer_grpc=False, timeout=10.0
        )
        assert store._client is mock_constructor_patch.return_value

    async def test_setup_local_persistent_client(self, qdrant_store: AsyncGenerator[QdrantVectorStorePlugin, None], tmp_path: Path):
        store = await anext(qdrant_store)
        mock_constructor_patch = store._test_mock_qdrant_constructor # type: ignore
        db_path = tmp_path / "qdrant_test_data"
        await store.setup(config={"path": str(db_path)})
        mock_constructor_patch.assert_called_with(
            prefer_grpc=False, timeout=10.0, path=str(db_path)
        )

    async def test_setup_remote_client_url(self, qdrant_store: AsyncGenerator[QdrantVectorStorePlugin, None]):
        store = await anext(qdrant_store)
        mock_constructor_patch = store._test_mock_qdrant_constructor # type: ignore
        test_url = "http://remote-qdrant:6333"
        await store.setup(config={"url": test_url})
        mock_constructor_patch.assert_called_with(
            prefer_grpc=False, timeout=10.0, url=test_url
        )

    async def test_setup_remote_client_host_port(self, qdrant_store: AsyncGenerator[QdrantVectorStorePlugin, None]):
        store = await anext(qdrant_store)
        mock_constructor_patch = store._test_mock_qdrant_constructor # type: ignore
        await store.setup(config={"host": "myqdrant", "port": 1234})
        mock_constructor_patch.assert_called_with(
            prefer_grpc=False, timeout=10.0, host="myqdrant", port=1234
        )

    async def test_setup_with_api_key(self, qdrant_store: AsyncGenerator[QdrantVectorStorePlugin, None]):
        store = await anext(qdrant_store)
        mock_constructor_patch = store._test_mock_qdrant_constructor # type: ignore
        mock_kp = MockKeyProviderForQdrant(api_key_value="secret_qdrant_key")
        await store.setup(config={"api_key_name": "QDRANT_API_KEY_TEST", "key_provider": mock_kp})
        mock_constructor_patch.assert_called_with(
            prefer_grpc=False, timeout=10.0, api_key="secret_qdrant_key"
        )

    async def test_setup_ensure_collection_exists(self, qdrant_store: AsyncGenerator[QdrantVectorStorePlugin, None]):
        store = await anext(qdrant_store)
        mock_client_instance = store._test_mock_qdrant_client # type: ignore
        mock_client_instance.get_collection.side_effect = MockCollectionNotFoundError() # type: ignore
        await store.setup(config={"embedding_dim": 128, "collection_name": "test_coll"})

        mock_client_instance.get_collection.assert_awaited_once_with(collection_name="test_coll") # type: ignore
        mock_client_instance.create_collection.assert_awaited_once_with( # type: ignore
            collection_name="test_coll",
            vectors_config=ActualVectorParams(size=128, distance=ActualQdrantDistance.COSINE)
        )

    async def test_setup_collection_already_exists(self, qdrant_store: AsyncGenerator[QdrantVectorStorePlugin, None]):
        store = await anext(qdrant_store)
        mock_client_instance = store._test_mock_qdrant_client # type: ignore
        mock_client_instance.get_collection.return_value = MagicMock() # type: ignore
        await store.setup(config={"embedding_dim": 768, "collection_name": "existing_coll"})
        mock_client_instance.get_collection.assert_awaited_once_with(collection_name="existing_coll") # type: ignore
        mock_client_instance.create_collection.assert_not_called() # type: ignore

    async def test_setup_client_init_fails(self, qdrant_store: AsyncGenerator[QdrantVectorStorePlugin, None], caplog: pytest.LogCaptureFixture):
        store = await anext(qdrant_store)
        mock_constructor_patch = store._test_mock_qdrant_constructor # type: ignore
        caplog.set_level(logging.ERROR, logger=QDRANT_STORE_LOGGER_NAME)
        mock_constructor_patch.side_effect = RuntimeError("Qdrant connection refused")
        await store.setup(config={})
        assert store._client is None
        assert "Failed to initialize Qdrant client: Qdrant connection refused" in caplog.text


@pytest.mark.asyncio()
class TestQdrantStoreAdd:
    async def test_add_successful_single_batch(self, qdrant_store: AsyncGenerator[QdrantVectorStorePlugin, None]):
        store = await anext(qdrant_store)
        mock_client_instance = store._test_mock_qdrant_client # type: ignore
        await store.setup(config={"embedding_dim": 3})

        result = await store.add(sample_embeddings_stream_qdrant(count=2, dim=3))

        assert result["added_count"] == 2
        assert not result["errors"]
        mock_client_instance.upsert.assert_awaited_once() # type: ignore
        call_args = mock_client_instance.upsert.call_args.kwargs # type: ignore
        assert call_args["collection_name"] == store._collection_name
        assert len(call_args["points"]) == 2
        assert call_args["points"][0].id == "q_chunk_0"

    async def test_add_infers_dim_and_creates_collection(self, qdrant_store: AsyncGenerator[QdrantVectorStorePlugin, None]):
        store = await anext(qdrant_store)
        mock_client_instance = store._test_mock_qdrant_client # type: ignore
        mock_client_instance.get_collection.side_effect = MockCollectionNotFoundError() # type: ignore
        await store.setup(config={"collection_name": "infer_dim_coll"})
        assert store._embedding_dim is None

        await store.add(sample_embeddings_stream_qdrant(count=1, dim=5))

        assert store._embedding_dim == 5
        mock_client_instance.create_collection.assert_awaited_once_with( # type: ignore
            collection_name="infer_dim_coll",
            vectors_config=ActualVectorParams(size=5, distance=ActualQdrantDistance.COSINE)
        )
        mock_client_instance.upsert.assert_awaited_once() # type: ignore

    async def test_add_dimension_mismatch(self, qdrant_store: AsyncGenerator[QdrantVectorStorePlugin, None], caplog: pytest.LogCaptureFixture):
        store = await anext(qdrant_store)
        caplog.set_level(logging.WARNING)
        await store.setup(config={"embedding_dim": 3})

        async def mixed_dim_stream():
            yield SimpleChunkForQdrant("c1", "text1", {}), [1.0, 2.0, 3.0]
            yield SimpleChunkForQdrant("c2", "text2", {}), [4.0, 5.0]
        result = await store.add(mixed_dim_stream())
        assert result["added_count"] == 1
        assert result["errors"] == ["Dimension mismatch for chunk ID 'c2'. Expected 3, got 2."]


    async def test_add_empty_vector_skipped(self, qdrant_store: AsyncGenerator[QdrantVectorStorePlugin, None]):
        store = await anext(qdrant_store)
        await store.setup(config={"embedding_dim": 3})
        async def stream_with_empty_vec():
            yield SimpleChunkForQdrant("c1", "text1", {}), [1.0, 2.0, 3.0]
            yield SimpleChunkForQdrant("c2", "text2", {}), []
        result = await store.add(stream_with_empty_vec())
        assert result["added_count"] == 1
        assert len(result["errors"]) == 1
        assert result["errors"][0] == "Skipping chunk ID 'c2' due to empty vector."


    async def test_add_upsert_fails(self, qdrant_store: AsyncGenerator[QdrantVectorStorePlugin, None], caplog: pytest.LogCaptureFixture):
        store = await anext(qdrant_store)
        mock_client_instance = store._test_mock_qdrant_client # type: ignore
        caplog.set_level(logging.ERROR, logger=QDRANT_STORE_LOGGER_NAME)
        await store.setup(config={"embedding_dim": 3})
        mock_client_instance.upsert.side_effect = RuntimeError("Qdrant upsert failed") # type: ignore
        result = await store.add(sample_embeddings_stream_qdrant(count=1, dim=3))
        assert result["added_count"] == 0
        assert "Error upserting final batch to Qdrant: Qdrant upsert failed" in result["errors"]


@pytest.mark.asyncio()
class TestQdrantStoreSearchDeleteTeardown:
    async def test_search_successful(self, qdrant_store: AsyncGenerator[QdrantVectorStorePlugin, None]):
        store = await anext(qdrant_store)
        mock_client_instance = store._test_mock_qdrant_client # type: ignore
        await store.setup(config={"embedding_dim": 2, "distance_metric": "Euclid"})

        mock_hit = ActualScoredPoint(id="hit1_id", version=0, score=0.5, payload={"content": "Found content", "metadata": {"src": "docA"}}, vector=None)
        mock_client_instance.search.return_value = [mock_hit] # type: ignore
        results = await store.search(query_embedding=[0.1, 0.2], top_k=1, filter_metadata={"tag": "test"})
        assert len(results) == 1
        assert results[0].id == "hit1_id"

    async def test_search_embedding_dim_mismatch(self, qdrant_store: AsyncGenerator[QdrantVectorStorePlugin, None], caplog: pytest.LogCaptureFixture):
        store = await anext(qdrant_store)
        caplog.set_level(logging.WARNING, logger=QDRANT_STORE_LOGGER_NAME)
        await store.setup(config={"embedding_dim": 3})
        results = await store.search(query_embedding=[0.1, 0.2], top_k=1)
        assert results == []
        assert "Query embedding dim 2 != index dim 3" in caplog.text

    async def test_delete_by_ids(self, qdrant_store: AsyncGenerator[QdrantVectorStorePlugin, None]):
        store = await anext(qdrant_store)
        mock_client_instance = store._test_mock_qdrant_client # type: ignore
        await store.setup()
        ids_to_delete = ["id1", "id2"]
        success = await store.delete(ids=ids_to_delete)
        assert success is True
        mock_client_instance.delete.assert_awaited_once() # type: ignore

    async def test_delete_all(self, qdrant_store: AsyncGenerator[QdrantVectorStorePlugin, None]):
        store = await anext(qdrant_store)
        mock_client_instance = store._test_mock_qdrant_client # type: ignore

        # Simulate get_collection for initial setup (assuming collection exists or is created)
        # Then, for the recreate step after delete_collection, make it raise "Not Found"
        get_collection_call_count = 0
        async def get_collection_side_effect(*args, **kwargs):
            nonlocal get_collection_call_count
            get_collection_call_count += 1
            if get_collection_call_count == 1: # First call during setup
                return MagicMock() # Simulate collection exists
            elif get_collection_call_count == 2: # Second call inside _ensure_collection_exists_async after delete
                raise MockCollectionNotFoundError("Collection deleted, now recreating")
            return MagicMock() # Default for any other unexpected calls

        mock_client_instance.get_collection.side_effect = get_collection_side_effect # type: ignore

        await store.setup(config={"embedding_dim": 3})

        # Reset create_collection mock before calling delete if setup might have called it
        mock_client_instance.create_collection.reset_mock() # type: ignore

        success = await store.delete(delete_all=True)
        assert success is True
        mock_client_instance.delete_collection.assert_awaited_once_with(collection_name=store._collection_name) # type: ignore
        mock_client_instance.create_collection.assert_awaited_once() # type: ignore

    async def test_teardown(self, qdrant_store: AsyncGenerator[QdrantVectorStorePlugin, None]):
        store = await anext(qdrant_store)
        mock_client_instance = store._test_mock_qdrant_client # type: ignore
        await store.setup()
        await store.teardown()
        mock_client_instance.close.assert_awaited_once() # type: ignore
        assert store._client is None
