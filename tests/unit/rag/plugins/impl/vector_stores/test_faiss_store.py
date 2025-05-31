### tests/unit/rag/plugins/impl/vector_stores/test_faiss_store.py
import asyncio
import logging
import pickle
import uuid
from pathlib import Path
from typing import Any, AsyncIterable, Dict, Tuple, Optional
from unittest.mock import MagicMock, patch

import pytest

# Conditional import of numpy for use in tests
try:
    import numpy as actual_np_module
except ImportError:
    actual_np_module = None

from genie_tooling.core.types import Chunk, EmbeddingVector
from genie_tooling.vector_stores.impl.faiss_store import (
    FAISSVectorStore,
    _RetrievedChunkImpl,
)

# Mock faiss and numpy at the module level for all tests in this file
mock_faiss_index_instance = MagicMock(name="GlobalMockFaissIndexInstance")
mock_faiss_module = MagicMock(name="MockFaissModule")
mock_numpy_module = MagicMock(name="MockNumpyModule") # This is what the source file sees as 'np'
mock_numpy_module.linalg = MagicMock(name="MockNumpyLinalgModule")

# Logger for the module under test
FAISS_STORE_LOGGER_NAME = "genie_tooling.vector_stores.impl.faiss_store"


def reset_global_faiss_mocks():
    global mock_faiss_index_instance, mock_faiss_module, mock_numpy_module

    mock_faiss_index_instance.reset_mock()
    mock_faiss_index_instance.d = 0
    mock_faiss_index_instance.ntotal = 0 # CRITICAL: Reset ntotal for each test run

    def mock_add_with_ids_func(vectors, ids_to_add):
        if hasattr(vectors, 'shape'):
            num_added = vectors.shape[0]
            mock_faiss_index_instance.ntotal += num_added
        # No return value for add_with_ids

    mock_faiss_index_instance.add_with_ids = MagicMock(side_effect=mock_add_with_ids_func)

    def mock_remove_ids_func(ids_to_remove_selector):
        if hasattr(ids_to_remove_selector, 'size'):
            num_to_remove_attempt = ids_to_remove_selector.size
            # Simulate actual removal based on what's "in the index"
            # For simplicity, assume all requested IDs are valid if ntotal > 0
            actually_removed = 0
            if mock_faiss_index_instance.ntotal > 0:
                actually_removed = min(num_to_remove_attempt, mock_faiss_index_instance.ntotal)
                mock_faiss_index_instance.ntotal -= actually_removed
            return actually_removed
        return 0

    mock_faiss_index_instance.remove_ids = MagicMock(side_effect=mock_remove_ids_func)

    # Default empty search result
    empty_distances_data = actual_np_module.empty((1,0), dtype=actual_np_module.float32) if actual_np_module else [[]]
    empty_indices_data = actual_np_module.empty((1,0), dtype=actual_np_module.int64) if actual_np_module else [[]]
    empty_distances = robust_np_array_mock_faiss(empty_distances_data)
    empty_indices = robust_np_array_mock_faiss(empty_indices_data)

    mock_faiss_index_instance.search = MagicMock(return_value=(empty_distances, empty_indices))
    mock_faiss_index_instance.reset = MagicMock() # Should reset ntotal to 0
    def mock_reset_func():
        mock_faiss_index_instance.ntotal = 0
    mock_faiss_index_instance.reset.side_effect = mock_reset_func


    mock_faiss_module.reset_mock()
    mock_flat_l2_sub_index = MagicMock(name="MockFlatL2SubIndex")
    mock_faiss_module.IndexFlatL2 = MagicMock(return_value=mock_flat_l2_sub_index)
    mock_faiss_module.IndexIDMap = MagicMock(return_value=mock_faiss_index_instance)
    mock_faiss_module.index_factory = MagicMock(return_value=mock_faiss_index_instance)

    mock_faiss_module.read_index = MagicMock(return_value=mock_faiss_index_instance)
    mock_faiss_module.write_index = MagicMock()

    # Configure the mock_numpy_module that the source file will import as 'np'
    mock_numpy_module.reset_mock()
    mock_numpy_module.array = MagicMock(side_effect=robust_np_array_mock_faiss)
    mock_numpy_module.concatenate = MagicMock(side_effect=lambda arr_list, axis: robust_np_array_mock_faiss([item for sublist in [a.tolist() for a in arr_list] for item in (sublist if isinstance(sublist[0], list) else [sublist])]))
    mock_numpy_module.float32 = type("float32", (), {})
    mock_numpy_module.linalg.norm = MagicMock(return_value=1.0)
    mock_numpy_module.int64 = type("int64", (), {})

def robust_np_array_mock_faiss(data, dtype=None):
    arr_instance = MagicMock(name=f"MockNpArray_{str(data)[:10]}")
    _data = list(data) if not isinstance(data, list) else data
    arr_instance.tolist = lambda: _data
    arr_instance.dtype = dtype or mock_numpy_module.float32 # Use the mocked numpy for dtype

    if not _data or (isinstance(_data, list) and len(_data) == 1 and isinstance(_data[0], list) and not _data[0]):
        arr_instance.ndim = 1 if not _data or not isinstance(_data[0], list) else 2
        arr_instance.shape = (0,) if arr_instance.ndim == 1 and not _data else (1, 0) if arr_instance.ndim == 2 and len(_data) == 1 and not _data[0] else (0, 0)
        arr_instance.size = 0
    else:
        first_el = _data[0]
        arr_instance.ndim = 2 if isinstance(first_el, list) else 1

        if arr_instance.ndim == 2:
            arr_instance.shape = (len(_data), len(first_el) if first_el and isinstance(first_el, list) else 0)
        else: # ndim == 1
            arr_instance.shape = (len(_data),)
        arr_instance.size = arr_instance.shape[0] * (arr_instance.shape[1] if arr_instance.ndim == 2 and arr_instance.shape[1] > 0 else 1 if arr_instance.ndim == 1 else 0)


    def mock_getitem(key):
        if isinstance(key, tuple) and len(key) == 2 and arr_instance.ndim == 2:
            row, col = key
            if 0 <= row < arr_instance.shape[0] and 0 <= col < arr_instance.shape[1]:
                return _data[row][col]
        elif isinstance(key, int) and arr_instance.ndim == 1:
             if 0 <= key < arr_instance.shape[0]:
                return _data[key]
        # This mock was too simple for how numpy arrays are used (e.g. .shape attribute)
        # For tests, it's better if the mock search returns arrays with correct .shape
        raise IndexError(f"MockNpArray: Basic __getitem__ doesn't support key {key} for data shape {arr_instance.shape}")
    arr_instance.__getitem__ = MagicMock(side_effect=mock_getitem)

    arr_instance.reshape = MagicMock(side_effect=lambda new_shape, order=None: robust_np_array_mock_faiss([arr_instance.tolist()] if new_shape == (1, -1) and arr_instance.ndim == 1 else _data, dtype=arr_instance.dtype))
    return arr_instance


@pytest.fixture
async def faiss_store_fixture(request) -> FAISSVectorStore:
    store = FAISSVectorStore()
    # No need to await store.setup() here, tests will call it with specific configs
    async def finalizer_async():
        # Check if _lock exists before trying to acquire, in case setup failed very early
        if hasattr(store, "_lock") and store._lock.locked():
            try:
                store._lock.release() # Try to release if locked to prevent deadlock in other tests
            except RuntimeError:
                pass # Already unlocked
        if hasattr(store, "_index") and store._index: await store.teardown()
        elif hasattr(store, "_index_file_path") and store._index_file_path: await store.teardown()

    request.addfinalizer(lambda: asyncio.run(finalizer_async()))
    return store

@pytest.fixture(autouse=True)
def mock_faiss_dependencies_fixt():
    reset_global_faiss_mocks() # This ensures ntotal is 0 on the global mock at start of each test
    # This patches 'np' and 'faiss' in the *source file* (faiss_store.py)
    with patch("genie_tooling.vector_stores.impl.faiss_store.faiss", mock_faiss_module), \
         patch("genie_tooling.vector_stores.impl.faiss_store.np", mock_numpy_module):
        yield

class SimpleChunk(Chunk):
    def __init__(self, id: Optional[str], content: str, metadata: Dict[str,Any]):
        self.id=id; self.content=content; self.metadata=metadata
async def sample_embeddings_stream_faiss(count=2,dim=3,start_id=0) -> AsyncIterable[Tuple[Chunk,EmbeddingVector]]:
    for i in range(count): yield SimpleChunk(f"c{start_id+i}",f"Ct{start_id+i}",{"idx":start_id+i}),[float(start_id+i+j*0.1) for j in range(dim)]

@pytest.mark.asyncio
async def test_faiss_setup_default_paths(faiss_store_fixture: FAISSVectorStore, caplog: pytest.LogCaptureFixture):
    store = await faiss_store_fixture
    caplog.set_level(logging.INFO, logger=FAISS_STORE_LOGGER_NAME)
    collection_name = "my_faiss_test_collection"
    with patch.object(Path, "mkdir") as mock_mkdir, \
         patch.object(Path, "exists", return_value=False):
        await store.setup(config={
            "collection_name": collection_name, "embedding_dim": 3, "persist_by_default": True
        })
    expected_idx_path = Path(f"./.genie_data/faiss/{collection_name}.faissindex")
    expected_docs_path = Path(f"./.genie_data/faiss/{collection_name}.faissdocs")

    assert store._index_file_path == expected_idx_path
    assert store._doc_store_file_path == expected_docs_path
    mock_mkdir.assert_any_call(parents=True, exist_ok=True)
    assert f"Index path set to '{str(expected_idx_path)}'." in caplog.text
    assert f"Doc store path set to '{str(expected_docs_path)}'." in caplog.text
    assert "Index/doc files not found at configured paths." in caplog.text
    assert "Initialized FAISS IndexIDMap wrapping Flat with dim 3." in caplog.text

@pytest.mark.asyncio
async def test_faiss_add_infer_dim_and_initialize(faiss_store_fixture: FAISSVectorStore):
    faiss_store = await faiss_store_fixture
    await faiss_store.setup(config={"persist_by_default": False})
    assert faiss_store._index is None
    dim_to_infer = 5

    async def mock_run_in_executor(executor, func_partial): return func_partial()
    with patch("asyncio.BaseEventLoop.run_in_executor", side_effect=mock_run_in_executor):
        result = await faiss_store.add(sample_embeddings_stream_faiss(1, dim=dim_to_infer))

    assert result["added_count"] == 1
    assert faiss_store._index is mock_faiss_index_instance
    assert faiss_store._embedding_dim == dim_to_infer
    mock_faiss_module.IndexFlatL2.assert_called_once_with(dim_to_infer)
    mock_faiss_module.IndexIDMap.assert_called_once_with(mock_faiss_module.IndexFlatL2.return_value)
    mock_faiss_index_instance.add_with_ids.assert_called_once()
    assert mock_faiss_index_instance.ntotal == 1


@pytest.mark.asyncio
async def test_faiss_setup_load_corrupt_index_file(faiss_store_fixture: FAISSVectorStore, tmp_path: Path, caplog: pytest.LogCaptureFixture):
    store = await faiss_store_fixture
    caplog.set_level(logging.INFO, logger=FAISS_STORE_LOGGER_NAME)
    idx_path = tmp_path / "corrupt.faissindex"
    doc_path = tmp_path / "corrupt.faissdocs"
    idx_path.write_bytes(b"not a faiss index") # write_bytes
    doc_path.write_bytes(pickle.dumps({"doc_store_by_faiss_idx": {}, "chunk_id_to_faiss_idx": {}}))

    mock_faiss_module.read_index.side_effect = RuntimeError("FAISS read error")
    await store.setup(config={"index_file_path": str(idx_path), "doc_store_file_path": str(doc_path), "embedding_dim": 3})

    assert store._index is not None
    assert any("Error loading FAISS from files: FAISS read error" in rec.message for rec in caplog.records if rec.levelno == logging.ERROR)
    assert any("Initialized FAISS IndexIDMap wrapping Flat with dim 3." in rec.message for rec in caplog.records if rec.levelno == logging.INFO)


@pytest.mark.asyncio
async def test_faiss_setup_load_corrupt_doc_store_file(faiss_store_fixture: FAISSVectorStore, tmp_path: Path, caplog: pytest.LogCaptureFixture):
    store = await faiss_store_fixture
    caplog.set_level(logging.INFO, logger=FAISS_STORE_LOGGER_NAME)
    idx_path = tmp_path / "good.faissindex"
    doc_path = tmp_path / "corrupt.faissdocs"

    mock_faiss_module.read_index.return_value = mock_faiss_index_instance
    mock_faiss_index_instance.d = 3
    mock_faiss_index_instance.ntotal = 0
    idx_path.touch()

    doc_path.write_text("not a pickle")

    await store.setup(config={"index_file_path": str(idx_path), "doc_store_file_path": str(doc_path), "embedding_dim": 3})

    assert store._index is not None
    assert any("Error unpickling doc store" in rec.message for rec in caplog.records if rec.levelno == logging.ERROR)
    assert any("Initialized FAISS IndexIDMap wrapping Flat with dim 3." in rec.message for rec in caplog.records if rec.levelno == logging.INFO)


@pytest.mark.asyncio
async def test_faiss_setup_index_factory_fails(faiss_store_fixture: FAISSVectorStore, caplog: pytest.LogCaptureFixture):
    store = await faiss_store_fixture
    caplog.set_level(logging.ERROR, logger=FAISS_STORE_LOGGER_NAME)
    mock_faiss_module.index_factory.side_effect = RuntimeError("FAISS factory error")
    await store.setup(config={"embedding_dim": 3, "faiss_index_factory_string": "IVF1,Flat"})
    assert store._index is None
    assert "Failed to initialize FAISS index (dim 3, factory 'IVF1,Flat'): FAISS factory error" in caplog.text

@pytest.mark.asyncio
async def test_faiss_add_to_uninitialized_index_no_dim(faiss_store_fixture: FAISSVectorStore, caplog: pytest.LogCaptureFixture):
    store = await faiss_store_fixture
    caplog.set_level(logging.INFO, logger=FAISS_STORE_LOGGER_NAME)
    await store.setup(config={"persist_by_default": False})
    assert store._index is None
    assert store._embedding_dim is None

    async def empty_stream():
        if False: yield
    result = await store.add(empty_stream())
    assert result["added_count"] == 0
    assert not result["errors"]
    assert store._index is None
    assert "Ready. Index will be init on first data add if dim known." in caplog.text

@pytest.mark.asyncio
async def test_faiss_add_vector_dim_mismatch(faiss_store_fixture: FAISSVectorStore):
    store = await faiss_store_fixture
    await store.setup(config={"embedding_dim": 3})
    assert store._index is not None

    async def mixed_dim_stream():
        yield SimpleChunk("c1", "Ct1", {}), [1.0, 2.0, 3.0]
        yield SimpleChunk("c2", "Ct2", {}), [4.0, 5.0]
        yield SimpleChunk("c3", "Ct3", {}), []

    result = await store.add(mixed_dim_stream())
    assert result["added_count"] == 1
    assert mock_faiss_index_instance.ntotal == 1
    assert len(result["errors"]) == 2
    assert "Dimension mismatch or empty vector for chunk 'c2'. Expected 3, got 2." in result["errors"]
    assert "Dimension mismatch or empty vector for chunk 'c3'. Expected 3, got empty." in result["errors"]

@pytest.mark.asyncio
async def test_faiss_search_empty_index(faiss_store_fixture: FAISSVectorStore):
    store = await faiss_store_fixture
    await store.setup(config={"embedding_dim": 3})
    results = await store.search([1.0, 2.0, 3.0], top_k=5)
    assert results == []

@pytest.mark.asyncio
async def test_faiss_search_query_dim_mismatch(faiss_store_fixture: FAISSVectorStore, caplog: pytest.LogCaptureFixture):
    store = await faiss_store_fixture
    caplog.set_level(logging.WARNING, logger=FAISS_STORE_LOGGER_NAME)
    await store.setup(config={"embedding_dim": 3})
    await store.add(sample_embeddings_stream_faiss(1, dim=3))

    results = await store.search([1.0, 2.0], top_k=1)
    assert results == []
    assert "Query embedding dim 2 != index dim 3." in caplog.text

@pytest.mark.asyncio
async def test_faiss_search_with_filter_metadata(faiss_store_fixture: FAISSVectorStore):
    store = await faiss_store_fixture
    await store.setup(config={"embedding_dim": 2})
    async def stream_for_filter():
        yield SimpleChunk("id1", "content one", {"source": "A", "tag": "X"}), [0.1, 0.1]
        yield SimpleChunk("id2", "content two", {"source": "B", "tag": "X"}), [0.2, 0.2]
        yield SimpleChunk("id3", "content three", {"source": "A", "tag": "Y"}), [0.3, 0.3]
    await store.add(stream_for_filter())
    assert mock_faiss_index_instance.ntotal == 3

    mock_faiss_index_instance.search = MagicMock(return_value=(
        robust_np_array_mock_faiss([[0.0, 0.01, 0.02]]),
        robust_np_array_mock_faiss([[0, 1, 2]])
    ))

    results_source_A = await store.search([0.15, 0.15], top_k=3, filter_metadata={"source": "A"})
    assert len(results_source_A) == 2
    assert {r.id for r in results_source_A} == {"id1", "id3"}

    results_tag_X = await store.search([0.15, 0.15], top_k=3, filter_metadata={"tag": "X"})
    assert len(results_tag_X) == 2
    assert {r.id for r in results_tag_X} == {"id1", "id2"}

    results_no_match = await store.search([0.15, 0.15], top_k=3, filter_metadata={"source": "C"})
    assert len(results_no_match) == 0

@pytest.mark.asyncio
async def test_faiss_delete_by_ids_some_not_exist(faiss_store_fixture: FAISSVectorStore):
    store = await faiss_store_fixture
    await store.setup(config={"embedding_dim": 2})
    await store.add(sample_embeddings_stream_faiss(3, dim=2, start_id=0))
    assert mock_faiss_index_instance.ntotal == 3

    success = await store.delete(ids=["c0", "c2", "c99"])
    assert success is True
    assert mock_faiss_index_instance.ntotal == 1
    assert "c0" not in store._chunk_id_to_faiss_idx
    assert "c2" not in store._chunk_id_to_faiss_idx
    assert "c1" in store._chunk_id_to_faiss_idx

@pytest.mark.asyncio
async def test_faiss_delete_by_filter_metadata(faiss_store_fixture: FAISSVectorStore, caplog: pytest.LogCaptureFixture):
    store = await faiss_store_fixture
    caplog.set_level(logging.WARNING, logger=FAISS_STORE_LOGGER_NAME)
    await store.setup(config={"embedding_dim": 2})
    async def stream_for_delete_filter():
        yield SimpleChunk("del_f1", "content one", {"category": "alpha"}), [0.1, 0.1]
        yield SimpleChunk("del_f2", "content two", {"category": "beta"}), [0.2, 0.2]
        yield SimpleChunk("del_f3", "content three", {"category": "alpha"}), [0.3, 0.3]
    await store.add(stream_for_delete_filter())
    assert mock_faiss_index_instance.ntotal == 3

    success = await store.delete(filter_metadata={"category": "alpha"})
    assert success is True
    assert "Delete by metadata filter is NOT performant for FAISS" in caplog.text
    assert mock_faiss_index_instance.ntotal == 1
    assert "del_f1" not in store._chunk_id_to_faiss_idx
    assert "del_f3" not in store._chunk_id_to_faiss_idx
    assert "del_f2" in store._chunk_id_to_faiss_idx

@pytest.mark.asyncio
async def test_faiss_delete_all(faiss_store_fixture: FAISSVectorStore):
    store = await faiss_store_fixture
    await store.setup(config={"embedding_dim": 2})
    await store.add(sample_embeddings_stream_faiss(5, dim=2))
    assert mock_faiss_index_instance.ntotal == 5
    assert len(store._doc_store_by_faiss_idx) == 5

    success = await store.delete(delete_all=True)
    assert success is True
    assert store._index is not None
    assert mock_faiss_index_instance.ntotal == 0
    assert len(store._doc_store_by_faiss_idx) == 0
    assert len(store._chunk_id_to_faiss_idx) == 0
    assert store._next_faiss_idx == 0

@pytest.mark.asyncio
async def test_faiss_delete_index_not_initialized(faiss_store_fixture: FAISSVectorStore, caplog: pytest.LogCaptureFixture):
    store = await faiss_store_fixture
    caplog.set_level(logging.WARNING, logger=FAISS_STORE_LOGGER_NAME)
    store._index = None
    success = await store.delete(ids=["any_id"])
    assert success is False
    assert f"{store.plugin_id}: Cannot delete by IDs or filter, FAISS index is not initialized." in caplog.text

@pytest.mark.asyncio
async def test_faiss_save_files_paths_not_set(faiss_store_fixture: FAISSVectorStore, caplog: pytest.LogCaptureFixture):
    store = await faiss_store_fixture
    caplog.set_level(logging.DEBUG, logger=FAISS_STORE_LOGGER_NAME)
    await store.setup(config={"embedding_dim": 3, "persist_by_default": False})
    await store.add(sample_embeddings_stream_faiss(1, dim=3))
    await store._save_to_files()
    assert "FAISS save skipped (paths/index not fully available)." not in caplog.text

@pytest.mark.asyncio
async def test_faiss_load_files_paths_not_set(faiss_store_fixture: FAISSVectorStore):
    store = await faiss_store_fixture
    await store.setup(config={"persist_by_default": False})
    await store._load_from_files()
    assert store._index is None

@pytest.mark.asyncio
async def test_faiss_save_write_index_fails(faiss_store_fixture: FAISSVectorStore, tmp_path: Path, caplog: pytest.LogCaptureFixture):
    store = await faiss_store_fixture
    caplog.set_level(logging.ERROR, logger=FAISS_STORE_LOGGER_NAME)
    idx_path = tmp_path / "save_fail.faissindex"
    doc_path = tmp_path / "save_fail.faissdocs"
    await store.setup(config={"embedding_dim": 3, "index_file_path": str(idx_path), "doc_store_file_path": str(doc_path)})
    await store.add(sample_embeddings_stream_faiss(1, dim=3))

    mock_faiss_module.write_index.side_effect = RuntimeError("FAISS write error")
    await store._save_to_files()
    assert "Error saving FAISS to files: FAISS write error" in caplog.text

@pytest.mark.asyncio
async def test_faiss_save_pickle_fails(faiss_store_fixture: FAISSVectorStore, tmp_path: Path, caplog: pytest.LogCaptureFixture):
    store = await faiss_store_fixture
    caplog.set_level(logging.ERROR, logger=FAISS_STORE_LOGGER_NAME)
    idx_path = tmp_path / "pickle_fail.faissindex"
    doc_path = tmp_path / "pickle_fail.faissdocs"
    await store.setup(config={"embedding_dim": 3, "index_file_path": str(idx_path), "doc_store_file_path": str(doc_path)})
    await store.add(sample_embeddings_stream_faiss(1, dim=3))

    with patch("pickle.dumps", side_effect=pickle.PicklingError("Pickle dump error")):
        await store._save_to_files()
    assert "Error saving FAISS to files: Pickle dump error" in caplog.text

@pytest.mark.asyncio
async def test_faiss_add_chunk_id_none(faiss_store_fixture: FAISSVectorStore):
    store = await faiss_store_fixture
    await store.setup(config={"embedding_dim": 2})
    async def stream_no_id():
        yield SimpleChunk(id=None, content="no id content", metadata={}), [0.5, 0.5]

    result = await store.add(stream_no_id())
    assert result["added_count"] == 1
    assert mock_faiss_index_instance.ntotal == 1
    assert len(store._chunk_id_to_faiss_idx) == 1
    generated_id = list(store._chunk_id_to_faiss_idx.keys())[0]
    try:
        uuid.UUID(generated_id)
        is_uuid = True
    except ValueError:
        is_uuid = False
    assert is_uuid

@pytest.mark.asyncio
async def test_faiss_add_batch_to_faiss_and_docstore_empty_inputs(faiss_store_fixture: FAISSVectorStore):
    store = await faiss_store_fixture
    await store.setup(config={"embedding_dim": 2})

    added_count = await store._add_batch_to_faiss_and_docstore([], [])
    assert added_count == 0

    store._index = None
    chunks_with_data = [SimpleChunk("id1", "content", {})]
    vectors_with_data = [robust_np_array_mock_faiss([[0.1,0.2]])]
    added_count_no_index = await store._add_batch_to_faiss_and_docstore(chunks_with_data, vectors_with_data)
    assert added_count_no_index == 0

@pytest.mark.asyncio
async def test_faiss_add_batch_to_faiss_and_docstore_faiss_add_fails(faiss_store_fixture: FAISSVectorStore, caplog: pytest.LogCaptureFixture):
    store = await faiss_store_fixture
    caplog.set_level(logging.ERROR, logger=FAISS_STORE_LOGGER_NAME)
    await store.setup(config={"embedding_dim": 2})

    mock_faiss_index_instance.add_with_ids.side_effect = RuntimeError("FAISS add_with_ids failed")

    chunks = [SimpleChunk("id1", "content", {})]
    vectors = [robust_np_array_mock_faiss([[0.1,0.2]])]

    added_count = await store._add_batch_to_faiss_and_docstore(chunks, vectors)
    assert added_count == 0
    assert "Error in _sync_add_batch (FAISS add_with_ids or mapping): FAISS add_with_ids failed" in caplog.text

@pytest.mark.asyncio
async def test_faiss_search_faiss_search_fails(faiss_store_fixture: FAISSVectorStore, caplog: pytest.LogCaptureFixture):
    store = await faiss_store_fixture
    caplog.set_level(logging.ERROR, logger=FAISS_STORE_LOGGER_NAME)
    await store.setup(config={"embedding_dim": 2})
    await store.add(sample_embeddings_stream_faiss(1, dim=2))

    mock_faiss_index_instance.search.side_effect = RuntimeError("FAISS search failed")

    results = await store.search([0.1, 0.2], top_k=1)
    assert results == []
    assert "FAISS search operation error: FAISS search failed" in caplog.text

@pytest.mark.asyncio
async def test_faiss_delete_faiss_remove_ids_fails(faiss_store_fixture: FAISSVectorStore, caplog: pytest.LogCaptureFixture):
    store = await faiss_store_fixture
    caplog.set_level(logging.ERROR, logger=FAISS_STORE_LOGGER_NAME)
    await store.setup(config={"embedding_dim": 2})
    await store.add(sample_embeddings_stream_faiss(1, dim=2, start_id=5)) # c5
    assert mock_faiss_index_instance.ntotal == 1

    mock_faiss_index_instance.remove_ids.side_effect = RuntimeError("FAISS remove_ids failed")

    success = await store.delete(ids=["c5"])
    assert success is False
    assert "Error during FAISS remove_ids or doc store cleanup: FAISS remove_ids failed" in caplog.text
    assert "c5" in store._chunk_id_to_faiss_idx
    assert mock_faiss_index_instance.ntotal == 1