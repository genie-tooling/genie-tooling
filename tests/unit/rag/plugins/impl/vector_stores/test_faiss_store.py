###tests/unit/rag/plugins/impl/vector_stores/test_faiss_store.py###
"""Unit tests for FAISSVectorStore."""
import asyncio
import logging
import pickle
from pathlib import Path
from typing import Any, AsyncIterable, Dict, Tuple
from unittest.mock import (  # Removed mock_open, it's not suitable here
    AsyncMock,
    MagicMock,
    patch,
)

import pytest
from genie_tooling.core.types import Chunk, EmbeddingVector
from genie_tooling.rag.plugins.impl.vector_stores.faiss_store import (
    FAISSVectorStore,
)

# Global mock for the FAISS index *instance*
mock_faiss_index_instance = MagicMock(name="GlobalMockFaissIndexInstance")

mock_faiss_module = MagicMock(name="MockFaissModule")
mock_numpy_module = MagicMock(name="MockNumpyModule")
mock_numpy_module.linalg = MagicMock(name="MockNumpyLinalgModule")


def reset_global_faiss_mocks():
    global mock_faiss_index_instance, mock_faiss_module, mock_numpy_module
    mock_faiss_index_instance.reset_mock()
    mock_faiss_index_instance.d = 0
    mock_faiss_index_instance.ntotal = 0
    mock_faiss_index_instance.add = MagicMock(name="FaissIndexInstance.add", return_value=None)
    empty_dist_arr = robust_np_array_mock_faiss([[]])
    empty_idx_arr = robust_np_array_mock_faiss([[]])
    if hasattr(empty_idx_arr, "size"): empty_idx_arr.size = 0
    mock_faiss_index_instance.search = MagicMock(name="FaissIndexInstance.search", return_value=(empty_dist_arr, empty_idx_arr))
    mock_faiss_index_instance.reset = MagicMock(name="FaissIndexInstance.reset")
    mock_faiss_index_instance.remove_ids = MagicMock(name="FaissIndexInstance.remove_ids", return_value=0)

    mock_faiss_module.reset_mock()
    mock_faiss_module.IndexFlatL2 = MagicMock(name="FaissModule.IndexFlatL2Constructor", return_value=mock_faiss_index_instance)
    mock_faiss_module.index_factory = MagicMock(name="FaissModule.index_factoryConstructor", return_value=mock_faiss_index_instance)
    mock_faiss_module.read_index = MagicMock(name="FaissModule.read_index", return_value=mock_faiss_index_instance)
    mock_faiss_module.write_index = MagicMock(name="FaissModule.write_index")
    mock_faiss_module.IndexIDMap = MagicMock(name="FaissModule.IndexIDMapConstructor", return_value=mock_faiss_index_instance)

    mock_numpy_module.reset_mock()
    mock_numpy_module.array = MagicMock(name="NumpyModule.array", side_effect=robust_np_array_mock_faiss)
    def concat_side_effect(list_of_mock_arrays, axis):
        all_data = []
        for mock_arr_item in list_of_mock_arrays:
            data_list = mock_arr_item.tolist()
            if data_list and isinstance(data_list[0], list):
                 all_data.extend(data_list)
            else:
                 all_data.append(data_list)
        flat_vectors = []
        for item_list_concat in all_data:
            if item_list_concat and isinstance(item_list_concat[0], list):
                flat_vectors.extend(item_list_concat)
            else:
                flat_vectors.append(item_list_concat)

        return robust_np_array_mock_faiss(flat_vectors if flat_vectors else [])


    mock_numpy_module.concatenate = MagicMock(name="NumpyModule.concatenate", side_effect=concat_side_effect)
    mock_numpy_module.float32 = type("float32", (), {})
    mock_numpy_module.linalg.norm = MagicMock(name="NumpyLinalg.norm", return_value=1.0)
    mock_numpy_module.int64 = type("int64", (), {})


def robust_np_array_mock_faiss(data, dtype=None):
    arr_instance = MagicMock(name=f"MockNpArray_{str(data)[:20]}")
    _data_internal = list(data) if not isinstance(data, list) else data
    arr_instance.tolist = lambda: _data_internal
    arr_instance.dtype = dtype if dtype else mock_numpy_module.float32

    if not _data_internal or \
       (isinstance(_data_internal, list) and len(_data_internal) == 1 and isinstance(_data_internal[0], list) and not _data_internal[0]) :
        arr_instance.ndim = 1 if not _data_internal or not isinstance(_data_internal[0], list) else 2
        arr_instance.shape = (0,) if arr_instance.ndim == 1 and not _data_internal else \
                             (1,0) if arr_instance.ndim == 2 and len(_data_internal) == 1 and not _data_internal[0] else \
                             (0,0)
        arr_instance.size = 0
    else:
        first_element = _data_internal[0]
        if isinstance(first_element, list):
            arr_instance.ndim = 2
            arr_instance.shape = (len(_data_internal), len(first_element) if first_element else 0)
        else:
            arr_instance.ndim = 1
            arr_instance.shape = (len(_data_internal),)

        if arr_instance.ndim == 1:
            arr_instance.size = arr_instance.shape[0]
        elif arr_instance.ndim == 2:
            arr_instance.size = arr_instance.shape[0] * arr_instance.shape[1]
        else:
            arr_instance.size = 0

    def _reshape_mock_for_arr(new_shape_arg, order=None):
        if isinstance(new_shape_arg, tuple):
            if new_shape_arg == (1, -1) and arr_instance.ndim == 1:
                arr_content_for_reshape = [arr_instance.tolist()]
                return robust_np_array_mock_faiss(arr_content_for_reshape, dtype=arr_instance.dtype)
            arr_instance.shape = new_shape_arg
        return arr_instance

    arr_instance.reshape = MagicMock(name=f"NpArray.reshape_for_{str(data)[:20]}", side_effect=_reshape_mock_for_arr)

    def getitem_mock(key):
        if isinstance(key, tuple) and len(key) == 2:
            row, col = key
            if row < len(_data_internal) and isinstance(_data_internal[row], list) and col < len(_data_internal[row]):
                return _data_internal[row][col]
            if row == 0 and arr_instance.shape[0] == 1 and col < arr_instance.shape[1]:
                 if isinstance(_data_internal[0], list) and col < len(_data_internal[0]):
                      return _data_internal[0][col]

        elif isinstance(key, int) and key < len(_data_internal):
             return _data_internal[key]
        raise IndexError(f"Mock np array index out of bounds: {key} for data {_data_internal}")

    arr_instance.__getitem__ = MagicMock(name=f"NpArray.getitem_for_{str(data)[:20]}", side_effect=getitem_mock)
    return arr_instance


@pytest.fixture
async def faiss_store_fixture(request) -> FAISSVectorStore:
    store = FAISSVectorStore()

    async def finalizer_async():
        if hasattr(store, "_index") and store._index is not None:
            await store.teardown()
        elif hasattr(store, "_index_file_path") and store._index_file_path is not None:
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
             #logger.warning("Async finalizer for faiss_store_fixture called while event loop is running.")
             asyncio.ensure_future(finalizer_async(), loop=loop)
        else:
            loop.run_until_complete(finalizer_async())

    request.addfinalizer(finalizer_sync)
    return store


@pytest.fixture(autouse=True)
def mock_faiss_dependencies_fixt():
    reset_global_faiss_mocks()
    with patch("genie_tooling.rag.plugins.impl.vector_stores.faiss_store.faiss", mock_faiss_module), \
         patch("genie_tooling.rag.plugins.impl.vector_stores.faiss_store.np", mock_numpy_module):
        yield


class SimpleChunk(Chunk):
    def __init__(self, id: str, content: str, metadata: Dict[str, Any]):
        self.id = id
        self.content = content
        self.metadata = metadata

async def sample_embeddings_stream_faiss(count: int = 2, dim: int = 3, start_id: int = 0) -> AsyncIterable[Tuple[Chunk, EmbeddingVector]]:
    for i in range(count):
        chunk_id = f"chunk{start_id + i}"
        chunk = SimpleChunk(id=chunk_id, content=f"Content for {chunk_id}", metadata={"idx": start_id + i})
        vector = [float(start_id + i + j * 0.1) for j in range(dim)]
        yield chunk, vector

@pytest.mark.asyncio
async def test_faiss_setup_no_dim_no_files(faiss_store_fixture: FAISSVectorStore, caplog):
    faiss_store = await faiss_store_fixture
    caplog.set_level(logging.INFO)
    await faiss_store.setup()
    assert faiss_store._index is None
    assert faiss_store._embedding_dim is None
    assert "Index will be initialized when the first batch of embeddings" in caplog.text

@pytest.mark.asyncio
async def test_faiss_setup_with_dim(faiss_store_fixture: FAISSVectorStore):
    faiss_store = await faiss_store_fixture
    dim = 128
    await faiss_store.setup(config={"embedding_dim": dim})
    assert faiss_store._index is mock_faiss_index_instance
    assert faiss_store._embedding_dim == dim
    mock_faiss_module.IndexFlatL2.assert_called_once_with(dim)

@pytest.mark.asyncio
async def test_faiss_setup_with_index_factory(faiss_store_fixture: FAISSVectorStore):
    faiss_store = await faiss_store_fixture
    dim = 64
    factory_str = "IVF10,Flat"
    await faiss_store.setup(config={"embedding_dim": dim, "faiss_index_factory_string": factory_str})
    assert faiss_store._index is mock_faiss_index_instance
    mock_faiss_module.index_factory.assert_called_once_with(dim, factory_str)

@pytest.mark.asyncio
async def test_faiss_add_infer_dim_and_initialize(faiss_store_fixture: FAISSVectorStore):
    faiss_store = await faiss_store_fixture
    await faiss_store.setup()
    assert faiss_store._index is None

    dim_to_infer = 5
    mock_faiss_index_instance.add.return_value = None

    async def mock_run_in_executor_add_infer(executor, func_partial):
        num_added_by_sync_func = func_partial()
        return num_added_by_sync_func

    with patch("asyncio.BaseEventLoop.run_in_executor", side_effect=mock_run_in_executor_add_infer):
        result = await faiss_store.add(sample_embeddings_stream_faiss(1, dim=dim_to_infer))

    assert result["added_count"] == 1
    assert faiss_store._index is mock_faiss_index_instance
    assert faiss_store._embedding_dim == dim_to_infer
    mock_faiss_module.IndexFlatL2.assert_called_once_with(dim_to_infer)
    mock_faiss_index_instance.add.assert_called_once()


@pytest.mark.asyncio
async def test_faiss_add_batching(faiss_store_fixture: FAISSVectorStore):
    faiss_store = await faiss_store_fixture
    dim = 3
    await faiss_store.setup(config={"embedding_dim": dim})

    mock_faiss_index_instance.add.return_value = None

    sync_add_batch_call_counts_returned = []

    async def mock_run_in_executor_batching(executor, func_partial):
        items_in_this_batch_call = func_partial()
        sync_add_batch_call_counts_returned.append(items_in_this_batch_call)
        return items_in_this_batch_call

    with patch("asyncio.BaseEventLoop.run_in_executor", side_effect=mock_run_in_executor_batching):
        result = await faiss_store.add(sample_embeddings_stream_faiss(5, dim=dim), config={"batch_size": 2})

    assert result["added_count"] == 5
    assert sync_add_batch_call_counts_returned == [2, 2, 1]
    assert mock_faiss_index_instance.add.call_count == 3


@pytest.mark.asyncio
async def test_faiss_add_dimension_mismatch(faiss_store_fixture: FAISSVectorStore, caplog):
    faiss_store = await faiss_store_fixture
    caplog.set_level(logging.WARNING)
    dim = 3
    await faiss_store.setup(config={"embedding_dim": dim})

    async def mismatched_stream():
        yield SimpleChunk("c1", "good", {}), [0.1, 0.2, 0.3]
        yield SimpleChunk("c2_bad", "bad dim", {}), [0.1, 0.2]

    async def mock_run_in_executor_dim_mismatch(executor, func_partial):
        items_added_by_sync = func_partial()
        return items_added_by_sync

    with patch("asyncio.BaseEventLoop.run_in_executor", side_effect=mock_run_in_executor_dim_mismatch):
        result = await faiss_store.add(mismatched_stream())

    assert result["added_count"] == 1
    assert len(result["errors"]) == 1
    assert "Embedding dimension mismatch for chunk ID 'c2_bad'" in result["errors"][0]
    assert "Embedding dimension mismatch for chunk ID 'c2_bad'" in caplog.text


@pytest.mark.asyncio
async def test_faiss_search_successful(faiss_store_fixture: FAISSVectorStore):
    faiss_store = await faiss_store_fixture
    dim = 3
    await faiss_store.setup(config={"embedding_dim": dim})

    faiss_store._doc_store_by_faiss_idx = {
        0: SimpleChunk(id="chunk0", content="Content for chunk0", metadata={"idx": 0}),
        1: SimpleChunk(id="chunk1", content="Content for chunk1", metadata={"idx": 1})
    }
    faiss_store._chunk_id_to_faiss_idx = {"chunk0": 0, "chunk1": 1}
    faiss_store._next_faiss_idx = 2
    mock_faiss_index_instance.ntotal = 2

    mock_distances_np = robust_np_array_mock_faiss([[0.1]])
    mock_indices_np = robust_np_array_mock_faiss([[0]])
    mock_faiss_index_instance.search.return_value = (mock_distances_np, mock_indices_np)

    query_vec = [0.0, 0.1, 0.2]
    results = await faiss_store.search(query_vec, top_k=1, filter_metadata={"idx": 0})

    assert len(results) == 1
    assert results[0].id == "chunk0"
    assert results[0].metadata == {"idx": 0}
    assert isinstance(results[0].score, float)
    assert 0.0 <= results[0].score <= 1.0

    mock_faiss_index_instance.search.assert_called_once()
    call_args, call_kwargs = mock_faiss_index_instance.search.call_args
    assert call_args[1] == 1


@pytest.mark.asyncio
async def test_faiss_search_empty_index(faiss_store_fixture: FAISSVectorStore, caplog):
    faiss_store = await faiss_store_fixture
    caplog.set_level(logging.DEBUG)
    await faiss_store.setup(config={"embedding_dim": 3})
    mock_faiss_index_instance.ntotal = 0

    results = await faiss_store.search([0.1,0.2,0.3], top_k=5)
    assert results == []
    assert "FAISSVectorStore Search: Index is empty." in caplog.text


@pytest.mark.asyncio
async def test_faiss_delete_all(faiss_store_fixture: FAISSVectorStore):
    faiss_store = await faiss_store_fixture
    dim = 3
    await faiss_store.setup(config={"embedding_dim": dim})

    faiss_store._doc_store_by_faiss_idx = {0: SimpleChunk("id0","c0",{})}
    faiss_store._chunk_id_to_faiss_idx = {"id0":0}
    faiss_store._next_faiss_idx = 1
    mock_faiss_index_instance.ntotal = 1

    mock_faiss_module.IndexFlatL2.reset_mock()

    success = await faiss_store.delete(delete_all=True)
    assert success is True
    mock_faiss_module.IndexFlatL2.assert_called_with(dim)
    assert faiss_store._doc_store_by_faiss_idx == {}
    assert faiss_store._chunk_id_to_faiss_idx == {}
    assert faiss_store._next_faiss_idx == 0


@pytest.mark.asyncio
async def test_faiss_persistence_save_load(faiss_store_fixture: FAISSVectorStore, tmp_path: Path):
    faiss_store = await faiss_store_fixture
    dim = 2
    index_file = tmp_path / "test.faissindex"
    doc_store_file = tmp_path / "test.faissdocs"

    await faiss_store.setup(config={
        "embedding_dim": dim,
        "index_file_path": str(index_file),
        "doc_store_file_path": str(doc_store_file)
    })
    faiss_store._index = mock_faiss_index_instance
    mock_faiss_index_instance.d = dim
    faiss_store._doc_store_by_faiss_idx = {0: SimpleChunk(id="chunk10", content="Content for chunk10", metadata={"idx": 10})}
    faiss_store._chunk_id_to_faiss_idx = {"chunk10": 0}
    faiss_store._next_faiss_idx = 1
    mock_faiss_index_instance.ntotal = 1

    await faiss_store.teardown()

    mock_faiss_module.write_index.assert_called_once_with(mock_faiss_index_instance, str(index_file))

    reset_global_faiss_mocks()
    mock_faiss_index_instance.d = dim
    mock_faiss_index_instance.ntotal = 1
    mock_faiss_module.read_index.return_value = mock_faiss_index_instance

    new_store = FAISSVectorStore()
    original_doc_store_data = {
        "doc_store_by_faiss_idx": {0: SimpleChunk(id="chunk10", content="Content for chunk10", metadata={"idx": 10})},
        "chunk_id_to_faiss_idx": {"chunk10": 0}
    }
    pickled_content = pickle.dumps(original_doc_store_data)

    # Create an AsyncMock for the file object returned by aiofiles.open's context manager
    mock_async_file = AsyncMock()
    mock_async_file.read = AsyncMock(return_value=pickled_content)

    # Create an AsyncMock for the context manager itself
    mock_async_context_manager = AsyncMock()
    mock_async_context_manager.__aenter__.return_value = mock_async_file
    mock_async_context_manager.__aexit__ = AsyncMock(return_value=None)

    # Mock for aiofiles.open function, which returns the context manager
    mock_aiofiles_open_function = MagicMock(return_value=mock_async_context_manager)

    with patch.object(Path, "exists", return_value=True), \
         patch("aiofiles.open", mock_aiofiles_open_function) as mock_aio_open_call_tracker:

        await new_store.setup(config={
            "index_file_path": str(index_file),
            "doc_store_file_path": str(doc_store_file)
        })

    mock_faiss_module.read_index.assert_called_once_with(str(index_file))
    mock_aio_open_call_tracker.assert_called_once_with(doc_store_file, "rb")

    assert new_store._index is mock_faiss_index_instance
    assert new_store._embedding_dim == dim
    assert new_store._index.ntotal == 1 # type: ignore
    assert len(new_store._doc_store_by_faiss_idx) == 1
    retrieved_chunk_from_store = new_store._doc_store_by_faiss_idx[0]
    assert isinstance(retrieved_chunk_from_store, SimpleChunk)
    assert retrieved_chunk_from_store.id == "chunk10"

    new_store._index_file_path = None
    new_store._doc_store_file_path = None
    await new_store.teardown()
    assert new_store._index is None


@pytest.mark.asyncio
async def test_faiss_delete_by_ids_basic_mapping_removal(faiss_store_fixture: FAISSVectorStore, caplog):
    faiss_store = await faiss_store_fixture
    caplog.set_level(logging.WARNING)
    dim = 2
    await faiss_store.setup(config={"embedding_dim": dim})

    faiss_store._index = mock_faiss_index_instance
    mock_faiss_index_instance.d = dim
    faiss_store._doc_store_by_faiss_idx = {
        0: SimpleChunk(id="chunk0", content="c0", metadata={}),
        1: SimpleChunk(id="chunk1", content="c1", metadata={}),
        2: SimpleChunk(id="chunk2", content="c2", metadata={})
    }
    faiss_store._chunk_id_to_faiss_idx = {"chunk0": 0, "chunk1": 1, "chunk2": 2}
    faiss_store._next_faiss_idx = 3
    mock_faiss_index_instance.ntotal = 3
    mock_faiss_index_instance.remove_ids = MagicMock(return_value=2) # Simulate 2 IDs removed by FAISS

    async def mock_run_in_executor_remove_ids(executor, remove_ids_method_mock, np_array_arg):
        # remove_ids_method_mock is mock_faiss_index_instance.remove_ids
        # Call it with its argument
        return_val_from_mock = remove_ids_method_mock(np_array_arg)
        return return_val_from_mock # Propagate the return value (e.g., num removed)

    with patch("asyncio.BaseEventLoop.run_in_executor", side_effect=mock_run_in_executor_remove_ids):
        success = await faiss_store.delete(ids=["chunk0", "chunk2"])

    assert success is True
    assert "chunk0" not in faiss_store._chunk_id_to_faiss_idx
    assert 0 not in faiss_store._doc_store_by_faiss_idx
    assert "chunk2" not in faiss_store._chunk_id_to_faiss_idx
    assert 2 not in faiss_store._doc_store_by_faiss_idx
    assert "chunk1" in faiss_store._chunk_id_to_faiss_idx

    mock_faiss_index_instance.remove_ids.assert_called_once()

    positional_args = mock_faiss_index_instance.remove_ids.call_args.args
    assert len(positional_args) == 1
    ids_array_mock = positional_args[0]

    assert isinstance(ids_array_mock, MagicMock)
    assert ids_array_mock.dtype is mock_numpy_module.int64
    assert sorted(ids_array_mock.tolist()) == sorted([0, 2])


@pytest.mark.asyncio
async def test_faiss_delete_filter_metadata_not_implemented(faiss_store_fixture: FAISSVectorStore, caplog):
    faiss_store = await faiss_store_fixture
    caplog.set_level(logging.WARNING)
    await faiss_store.setup(config={"embedding_dim": 2})
    faiss_store._index = mock_faiss_index_instance
    mock_faiss_index_instance.d = 2

    success = await faiss_store.delete(filter_metadata={"key": "val"})
    assert success is False
    assert "Deletion by metadata filter is not implemented" in caplog.text

@pytest.mark.asyncio
async def test_faiss_libraries_not_installed(caplog):
    caplog.set_level(logging.ERROR)
    with patch("genie_tooling.rag.plugins.impl.vector_stores.faiss_store.faiss", None), \
         patch("genie_tooling.rag.plugins.impl.vector_stores.faiss_store.np", None):

        store_no_libs = FAISSVectorStore()
        await store_no_libs.setup()

        assert "FAISSVectorStore Error: 'faiss-cpu' (or 'faiss-gpu') and 'numpy' libraries not installed." in caplog.text

        result_add = await store_no_libs.add(sample_embeddings_stream_faiss(1))
        assert result_add["added_count"] == 0
        assert len(result_add["errors"]) > 0
        assert "faiss or numpy not available" in result_add["errors"][0].lower()

        result_search = await store_no_libs.search([0.1], 1)
        assert result_search == []

        result_delete = await store_no_libs.delete(ids=["test"])
        assert result_delete is False

@pytest.mark.asyncio
async def test_faiss_add_first_vector_empty(faiss_store_fixture: FAISSVectorStore, caplog):
    faiss_store = await faiss_store_fixture
    caplog.set_level(logging.ERROR) # Keep ERROR to see if other errors occur
    await faiss_store.setup()

    async def empty_first_vec_stream():
        yield SimpleChunk("c_empty", "empty_vec_content", {}), []
        yield SimpleChunk("c_good", "good_vec_content", {}), [0.1,0.2]

    async def mock_run_in_executor_first_empty(executor, func_partial):
        num_added = func_partial()
        return num_added

    with patch("asyncio.BaseEventLoop.run_in_executor", side_effect=mock_run_in_executor_first_empty):
        result = await faiss_store.add(empty_first_vec_stream())

    assert "Cannot infer embedding dimension from empty first vector." in result["errors"][0]
    assert result["added_count"] == 1


@pytest.mark.asyncio
async def test_faiss_add_fails_index_init(faiss_store_fixture: FAISSVectorStore, caplog):
    faiss_store = await faiss_store_fixture
    caplog.set_level(logging.ERROR)
    await faiss_store.setup()

    mock_faiss_module.IndexFlatL2.side_effect = RuntimeError("FAISS init failed")

    result = await faiss_store.add(sample_embeddings_stream_faiss(1, dim=3))
    assert len(result["errors"]) > 0
    assert "Failed to initialize FAISS index with dim 3" in result["errors"][0]
    assert result["added_count"] == 0


@pytest.mark.asyncio
async def test_faiss_add_batch_sync_add_fails(faiss_store_fixture: FAISSVectorStore, caplog):
    faiss_store = await faiss_store_fixture
    caplog.set_level(logging.ERROR)
    await faiss_store.setup(config={"embedding_dim": 3})

    mock_faiss_index_instance.add.side_effect = RuntimeError("FAISS add error")

    stream = sample_embeddings_stream_faiss(1, dim=3)
    async def mock_run_executor_add_fail(executor, func_partial):
        try:
            return func_partial()
        except RuntimeError:
             return 0

    with patch("asyncio.BaseEventLoop.run_in_executor", side_effect=mock_run_executor_add_fail):
        result = await faiss_store.add(stream)

    assert result["added_count"] == 0
    assert any("Error in _sync_add_batch: FAISS add error" in rec.message for rec in caplog.records)
    assert len(result["errors"]) == 0


@pytest.mark.asyncio
async def test_faiss_search_no_results_from_faiss(faiss_store_fixture: FAISSVectorStore):
    faiss_store = await faiss_store_fixture
    dim=3
    await faiss_store.setup(config={"embedding_dim": dim})
    faiss_store._index = mock_faiss_index_instance
    mock_faiss_index_instance.ntotal = 1

    empty_distances = robust_np_array_mock_faiss([[]])
    empty_indices = robust_np_array_mock_faiss([[]])
    empty_indices.size = 0
    empty_distances.size = 0
    mock_faiss_index_instance.search.return_value = (empty_distances, empty_indices)

    results = await faiss_store.search([0.1]*dim, top_k=1)
    assert results == []


@pytest.mark.asyncio
async def test_faiss_search_faiss_returns_minus_one_index(faiss_store_fixture: FAISSVectorStore):
    faiss_store = await faiss_store_fixture
    dim=3
    await faiss_store.setup(config={"embedding_dim": dim})
    faiss_store._index = mock_faiss_index_instance
    mock_faiss_index_instance.ntotal = 1

    mock_distances_np = robust_np_array_mock_faiss([[0.123]])
    mock_indices_np = robust_np_array_mock_faiss([[-1]])
    mock_faiss_index_instance.search.return_value = (mock_distances_np, mock_indices_np)

    results = await faiss_store.search([0.1]*dim, top_k=1)
    assert results == []
