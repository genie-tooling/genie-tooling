import asyncio
import logging
from pathlib import Path
from typing import Any, AsyncIterable, Dict, Tuple
from unittest.mock import MagicMock, patch

import pytest
from genie_tooling.core.types import Chunk, EmbeddingVector
from genie_tooling.vector_stores.impl.faiss_store import (
    FAISSVectorStore,  # Ensure _RetrievedChunkImpl is imported if used
    )

# Mock faiss and numpy at the module level for all tests in this file
mock_faiss_index_instance = MagicMock(name="GlobalMockFaissIndexInstance")
mock_faiss_module = MagicMock(name="MockFaissModule")
mock_numpy_module = MagicMock(name="MockNumpyModule")
mock_numpy_module.linalg = MagicMock(name="MockNumpyLinalgModule")

def reset_global_faiss_mocks():
    global mock_faiss_index_instance, mock_faiss_module, mock_numpy_module
    mock_faiss_index_instance.reset_mock(); mock_faiss_index_instance.d = 0; mock_faiss_index_instance.ntotal = 0
    mock_faiss_index_instance.add_with_ids = MagicMock() # Changed from .add to .add_with_ids
    empty_dist_arr = robust_np_array_mock_faiss([[]]); empty_idx_arr = robust_np_array_mock_faiss([[]])
    if hasattr(empty_idx_arr, "size"): empty_idx_arr.size = 0
    mock_faiss_index_instance.search = MagicMock(return_value=(empty_dist_arr, empty_idx_arr))
    mock_faiss_index_instance.reset = MagicMock(); mock_faiss_index_instance.remove_ids = MagicMock(return_value=0)

    mock_faiss_module.reset_mock()
    # IndexFlatL2 is wrapped by IndexIDMap by default now in the implementation
    # So, index_factory or IndexIDMap(IndexFlatL2(...)) will be called.
    # Let's make IndexFlatL2 return a mock that can be wrapped, and IndexIDMap return the main mock.
    mock_flat_l2_sub_index = MagicMock(name="MockFlatL2SubIndex")
    mock_faiss_module.IndexFlatL2 = MagicMock(return_value=mock_flat_l2_sub_index)
    mock_faiss_module.IndexIDMap = MagicMock(return_value=mock_faiss_index_instance) # This is what's stored in self._index
    mock_faiss_module.index_factory = MagicMock(return_value=mock_faiss_index_instance) # If user provides "IDMap,..."

    mock_faiss_module.read_index = MagicMock(return_value=mock_faiss_index_instance)
    mock_faiss_module.write_index = MagicMock()

    mock_numpy_module.reset_mock(); mock_numpy_module.array = MagicMock(side_effect=robust_np_array_mock_faiss)
    mock_numpy_module.concatenate = MagicMock(side_effect=lambda arr_list, axis: robust_np_array_mock_faiss([item for sublist in [a.tolist() for a in arr_list] for item in (sublist if isinstance(sublist[0], list) else [sublist])]))
    mock_numpy_module.float32 = type("float32", (), {}); mock_numpy_module.linalg.norm = MagicMock(return_value=1.0)
    mock_numpy_module.int64 = type("int64", (), {})

def robust_np_array_mock_faiss(data, dtype=None):
    # (Keep robust_np_array_mock_faiss as it was)
    arr_instance = MagicMock(name=f"MockNpArray_{str(data)[:10]}"); _data = list(data) if not isinstance(data,list) else data
    arr_instance.tolist = lambda: _data; arr_instance.dtype = dtype or mock_numpy_module.float32
    if not _data or (isinstance(_data,list) and len(_data)==1 and isinstance(_data[0],list) and not _data[0]):
        arr_instance.ndim=1 if not _data or not isinstance(_data[0],list) else 2; arr_instance.shape=(0,) if arr_instance.ndim==1 and not _data else (1,0) if arr_instance.ndim==2 and len(_data)==1 and not _data[0] else (0,0); arr_instance.size=0
    else:
        first_el = _data[0]; arr_instance.ndim = 2 if isinstance(first_el, list) else 1
        arr_instance.shape = (len(_data), len(first_el) if first_el else 0) if arr_instance.ndim == 2 else (len(_data),)
        arr_instance.size = arr_instance.shape[0] * (arr_instance.shape[1] if arr_instance.ndim == 2 and arr_instance.shape[1] > 0 else 1 if arr_instance.ndim == 1 else 0 )
    arr_instance.reshape = MagicMock(side_effect=lambda new_shape, order=None: robust_np_array_mock_faiss([arr_instance.tolist()] if new_shape==(1,-1) and arr_instance.ndim==1 else _data, dtype=arr_instance.dtype))
    return arr_instance


@pytest.fixture
async def faiss_store_fixture(request) -> FAISSVectorStore:
    store = FAISSVectorStore()
    async def finalizer_async():
        if hasattr(store, "_index") and store._index: await store.teardown()
        elif hasattr(store, "_index_file_path") and store._index_file_path: await store.teardown()
    request.addfinalizer(lambda: asyncio.run(finalizer_async()))
    return store

@pytest.fixture(autouse=True)
def mock_faiss_dependencies_fixt():
    reset_global_faiss_mocks()
    with patch("genie_tooling.vector_stores.impl.faiss_store.faiss", mock_faiss_module), \
         patch("genie_tooling.vector_stores.impl.faiss_store.np", mock_numpy_module):
        yield

class SimpleChunk(Chunk):
    def __init__(self, id: str, content: str, metadata: Dict[str,Any]):
        self.id=id; self.content=content; self.metadata=metadata
async def sample_embeddings_stream_faiss(count=2,dim=3,start_id=0) -> AsyncIterable[Tuple[Chunk,EmbeddingVector]]:
    for i in range(count): yield SimpleChunk(f"c{start_id+i}",f"Ct{start_id+i}",{"idx":start_id+i}),[float(start_id+i+j*0.1) for j in range(dim)]

@pytest.mark.asyncio
async def test_faiss_setup_default_paths(faiss_store_fixture: FAISSVectorStore, caplog):
    store = await faiss_store_fixture
    caplog.set_level(logging.INFO)
    collection_name = "my_faiss_test_collection"
    with patch.object(Path, "mkdir") as mock_mkdir, \
         patch.object(Path, "exists", return_value=False): # Simulate files not existing
        await store.setup(config={
            "collection_name": collection_name, "embedding_dim": 3, "persist_by_default": True
        })
    expected_idx_path = Path(f"./.genie_data/faiss/{collection_name}.faissindex")
    expected_docs_path = Path(f"./.genie_data/faiss/{collection_name}.faissdocs")

    assert store._index_file_path == expected_idx_path
    assert store._doc_store_file_path == expected_docs_path
    mock_mkdir.assert_any_call(parents=True, exist_ok=True) # For parent of index/doc files

    # Check for the specific log messages indicating default path usage
    assert f"Index path set to '{str(expected_idx_path)}'." in caplog.text
    assert f"Doc store path set to '{str(expected_docs_path)}'." in caplog.text
    # Check for the log indicating files were not found (since exists is False)
    assert "Index/doc files not found at configured paths." in caplog.text
    # Check for the log indicating new index initialization
    assert "Initialized FAISS IndexIDMap wrapping Flat with dim 3." in caplog.text


@pytest.mark.asyncio
async def test_faiss_add_infer_dim_and_initialize(faiss_store_fixture: FAISSVectorStore):
    faiss_store = await faiss_store_fixture
    await faiss_store.setup(config={"persist_by_default": False}) # No initial dim
    assert faiss_store._index is None
    dim_to_infer = 5
    mock_faiss_index_instance.add_with_ids.return_value = None # Mock the method used

    async def mock_run_in_executor(executor, func_partial): return func_partial()
    with patch("asyncio.BaseEventLoop.run_in_executor", side_effect=mock_run_in_executor):
        result = await faiss_store.add(sample_embeddings_stream_faiss(1, dim=dim_to_infer))

    assert result["added_count"] == 1
    assert faiss_store._index is mock_faiss_index_instance # Check if the main mock instance is set
    assert faiss_store._embedding_dim == dim_to_infer

    # Check that IndexFlatL2 was called to create the base index, then IndexIDMap to wrap it.
    mock_faiss_module.IndexFlatL2.assert_called_once_with(dim_to_infer)
    mock_faiss_module.IndexIDMap.assert_called_once_with(mock_faiss_module.IndexFlatL2.return_value)
    mock_faiss_index_instance.add_with_ids.assert_called_once() # Check the correct add method
