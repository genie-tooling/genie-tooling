import asyncio
import logging
import pickle
import uuid
from pathlib import Path
from typing import Any, AsyncIterable, Dict, List, Optional, Tuple, cast

import aiofiles

logger = logging.getLogger(__name__)

try:
    import faiss
    import numpy as np
except ImportError:
    faiss = None
    np = None
    logger.warning("FAISSVectorStore: 'faiss-cpu' or 'numpy' not installed. This plugin will not be functional.")

from genie_tooling.core.types import Chunk, EmbeddingVector, RetrievedChunk
from genie_tooling.vector_stores.abc import VectorStorePlugin


class _RetrievedChunkImpl(RetrievedChunk, Chunk):
    def __init__(self, content: str, metadata: Dict[str, Any], score: float, id: Optional[str] = None, rank: Optional[int] = None):
        self.content: str = content; self.metadata: Dict[str, Any] = metadata
        self.id: Optional[str] = id; self.score: float = score; self.rank: Optional[int] = rank

class FAISSVectorStore(VectorStorePlugin):
    plugin_id: str = "faiss_vector_store_v1"
    description: str = "In-memory vector store using FAISS, with optional persistence to disk."

    _index: Optional[Any] = None
    _doc_store_by_faiss_idx: Dict[int, Chunk] = {}
    _chunk_id_to_faiss_idx: Dict[str, int] = {}
    _next_faiss_idx: int = 0
    _embedding_dim: Optional[int] = None
    _index_file_path: Optional[Path] = None
    _doc_store_file_path: Optional[Path] = None
    _faiss_index_factory: str = "Flat"
    _lock: asyncio.Lock

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._lock = asyncio.Lock()
        if not faiss or not np:
            logger.error(f"{self.plugin_id} Error: 'faiss-cpu' or 'numpy' not installed.")
            return

        cfg = config or {}
        self._embedding_dim = cfg.get("embedding_dim")
        self._faiss_index_factory = cfg.get("faiss_index_factory_string", self._faiss_index_factory)
        collection_name = cfg.get("collection_name", "default_faiss_collection")
        persist = cfg.get("persist_by_default", True)

        index_fp_str = cfg.get("index_file_path")
        doc_store_fp_str = cfg.get("doc_store_file_path")
        default_base_path = Path("./.genie_data/faiss")

        if index_fp_str is not None: self._index_file_path = Path(index_fp_str) if index_fp_str else None
        elif persist: self._index_file_path = default_base_path / f"{collection_name}.faissindex"
        else: self._index_file_path = None

        if doc_store_fp_str is not None: self._doc_store_file_path = Path(doc_store_fp_str) if doc_store_fp_str else None
        elif persist and self._index_file_path: self._doc_store_file_path = default_base_path / f"{collection_name}.faissdocs"
        else: self._doc_store_file_path = None

        if self._index_file_path: logger.info(f"{self.plugin_id}: Index path set to '{self._index_file_path}'.")
        if self._doc_store_file_path: logger.info(f"{self.plugin_id}: Doc store path set to '{self._doc_store_file_path}'.")

        loaded_from_file = False
        if self._index_file_path and self._doc_store_file_path:
            self._index_file_path.parent.mkdir(parents=True, exist_ok=True)
            if self._index_file_path.exists() and self._doc_store_file_path.exists():
                # P1 VERIFICATION POINT: Add more specific error handling for _load_from_files
                await self._load_from_files()
                loaded_from_file = bool(self._index)
            else: logger.info(f"{self.plugin_id}: Index/doc files not found at configured paths.")

        if not loaded_from_file and self._embedding_dim:
            self._initialize_faiss_index(self._embedding_dim)
        elif not loaded_from_file:
            logger.info(f"{self.plugin_id}: Ready. Index will be init on first data add if dim known.")
        logger.debug(f"{self.plugin_id} setup done. Index items: {self._index.ntotal if self._index else 0}")

    def _initialize_faiss_index(self, dimension: int):
        if not faiss: return
        try:
            if "IDMap" not in self._faiss_index_factory.upper():
                base_index_factory = self._faiss_index_factory
                if base_index_factory.upper() == "FLAT" or base_index_factory.upper() == "INDEXFLATL2":
                    base_index = faiss.IndexFlatL2(dimension)
                else:
                    base_index = faiss.index_factory(dimension, base_index_factory)
                self._index = faiss.IndexIDMap(base_index)
                logger.info(f"{self.plugin_id}: Initialized FAISS IndexIDMap wrapping {base_index_factory} with dim {dimension}.")
            else:
                 self._index = faiss.index_factory(dimension, self._faiss_index_factory)
                 logger.info(f"{self.plugin_id}: Initialized FAISS index with factory '{self._faiss_index_factory}' and dim {dimension}.")

            self._embedding_dim = dimension
        except Exception as e:
            logger.error(f"{self.plugin_id}: Failed to initialize FAISS index (dim {dimension}, factory '{self._faiss_index_factory}'): {e}", exc_info=True)
            self._index = None


    async def _load_from_files(self) -> None:
        if not self._index_file_path or not self._doc_store_file_path or not faiss: return
        loop = asyncio.get_running_loop()
        async with self._lock:
            try:
                logger.info(f"Attempting to load FAISS index from: {self._index_file_path}")
                self._index = await loop.run_in_executor(None, faiss.read_index, str(self._index_file_path))
                if self._index:
                    self._embedding_dim = self._index.d
                    self._next_faiss_idx = self._index.ntotal
                    logger.info(f"FAISS index loaded. Dim: {self._embedding_dim}, NTotal: {self._index.ntotal}")

                logger.info(f"Attempting to load doc store from: {self._doc_store_file_path}")
                async with aiofiles.open(self._doc_store_file_path, "rb") as f:
                    pickled_data = await f.read()
                loaded_stores = pickle.loads(pickled_data)
                self._doc_store_by_faiss_idx = loaded_stores.get("doc_store_by_faiss_idx", {})
                self._chunk_id_to_faiss_idx = loaded_stores.get("chunk_id_to_faiss_idx", {})
                if self._doc_store_by_faiss_idx:
                    self._next_faiss_idx = max(self._doc_store_by_faiss_idx.keys(), default=-1) + 1
                else:
                    self._next_faiss_idx = 0

                logger.info(f"Doc store loaded. {len(self._doc_store_by_faiss_idx)} items. Next FAISS ID: {self._next_faiss_idx}")

                if self._index and self._index.ntotal != len(self._doc_store_by_faiss_idx):
                    logger.warning(f"FAISS index ntotal ({self._index.ntotal}) and doc store size ({len(self._doc_store_by_faiss_idx)}) mismatch after load. This might indicate issues if using IndexIDMap with non-contiguous IDs or if persistence was interrupted.")

            except FileNotFoundError:
                logger.info(f"{self.plugin_id}: Index or docstore file not found. Will create new if paths are set.")
                self._index = None
            except pickle.UnpicklingError as e_pickle:
                logger.error(f"Error unpickling doc store from {self._doc_store_file_path}: {e_pickle}. Data may be corrupt.", exc_info=True)
                self._index = None # Consider invalidating index if doc store is corrupt
            except Exception as e: # Catch other FAISS or I/O errors
                logger.error(f"Error loading FAISS from files: {e}", exc_info=True)
                self._index = None

    async def _save_to_files(self) -> None:
        if not self._index_file_path or not self._doc_store_file_path or not self._index or not faiss:
            if self._index_file_path: logger.debug("FAISS save skipped (paths/index not fully available).")
            return

        self._index_file_path.parent.mkdir(parents=True, exist_ok=True)
        self._doc_store_file_path.parent.mkdir(parents=True, exist_ok=True)

        loop = asyncio.get_running_loop()
        async with self._lock:
            try:
                logger.info(f"Saving FAISS index ({self._index.ntotal} items) to: {self._index_file_path}")
                await loop.run_in_executor(None, faiss.write_index, self._index, str(self._index_file_path))

                data_to_pickle = {
                    "doc_store_by_faiss_idx": self._doc_store_by_faiss_idx,
                    "chunk_id_to_faiss_idx": self._chunk_id_to_faiss_idx
                }
                pickled_data = pickle.dumps(data_to_pickle)
                logger.info(f"Saving doc store ({len(self._doc_store_by_faiss_idx)} items) to: {self._doc_store_file_path}")
                async with aiofiles.open(self._doc_store_file_path, "wb") as f:
                    await f.write(pickled_data)
                logger.info("FAISS index and doc store saved successfully.")
            except Exception as e:
                logger.error(f"Error saving FAISS to files: {e}", exc_info=True)

    async def add(self, embeddings: AsyncIterable[Tuple[Chunk, EmbeddingVector]], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not faiss or not np: return {"added_count": 0, "errors": ["FAISS or NumPy not available."]}

        cfg = config or {}; batch_size = int(cfg.get("batch_size", 64))
        current_batch_chunks: List[Chunk] = []
        current_batch_vectors_np: List[np.ndarray] = []
        added_count_total = 0
        errors_list: List[str] = []

        async with self._lock:
            first_vector_processed = False
            async for chunk, vec_list in embeddings:
                if not first_vector_processed:
                    if not self._embedding_dim and vec_list: self._embedding_dim = len(vec_list)
                    if not self._index and self._embedding_dim: self._initialize_faiss_index(self._embedding_dim)
                    first_vector_processed = True

                if not self._index:
                    err_msg = "FAISS index not initialized (likely missing embedding_dim or failed init)."
                    if err_msg not in errors_list: errors_list.append(err_msg)
                    continue

                if not vec_list or len(vec_list) != self._embedding_dim:
                    errors_list.append(f"Dimension mismatch or empty vector for chunk '{chunk.id}'. Expected {self._embedding_dim}, got {len(vec_list) if vec_list else 'empty'}.")
                    continue

                current_batch_chunks.append(chunk)
                current_batch_vectors_np.append(np.array(vec_list, dtype=np.float32).reshape(1, -1))

                if len(current_batch_chunks) >= batch_size:
                    added_in_batch = await self._add_batch_to_faiss_and_docstore(current_batch_chunks, current_batch_vectors_np)
                    added_count_total += added_in_batch
                    current_batch_chunks, current_batch_vectors_np = [], []

            if current_batch_chunks:
                added_in_batch = await self._add_batch_to_faiss_and_docstore(current_batch_chunks, current_batch_vectors_np)
                added_count_total += added_in_batch

        if self._index_file_path: await self._save_to_files()
        return {"added_count": added_count_total, "errors": errors_list}

    async def _add_batch_to_faiss_and_docstore(self, chunks: List[Chunk], vectors_np_list: List[Any]) -> int:
        if not self._index or not chunks or not vectors_np_list or not np: return 0

        num_to_add = len(chunks)
        faiss_ids_for_batch = np.array(range(self._next_faiss_idx, self._next_faiss_idx + num_to_add), dtype=np.int64)

        def _sync_add_batch():
            if not vectors_np_list: return 0
            try:
                concatenated_vectors = np.concatenate(vectors_np_list, axis=0)
                self._index.add_with_ids(concatenated_vectors, faiss_ids_for_batch)

                count = 0
                for i, chunk_item in enumerate(chunks):
                    current_faiss_id = int(faiss_ids_for_batch[i])
                    self._doc_store_by_faiss_idx[current_faiss_id] = chunk_item

                    original_chunk_id = chunk_item.id or str(uuid.uuid4())
                    if chunk_item.id is None: chunk_item.id = original_chunk_id

                    self._chunk_id_to_faiss_idx[original_chunk_id] = current_faiss_id
                    count += 1

                self._next_faiss_idx += num_to_add
                return count
            except Exception as e:
                logger.error(f"Error in _sync_add_batch (FAISS add_with_ids or mapping): {e}", exc_info=True)
                return 0

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_add_batch)

    async def search(self, query_embedding: EmbeddingVector, top_k: int, filter_metadata: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None) -> List[RetrievedChunk]:
        if not self._index or not self._embedding_dim or not np or self._index.ntotal == 0: return []
        if len(query_embedding) != self._embedding_dim:
            logger.warning(f"Query embedding dim {len(query_embedding)} != index dim {self._embedding_dim}.")
            return []

        query_vec_np = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        actual_k = min(top_k, self._index.ntotal)
        if actual_k == 0: return []

        def _sync_search_and_filter():
            try:
                distances, faiss_indices = self._index.search(query_vec_np, actual_k)
            except Exception as e_search_faiss:
                logger.error(f"FAISS search operation error: {e_search_faiss}", exc_info=True)
                return []

            retrieved_chunks_list: List[RetrievedChunk] = []
            if hasattr(faiss_indices, "size") and faiss_indices.size > 0:
                for i in range(faiss_indices.shape[1]):
                    faiss_idx = int(faiss_indices[0, i])
                    if faiss_idx == -1: continue

                    original_chunk = self._doc_store_by_faiss_idx.get(faiss_idx)
                    if original_chunk:
                        if filter_metadata:
                            match = all(original_chunk.metadata.get(k) == v for k, v in filter_metadata.items())
                            if not match: continue

                        score = float(1.0 / (1.0 + distances[0, i])) if distances[0,i] >= 0 else 0.0
                        score = max(0.0, min(1.0, score))

                        retrieved_chunks_list.append(cast(RetrievedChunk, _RetrievedChunkImpl(
                            content=original_chunk.content, metadata=original_chunk.metadata,
                            score=score, id=original_chunk.id, rank=len(retrieved_chunks_list) + 1
                        )))
            return retrieved_chunks_list

        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, _sync_search_and_filter)

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        delete_all: bool = False,
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        if not self._index or not faiss or not np: return False

        removed_count = 0
        async with self._lock:
            if delete_all:
                logger.info(f"{self.plugin_id}: Deleting all data from FAISS store.")
                if self._embedding_dim: self._initialize_faiss_index(self._embedding_dim)
                else: self._index = None
                self._doc_store_by_faiss_idx.clear()
                self._chunk_id_to_faiss_idx.clear()
                self._next_faiss_idx = 0
                removed_count = -1
            elif ids:
                faiss_indices_to_remove_np = np.array(
                    [self._chunk_id_to_faiss_idx[chunk_id] for chunk_id in ids if chunk_id in self._chunk_id_to_faiss_idx],
                    dtype=np.int64
                )
                if faiss_indices_to_remove_np.size > 0:
                    try:
                        loop = asyncio.get_running_loop()
                        num_actually_removed_from_faiss = await loop.run_in_executor(None, self._index.remove_ids, faiss_indices_to_remove_np)
                        logger.debug(f"FAISS remove_ids call removed {num_actually_removed_from_faiss} items from index.")
                        removed_count = num_actually_removed_from_faiss

                        for chunk_id in ids:
                            if chunk_id in self._chunk_id_to_faiss_idx:
                                faiss_idx_to_clear = self._chunk_id_to_faiss_idx.pop(chunk_id)
                                self._doc_store_by_faiss_idx.pop(faiss_idx_to_clear, None)
                        logger.info(f"Removed {removed_count} items by ID from FAISS and doc store.")
                    except Exception as e_remove:
                        logger.error(f"Error during FAISS remove_ids or doc store cleanup: {e_remove}", exc_info=True)
                        return False
                else:
                    logger.info(f"{self.plugin_id}: No matching FAISS indices found for provided IDs to delete.")
            elif filter_metadata:
                # P1 VERIFICATION: Add prominent log warning about performance of metadata-based deletion.
                logger.warning(
                    f"{self.plugin_id}: Delete by metadata filter is NOT performant for FAISS. "
                    "This involves iterating all stored documents in Python. For large stores, "
                    "filter externally and delete by IDs, or rebuild the index for true metadata filtering."
                )
                ids_to_delete_by_filter: List[str] = []
                num_scanned_for_filter = 0
                for faiss_idx, chunk_obj in list(self._doc_store_by_faiss_idx.items()):
                    num_scanned_for_filter += 1
                    if chunk_obj.id and all(chunk_obj.metadata.get(k) == v for k, v in filter_metadata.items()):
                        ids_to_delete_by_filter.append(chunk_obj.id)

                logger.info(f"Scanned {num_scanned_for_filter} documents for metadata filter. Found {len(ids_to_delete_by_filter)} IDs to delete.")
                if ids_to_delete_by_filter:
                    return await self.delete(ids=ids_to_delete_by_filter)
                else:
                    logger.info("No items matched metadata filter for deletion.")
                    return True
            else:
                logger.warning(f"{self.plugin_id}: Delete called without specific IDs or delete_all=True. No action taken.")
                return False

        if self._index_file_path and (removed_count > 0 or removed_count == -1):
            await self._save_to_files()

        return True if (removed_count > 0 or removed_count == -1) else False


    async def teardown(self) -> None:
        await self._save_to_files()
        self._index = None
        self._doc_store_by_faiss_idx.clear()
        self._chunk_id_to_faiss_idx.clear()
        logger.info(f"{self.plugin_id}: Torn down. Index and doc store cleared from memory.")
