"""FAISSVectorStore: In-memory or file-backed vector store using FAISS."""
import asyncio
import logging
import pickle  # For saving/loading doc_store
import uuid  # For generating default chunk IDs
from pathlib import Path
from typing import Any, AsyncIterable, Dict, List, Optional, Tuple, cast

import aiofiles

logger = logging.getLogger(__name__)

# Attempt to import FAISS and NumPy, make them optional
try:
    import faiss  # type: ignore
    import numpy as np  # type: ignore
except ImportError:
    faiss = None # type: ignore
    np = None # type: ignore

from genie_tooling.core.types import Chunk, EmbeddingVector, RetrievedChunk
from genie_tooling.rag.plugins.abc import VectorStorePlugin


# Concrete RetrievedChunk for internal use by this store
class _RetrievedChunkImpl(RetrievedChunk, Chunk):
    def __init__(self, content: str, metadata: Dict[str, Any], score: float, id: Optional[str] = None, rank: Optional[int] = None):
        self.content: str = content
        self.metadata: Dict[str, Any] = metadata
        self.id: Optional[str] = id
        self.score: float = score
        self.rank: Optional[int] = rank

class FAISSVectorStore(VectorStorePlugin):
    plugin_id: str = "faiss_vector_store_v1"
    description: str = "Vector store using FAISS for similarity search. Supports in-memory and basic file persistence."

    _index: Optional[Any] = None  # faiss.Index
    # _doc_store: Maps FAISS index (int) to original Chunk object
    _doc_store_by_faiss_idx: Dict[int, Chunk] = {}
    # _id_to_faiss_idx: Maps custom chunk ID (str) to FAISS index (int) for deletion/updates
    _chunk_id_to_faiss_idx: Dict[str, int] = {}
    _next_faiss_idx: int = 0 # Counter for next available FAISS index

    _embedding_dim: Optional[int] = None
    _index_file_path: Optional[Path] = None # Path to save/load FAISS index
    _doc_store_file_path: Optional[Path] = None # Path to save/load doc_store dict

    _lock: asyncio.Lock # To protect access to FAISS index and doc_store

    # No __init__ for PluginManager. Config via setup.

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the FAISS index and document store.
        Config options:
            "embedding_dim": Optional[int] - Dimension of embeddings. If not provided and no index loaded,
                                             it will be inferred from the first batch of embeddings added.
            "index_file_path": Optional[str] - Path to load/save the FAISS index file.
            "doc_store_file_path": Optional[str] - Path to load/save the document store (pickled dict).
            "faiss_index_factory_string": str (default: "Flat") - FAISS index factory string, e.g., "Flat", "IVF256,Flat".
                                            "Flat" or "IndexFlatL2" are common for exact search.
        """
        self._lock = asyncio.Lock()
        if not faiss or not np:
            logger.error("FAISSVectorStore Error: 'faiss-cpu' (or 'faiss-gpu') and 'numpy' libraries not installed. "
                         "Please install them: poetry install --extras vectorstores")
            return

        cfg = config or {}
        self._embedding_dim = cfg.get("embedding_dim")
        self._faiss_index_factory = cfg.get("faiss_index_factory_string", "Flat") # "IndexFlatL2" is explicit for L2

        index_fp_str = cfg.get("index_file_path")
        doc_store_fp_str = cfg.get("doc_store_file_path")

        loaded_from_file = False
        if index_fp_str:
            self._index_file_path = Path(index_fp_str)
            if doc_store_fp_str:
                 self._doc_store_file_path = Path(doc_store_fp_str)
                 if self._index_file_path.exists() and self._doc_store_file_path.exists():
                    logger.info(f"FAISSVectorStore: Attempting to load index from '{self._index_file_path}' and doc store from '{self._doc_store_file_path}'.")
                    await self._load_from_files()
                    loaded_from_file = True # If _load_from_files succeeds and sets _index
                 else:
                    logger.info("FAISSVectorStore: Index or doc store file not found. Will create new store at specified paths if data is added.")
            else:
                logger.warning("FAISSVectorStore: 'index_file_path' provided but 'doc_store_file_path' is missing. Cannot reliably load/save without both.")
                self._index_file_path = None # Disable file persistence if incomplete

        if not loaded_from_file and self._embedding_dim:
            # If not loaded from file but embedding_dim is known, initialize empty index
            try:
                logger.info(f"FAISSVectorStore: Initializing new FAISS index with factory '{self._faiss_index_factory}' and dimension {self._embedding_dim}.")
                # Using IndexFlatL2 directly for clarity if factory is "Flat" or similar
                if self._faiss_index_factory.upper() == "FLAT" or self._faiss_index_factory.upper() == "INDEXFLATL2":
                    self._index = faiss.IndexFlatL2(self._embedding_dim)
                else: # For more complex factory strings
                    self._index = faiss.index_factory(self._embedding_dim, self._faiss_index_factory)
            except Exception as e:
                logger.error(f"FAISSVectorStore: Failed to initialize FAISS index with dim {self._embedding_dim}: {e}", exc_info=True)
                self._index = None # Ensure index is None on failure
        elif not loaded_from_file:
            logger.info("FAISSVectorStore: Ready. Index will be initialized when the first batch of embeddings (with known dimension) is added.")

        logger.debug(f"FAISSVectorStore setup complete. Index populated: {self._index is not None and self._index.ntotal > 0 if self._index else False}")


    async def _load_from_files(self) -> None:
        """Helper to load index and doc_store from files. Run under lock."""
        if not self._index_file_path or not self._doc_store_file_path or not faiss: return
        loop = asyncio.get_running_loop()
        async with self._lock:
            try:
                self._index = await loop.run_in_executor(None, faiss.read_index, str(self._index_file_path))
                if self._index:
                    self._embedding_dim = self._index.d # Get dimension from loaded index
                    self._next_faiss_idx = self._index.ntotal # Current number of vectors

                async with aiofiles.open(self._doc_store_file_path, "rb") as f:
                    pickled_data = await f.read()
                loaded_stores = pickle.loads(pickled_data)
                self._doc_store_by_faiss_idx = loaded_stores.get("doc_store_by_faiss_idx", {})
                self._chunk_id_to_faiss_idx = loaded_stores.get("chunk_id_to_faiss_idx", {})
                # Ensure _next_faiss_idx is consistent if index was empty but doc_store wasn't or vice-versa
                # This simple load assumes consistency. A robust solution needs more checks.
                if self._index and self._index.ntotal != len(self._doc_store_by_faiss_idx):
                    logger.warning("FAISSVectorStore: Loaded index size and doc store size mismatch. Data may be inconsistent.")

                logger.info(f"FAISSVectorStore: Loaded {self._index.ntotal if self._index else 0} vectors and "
                            f"{len(self._doc_store_by_faiss_idx)} documents from files.")
            except Exception as e:
                logger.error(f"FAISSVectorStore: Error loading from files: {e}", exc_info=True)
                # Reset to empty state on load failure to avoid partial inconsistent state
                self._index = None
                self._doc_store_by_faiss_idx.clear()
                self._chunk_id_to_faiss_idx.clear()
                self._next_faiss_idx = 0
                # Keep _embedding_dim if it was set by config, otherwise it's None

    async def _save_to_files(self) -> None:
        """Helper to save index and doc_store to files. Run under lock."""
        if not self._index_file_path or not self._doc_store_file_path or not self._index or not faiss:
            if self._index_file_path: # Only log if save was intended
                 logger.debug("FAISSVectorStore: Save to file skipped (paths or index not configured/available).")
            return

        loop = asyncio.get_running_loop()
        async with self._lock:
            try:
                logger.debug(f"FAISSVectorStore: Saving index with {self._index.ntotal} vectors to '{self._index_file_path}'.")
                await loop.run_in_executor(None, faiss.write_index, self._index, str(self._index_file_path))

                data_to_pickle = {
                    "doc_store_by_faiss_idx": self._doc_store_by_faiss_idx,
                    "chunk_id_to_faiss_idx": self._chunk_id_to_faiss_idx,
                    # "embedding_dim": self._embedding_dim, # Not strictly needed if index is saved
                    # "next_faiss_idx": self._next_faiss_idx # Can be inferred from index.ntotal on load
                }
                pickled_data = pickle.dumps(data_to_pickle)
                async with aiofiles.open(self._doc_store_file_path, "wb") as f:
                    await f.write(pickled_data)
                logger.info("FAISSVectorStore: Saved index and doc store to files.")
            except Exception as e:
                logger.error(f"FAISSVectorStore: Error saving to files: {e}", exc_info=True)


    async def add(self, embeddings: AsyncIterable[Tuple[Chunk, EmbeddingVector]], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Adds chunks and their embeddings to the FAISS index and document store."""
        if not faiss or not np:
            msg = "FAISS or NumPy not available."
            logger.error(f"FAISSVectorStore Add: {msg}")
            return {"added_count": 0, "errors": [msg]}

        cfg = config or {}
        batch_size = int(cfg.get("batch_size", 64))

        current_batch_chunks: List[Chunk] = []
        current_batch_vectors_np: List[Any] = [] # List of np.ndarray
        added_count = 0
        errors_list: List[str] = []

        async with self._lock: # Ensure exclusive access for a full add operation cycle if possible
                             # Or lock only for the _add_batch_sync part.
                             # For simplicity, lock around the loop.
            async for chunk, vector_list in embeddings:
                if not self._embedding_dim: # Infer dimension from first valid vector
                    if vector_list:
                        self._embedding_dim = len(vector_list)
                        if not self._index: # Initialize index if not loaded and dim is now known
                            try:
                                logger.info(f"FAISSVectorStore: Initializing new FAISS index with factory '{self._faiss_index_factory}' and inferred dimension {self._embedding_dim}.")
                                if self._faiss_index_factory.upper() == "FLAT" or self._faiss_index_factory.upper() == "INDEXFLATL2":
                                    self._index = faiss.IndexFlatL2(self._embedding_dim)
                                else:
                                    self._index = faiss.index_factory(self._embedding_dim, self._faiss_index_factory)
                            except Exception as e_init:
                                msg = f"Failed to initialize FAISS index with dim {self._embedding_dim}: {e_init}"
                                logger.error(f"FAISSVectorStore Add: {msg}", exc_info=True)
                                errors_list.append(msg)
                                return {"added_count": 0, "errors": errors_list} # Fatal error for add
                    else: # First vector is empty, cannot infer dimension
                        msg = "Cannot infer embedding dimension from empty first vector."
                        logger.error(f"FAISSVectorStore Add: {msg}")
                        errors_list.append(msg)
                        continue # Skip this chunk

                if not self._index: # Should be initialized by now if _embedding_dim is set
                    msg = "FAISS index not initialized."
                    logger.error(f"FAISSVectorStore Add: {msg}")
                    errors_list.append(msg)
                    # This is a critical state, maybe stop processing further
                    return {"added_count": added_count, "errors": errors_list}


                if not vector_list or len(vector_list) != self._embedding_dim:
                    msg = f"Embedding dimension mismatch for chunk ID '{chunk.id}'. Expected {self._embedding_dim}, got {len(vector_list) if vector_list else 'None'}. Skipping."
                    logger.warning(f"FAISSVectorStore Add: {msg}")
                    errors_list.append(msg)
                    continue

                current_batch_chunks.append(chunk)
                current_batch_vectors_np.append(np.array(vector_list, dtype=np.float32).reshape(1, -1))

                if len(current_batch_chunks) >= batch_size:
                    num_added_this_batch = await self._add_batch_to_faiss_and_docstore(current_batch_chunks, current_batch_vectors_np)
                    added_count += num_added_this_batch
                    current_batch_chunks, current_batch_vectors_np = [], []

            # Process any remaining chunks in the last batch
            if current_batch_chunks:
                num_added_this_batch = await self._add_batch_to_faiss_and_docstore(current_batch_chunks, current_batch_vectors_np)
                added_count += num_added_this_batch

        if self._index_file_path: # Auto-save after add operations
            await self._save_to_files()

        logger.info(f"FAISSVectorStore Add: Completed. Added {added_count} embeddings. Encountered {len(errors_list)} errors.")
        return {"added_count": added_count, "errors": errors_list}

    async def _add_batch_to_faiss_and_docstore(self, chunks: List[Chunk], vectors_np_list: List[Any]) -> int:
        """Synchronous part of adding a batch, run in executor. Assumes lock is held."""
        if not self._index or not chunks or not vectors_np_list or not np: return 0

        def _sync_add_batch():
            if not vectors_np_list: return 0
            try:
                concatenated_vectors = np.concatenate(vectors_np_list, axis=0)
                self._index.add(concatenated_vectors) # type: ignore

                count_this_op = 0
                for i, chunk_to_add in enumerate(chunks):
                    faiss_idx_for_chunk = self._next_faiss_idx + i
                    self._doc_store_by_faiss_idx[faiss_idx_for_chunk] = chunk_to_add

                    chunk_id_str = chunk_to_add.id or str(uuid.uuid4()) # Ensure ID exists
                    if chunk_id_str in self._chunk_id_to_faiss_idx:
                        logger.warning(f"FAISSVectorStore: Duplicate chunk ID '{chunk_id_str}' detected. Overwriting mapping. "
                                       "Note: Old vector in FAISS at previous index is not removed by this simple overwrite.")
                        # To handle updates properly, one might need to remove from FAISS first if supported.
                    self._chunk_id_to_faiss_idx[chunk_id_str] = faiss_idx_for_chunk
                    count_this_op +=1

                self._next_faiss_idx += len(chunks) # Update global counter for FAISS indices
                return count_this_op
            except Exception as e:
                logger.error(f"FAISSVectorStore: Error in _sync_add_batch: {e}", exc_info=True)
                return 0

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_add_batch)


    async def search(self, query_embedding: EmbeddingVector, top_k: int, filter_metadata: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None) -> List[RetrievedChunk]:
        """Searches the FAISS index for similar embeddings."""
        if not self._index or not self._embedding_dim or not np:
            logger.warning("FAISSVectorStore Search: Index not initialized or NumPy not available.")
            return []

        if len(query_embedding) != self._embedding_dim:
            logger.warning(f"FAISSVectorStore Search: Query embedding dimension mismatch. Expected {self._embedding_dim}, got {len(query_embedding)}.")
            return []
        if self._index.ntotal == 0: # type: ignore
            logger.debug("FAISSVectorStore Search: Index is empty.")
            return []

        query_vector_np = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        actual_top_k = min(top_k, self._index.ntotal) # type: ignore # Don't request more than available
        if actual_top_k == 0: return []


        def _sync_search_and_filter():
            try:
                distances, faiss_indices = self._index.search(query_vector_np, actual_top_k) # type: ignore
            except Exception as e_search:
                logger.error(f"FAISSVectorStore: Error during FAISS search: {e_search}", exc_info=True)
                return []

            results: List[RetrievedChunk] = []
            if faiss_indices.size > 0:
                for i in range(faiss_indices.shape[1]): # Iterate through results for the single query
                    faiss_idx = faiss_indices[0, i]
                    if faiss_idx == -1: continue # FAISS can return -1 if not enough results or error

                    dist = distances[0, i]
                    original_chunk = self._doc_store_by_faiss_idx.get(int(faiss_idx))

                    if original_chunk:
                        # Post-retrieval metadata filtering (less efficient than pre-filtering if FAISS supports it)
                        if filter_metadata:
                            match = all(original_chunk.metadata.get(k) == v for k, v in filter_metadata.items())
                            if not match:
                                continue

                        # Convert L2 distance to a similarity score (0-1 range, higher is better)
                        # This is a common heuristic, not a strict probability.
                        # Max distance could be estimated or normalized if needed for better score range.
                        # For normalized vectors, L2 distance squared = 2 - 2 * cosine_similarity
                        # score = 1.0 / (1.0 + dist) # Simple inverse, good for positive distances
                        # If vectors are normalized, cosine_similarity = 1 - (dist^2 / 2)
                        # Assuming vectors are not necessarily normalized for IndexFlatL2 for this score:
                        score = float(1.0 - (dist / (self._embedding_dim**0.5 + 1e-6))) # Heuristic normalization based on potential max L2
                        score = max(0.0, min(1.0, score)) # Clamp to 0-1

                        results.append(
                            cast(RetrievedChunk, _RetrievedChunkImpl(
                                content=original_chunk.content,
                                metadata=original_chunk.metadata,
                                score=score,
                                id=original_chunk.id,
                                rank=len(results) + 1
                            ))
                        )
            return results

        loop = asyncio.get_running_loop()
        # Lock for read access to doc_store if modifications could happen concurrently
        # For search, if add operations are also locked, this might be fine.
        # If add can happen during search, a ReadWriteLock or finer-grained locking is needed.
        # For V1, assume operations are not heavily interleaved or a higher-level lock serializes.
        async with self._lock: # Simple lock for now for both read/write parts
            return await loop.run_in_executor(None, _sync_search_and_filter)

    async def delete(self, ids: Optional[List[str]] = None, filter_metadata: Optional[Dict[str, Any]] = None, delete_all: bool = False, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Deletes items from FAISS. Complex for FAISS IndexFlatL2.
        This implementation will be basic: supports `delete_all` or no-op for selective.
        True selective deletion in FAISS often requires rebuilding the index or using specific index types
        that support `remove_ids` efficiently (and even then, it might just mark them, not shrink).
        """
        if not self._index or not faiss:
            logger.warning("FAISSVectorStore Delete: Index not initialized or FAISS not available.")
            return False

        async with self._lock:
            if delete_all:
                logger.info("FAISSVectorStore: Clearing all data (delete_all=True).")
                if self._embedding_dim : # Re-initialize with same dimension if known
                     if self._faiss_index_factory.upper() == "FLAT" or self._faiss_index_factory.upper() == "INDEXFLATL2":
                         self._index = faiss.IndexFlatL2(self._embedding_dim)
                     else:
                         self._index = faiss.index_factory(self._embedding_dim, self._faiss_index_factory)
                else: # Dimension not known, set index to None, will be re-inferred on next add
                    self._index = None
                self._doc_store_by_faiss_idx.clear()
                self._chunk_id_to_faiss_idx.clear()
                self._next_faiss_idx = 0
                deleted_count = -1 # Indicates all cleared
            elif ids:
                logger.warning("FAISSVectorStore: Selective deletion by ID is non-trivial for IndexFlatL2 and not fully implemented. "
                               "This operation may not remove vectors from the FAISS index itself, only from the doc store mapping.")
                # This basic version only removes from our Python-side mappings.
                # FAISS `remove_ids` requires the `IDMap` wrapper or specific index types.
                # For IndexFlatL2, direct removal is not standard.
                removed_from_mapping_count = 0
                faiss_indices_to_attempt_remove: List[int] = []
                for chunk_id_to_delete in ids:
                    if chunk_id_to_delete in self._chunk_id_to_faiss_idx:
                        faiss_idx = self._chunk_id_to_faiss_idx.pop(chunk_id_to_delete)
                        if faiss_idx in self._doc_store_by_faiss_idx:
                            del self._doc_store_by_faiss_idx[faiss_idx]
                        faiss_indices_to_attempt_remove.append(faiss_idx)
                        removed_from_mapping_count += 1

                if faiss_indices_to_attempt_remove and hasattr(self._index, "remove_ids"):
                     # This will only work if self._index is an IndexIDMap or supports remove_ids
                     # Needs np.array of int64
                    try:
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(None, self._index.remove_ids, np.array(faiss_indices_to_attempt_remove, dtype=np.int64))
                        logger.info(f"FAISSVectorStore: Attempted remove_ids for {len(faiss_indices_to_attempt_remove)} FAISS indices. "
                                    "Note: Actual vector removal depends on FAISS index type.")
                        # After remove_ids, FAISS index might need reconstruction or compaction for space reclamation.
                        # self._next_faiss_idx might become inconsistent if not managed carefully.
                        # For robust deletion, rebuilding the index without deleted items is safer for IndexFlat.
                    except Exception as e_remove:
                        logger.error(f"FAISSVectorStore: Error calling FAISS remove_ids: {e_remove}. "
                                     "Data might be inconsistent. Consider re-indexing.")

                logger.info(f"FAISSVectorStore: Removed {removed_from_mapping_count} items from internal mappings based on IDs.")
                # Does not guarantee removal from FAISS index data for IndexFlatL2.
                # To truly remove, one would typically rebuild the index from remaining items.
                # This is a V1 simplification.
                if removed_from_mapping_count > 0 and self._index_file_path: await self._save_to_files() # Resave if mappings changed
                return removed_from_mapping_count > 0

            elif filter_metadata:
                logger.warning("FAISSVectorStore: Deletion by metadata filter is not implemented. "
                               "Requires iterating all docs, then selective ID removal (see 'ids' case).")
                return False # Not implemented
            else: # No specific delete criteria
                return False

        if self._index_file_path and (delete_all or (ids and removed_from_mapping_count > 0)): # type: ignore # removed_from_mapping_count might be unbound
            await self._save_to_files()

        return True if delete_all or (ids and removed_from_mapping_count > 0) else False # type: ignore

    async def teardown(self) -> None:
        """Saves the index and doc store to files if paths are configured."""
        logger.debug("FAISSVectorStore: Tearing down...")
        await self._save_to_files() # Attempt to save on teardown
        self._index = None # Release FAISS index object
        self._doc_store_by_faiss_idx.clear()
        self._chunk_id_to_faiss_idx.clear()
        logger.info("FAISSVectorStore: Torn down. In-memory data cleared. Index (if configured) saved.")
