###src/genie_tooling/rag/plugins/impl/vector_stores/chromadb_store.py###
"""ChromaDBVectorStore: Vector store using ChromaDB (local or remote)."""
import asyncio
import functools  # Import functools
import logging
import uuid  # For default IDs
from pathlib import Path
from typing import Any, AsyncIterable, Dict, List, Optional, Tuple, Union, cast

logger = logging.getLogger(__name__)

# Attempt to import chromadb, make it optional
try:
    import chromadb  # type: ignore
    from chromadb.config import Settings as ChromaSettings  # type: ignore
    ChromaCollection = Any
    ChromaAPIClient = Any
except ImportError:
    chromadb = None # type: ignore
    ChromaSettings = None # type: ignore
    ChromaCollection = None # type: ignore
    ChromaAPIClient = None # type: ignore


from genie_tooling.core.types import Chunk, EmbeddingVector, RetrievedChunk
# Updated import path for VectorStorePlugin
from genie_tooling.vector_stores.abc import VectorStorePlugin


class _RetrievedChunkImpl(RetrievedChunk, Chunk):
    def __init__(self, content: str, metadata: Dict[str, Any], score: float, id: Optional[str] = None, rank: Optional[int] = None):
        self.content: str = content
        self.metadata: Dict[str, Any] = metadata
        self.id: Optional[str] = id
        self.score: float = score
        self.rank: Optional[int] = rank


class ChromaDBVectorStore(VectorStorePlugin):
    plugin_id: str = "chromadb_vector_store_v1"
    description: str = "Vector store using ChromaDB, supporting persistent local storage or a remote ChromaDB server."

    _client: Optional[ChromaAPIClient] = None
    _collection: Optional[ChromaCollection] = None
    _collection_name: str = "my_agentic_collection"
    _path: Optional[str] = None
    _host: Optional[str] = None
    _port: Optional[int] = None
    _use_hnsw_indexing: bool = False
    _hnsw_space: str = "l2"
    _lock: asyncio.Lock

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._lock = asyncio.Lock()
        if not chromadb or not ChromaSettings:
            logger.error("ChromaDBVectorStore Error: 'chromadb-client' library not installed. "
                         "Please install it with: poetry install --extras vectorstores")
            return

        cfg = config or {}
        self._collection_name = cfg.get("collection_name", self._collection_name)
        self._path = cfg.get("path")
        self._host = cfg.get("host")
        self._port = cfg.get("port")
        if self._port and isinstance(self._port, str): self._port = int(self._port)

        self._use_hnsw_indexing = bool(cfg.get("use_hnsw_indexing", self._use_hnsw_indexing))
        self._hnsw_space = cfg.get("hnsw_space", self._hnsw_space).lower()
        if self._hnsw_space not in ["l2", "ip", "cosine"]:
            logger.warning(f"ChromaDB: Invalid hnsw_space '{self._hnsw_space}', defaulting to 'l2'.")
            self._hnsw_space = "l2"

        def _init_client_sync():
            try:
                chroma_settings = ChromaSettings(anonymized_telemetry=False)
                if self._host and self._port:
                    logger.info(f"ChromaDBVectorStore: Connecting to remote ChromaDB at {self._host}:{self._port}")
                    return chromadb.HttpClient(host=self._host, port=self._port, settings=chroma_settings)
                elif self._path:
                    logger.info(f"ChromaDBVectorStore: Using persistent ChromaDB at path '{self._path}'")
                    Path(self._path).mkdir(parents=True, exist_ok=True)
                    return chromadb.PersistentClient(path=self._path, settings=chroma_settings)
                else:
                    logger.info("ChromaDBVectorStore: Using ephemeral in-memory ChromaDB client.")
                    return chromadb.Client(settings=chroma_settings)
            except Exception as e_client:
                logger.error(f"ChromaDBVectorStore: Failed to initialize ChromaDB client: {e_client}", exc_info=True)
                return None

        loop = asyncio.get_running_loop()
        self._client = await loop.run_in_executor(None, _init_client_sync)

        if not self._client: return

        def _get_or_create_collection_sync():
            if not self._client: return None
            try:
                collection_metadata = None
                if self._use_hnsw_indexing:
                    collection_metadata = {"hnsw:space": self._hnsw_space}
                    logger.info(f"ChromaDB: Configuring collection '{self._collection_name}' for HNSW indexing with space '{self._hnsw_space}'.")

                return self._client.get_or_create_collection(
                    name=self._collection_name,
                    metadata=collection_metadata,
                    embedding_function=None
                )
            except Exception as e_coll:
                logger.error(f"ChromaDBVectorStore: Failed to get or create collection '{self._collection_name}': {e_coll}", exc_info=True)
                return None

        self._collection = await loop.run_in_executor(None, _get_or_create_collection_sync)
        if self._collection:
            logger.info(f"ChromaDBVectorStore: Collection '{self._collection_name}' ensured. Current count: {self._collection.count()}.")
        else:
            logger.error(f"ChromaDBVectorStore: Setup failed, collection '{self._collection_name}' could not be accessed.")


    async def add(self, embeddings: AsyncIterable[Tuple[Chunk, EmbeddingVector]], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self._collection:
            msg = "ChromaDB collection not initialized."
            logger.error(f"ChromaDBVectorStore Add: {msg}")
            return {"added_count": 0, "errors": [msg]}

        cfg = config or {}
        batch_size = int(cfg.get("batch_size", 100))

        current_batch_ids: List[str] = []
        current_batch_vectors: List[EmbeddingVector] = []
        current_batch_metadatas: List[Dict[str, Union[str, int, float, bool]]] = []
        current_batch_documents: List[str] = []

        added_count = 0
        errors_list: List[str] = []
        processed_count = 0

        async with self._lock:
            loop = asyncio.get_running_loop()
            async for chunk, vector_list in embeddings:
                processed_count += 1
                if not vector_list:
                    msg = f"Skipping chunk ID '{chunk.id}' due to empty embedding vector."
                    logger.warning(f"ChromaDBVectorStore Add: {msg}")
                    errors_list.append(msg)
                    continue

                chunk_id_str = chunk.id or str(uuid.uuid4())
                current_batch_ids.append(chunk_id_str)
                current_batch_vectors.append(vector_list)
                current_batch_documents.append(chunk.content)

                sanitized_meta: Dict[str, Union[str, int, float, bool]] = {}
                for k, v in chunk.metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        sanitized_meta[k] = v
                    else:
                        sanitized_meta[k] = str(v)
                current_batch_metadatas.append(sanitized_meta)

                if len(current_batch_ids) >= batch_size:
                    try:
                        add_partial = functools.partial(
                            self._collection.add,
                            ids=current_batch_ids,
                            embeddings=current_batch_vectors,
                            metadatas=current_batch_metadatas,
                            documents=current_batch_documents
                        )
                        await loop.run_in_executor(None, add_partial)
                        added_count += len(current_batch_ids)
                        logger.debug(f"ChromaDBVectorStore: Added batch of {len(current_batch_ids)} items.")
                    except Exception as e_add:
                        msg = f"Error adding batch to ChromaDB: {e_add}"
                        logger.error(f"ChromaDBVectorStore Add: {msg}", exc_info=True)
                        errors_list.append(msg)
                    finally:
                        current_batch_ids, current_batch_vectors, current_batch_metadatas, current_batch_documents = [], [], [], []

            if current_batch_ids:
                try:
                    add_partial_final = functools.partial(
                        self._collection.add,
                        ids=current_batch_ids,
                        embeddings=current_batch_vectors,
                        metadatas=current_batch_metadatas,
                        documents=current_batch_documents
                    )
                    await loop.run_in_executor(None, add_partial_final)
                    added_count += len(current_batch_ids)
                    logger.debug(f"ChromaDBVectorStore: Added final batch of {len(current_batch_ids)} items.")
                except Exception as e_add_final:
                    msg = f"Error adding final batch to ChromaDB: {e_add_final}"
                    logger.error(f"ChromaDBVectorStore Add: {msg}", exc_info=True)
                    errors_list.append(msg)

        logger.info(f"ChromaDBVectorStore Add: Completed. Processed {processed_count} chunks. Successfully added {added_count} to collection '{self._collection_name}'. Encountered {len(errors_list)} errors.")
        return {"added_count": added_count, "errors": errors_list}


    async def search(self, query_embedding: EmbeddingVector, top_k: int, filter_metadata: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None) -> List[RetrievedChunk]:
        if not self._collection:
            logger.warning("ChromaDBVectorStore Search: Collection not initialized.")
            return []
        if not query_embedding:
            logger.warning("ChromaDBVectorStore Search: Query embedding is empty.")
            return []

        logger.debug(f"ChromaDBVectorStore: Searching collection '{self._collection_name}' with top_k={top_k}, filter={filter_metadata is not None}.")

        def _sync_search():
            if not self._collection: return None
            try:
                current_collection_count = self._collection.count() or 0

                if current_collection_count == 0:
                    logger.debug(f"{self.plugin_id}: Collection is empty. Returning empty results from _sync_search.")
                    return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

                num_results_to_request = min(top_k, current_collection_count)
                if num_results_to_request <= 0 and top_k > 0 : # if top_k is positive but collection count made it 0
                    num_results_to_request = top_k # Chroma might handle n_results > count, or query for at least 1 if top_k > 0
                                                # Let's try with num_results_to_request as is, it might be more robust.
                                                # If top_k is 5 and count is 3, num_results_to_request is 3.
                                                # If top_k is 5 and count is 0, num_results_to_request is 0.
                                                # This was simplified, original logic was fine.
                                                # Reverting to: num_results_to_request = min(top_k, current_collection_count) if current_collection_count > 0 else top_k
                                                # The problem is if count is 0, n_results becomes top_k, which is not what we want.
                                                # The early return for current_collection_count == 0 is the key.

                return self._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=num_results_to_request,
                    where=filter_metadata,
                    include=["metadatas", "documents", "distances"]
                )
            except Exception as e_search:
                logger.error(f"ChromaDBVectorStore: Error during sync search: {e_search}", exc_info=True)
                return None

        async with self._lock:
            loop = asyncio.get_running_loop()
            chroma_results = await loop.run_in_executor(None, _sync_search)

        retrieved_chunks: List[RetrievedChunk] = []
        if chroma_results and chroma_results.get("ids") and chroma_results["ids"][0]:
            ids_list = chroma_results["ids"][0]
            docs_list = chroma_results.get("documents", [[]])[0]
            meta_list = chroma_results.get("metadatas", [[]])[0]
            dist_list = chroma_results.get("distances", [[]])[0]

            for i, chunk_id_str in enumerate(ids_list):
                content = docs_list[i] if docs_list and i < len(docs_list) else ""
                metadata = meta_list[i] if meta_list and i < len(meta_list) else {}
                distance = dist_list[i] if dist_list and i < len(dist_list) else float("inf")

                score = 0.0
                if self._hnsw_space == "cosine":
                    score = 1.0 - distance
                elif self._hnsw_space == "ip":
                    score = -distance if distance < 0 else (1.0 / (1.0 + distance))
                else:
                    score = 1.0 / (1.0 + distance)

                score = max(0.0, min(1.0, score))

                retrieved_chunks.append(
                    cast(RetrievedChunk, _RetrievedChunkImpl(
                        id=chunk_id_str,
                        content=content,
                        metadata=cast(Dict[str,Any], metadata),
                        score=score,
                        rank=i + 1
                    ))
                )
        logger.info(f"ChromaDBVectorStore: Search yielded {len(retrieved_chunks)} results.")
        return retrieved_chunks

    async def delete(self, ids: Optional[List[str]] = None, filter_metadata: Optional[Dict[str, Any]] = None, delete_all: bool = False, config: Optional[Dict[str, Any]] = None) -> bool:
        if not self._client and not self._collection :
            logger.warning("ChromaDBVectorStore Delete: Client or collection not initialized.")
            return False

        def _sync_delete():
            if delete_all and self._client:
                try:
                    self._client.delete_collection(name=self._collection_name)
                    logger.info(f"ChromaDBVectorStore: Collection '{self._collection_name}' deleted.")
                    self._collection = None
                    return True
                except Exception as e_del_coll:
                    logger.error(f"ChromaDBVectorStore: Error deleting collection '{self._collection_name}': {e_del_coll}", exc_info=True)
                    return False
            elif self._collection:
                if ids and filter_metadata:
                    logger.warning("ChromaDBVectorStore: Both 'ids' and 'filter_metadata' provided to delete. Using 'ids'.")
                    try:
                        self._collection.delete(ids=ids)
                        logger.info(f"ChromaDBVectorStore: Deleted {len(ids)} items by ID from '{self._collection_name}'.")
                        return True
                    except Exception as e_del_ids_with_filter:
                        logger.error(f"ChromaDBVectorStore: Error deleting items by IDs (with filter_metadata also present): {e_del_ids_with_filter}", exc_info=True)
                        return False
                elif ids:
                    try:
                        self._collection.delete(ids=ids)
                        logger.info(f"ChromaDBVectorStore: Deleted {len(ids)} items by ID from '{self._collection_name}'.")
                        return True
                    except Exception as e_del_ids:
                        logger.error(f"ChromaDBVectorStore: Error deleting items by IDs: {e_del_ids}", exc_info=True)
                        return False
                elif filter_metadata:
                    try:
                        self._collection.delete(where=filter_metadata)
                        logger.info(f"ChromaDBVectorStore: Deleted items by metadata filter from '{self._collection_name}'.")
                        return True
                    except Exception as e_del_where:
                        logger.error(f"ChromaDBVectorStore: Error deleting items by metadata filter: {e_del_where}", exc_info=True)
                        return False
            return False

        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, _sync_delete)

    async def teardown(self) -> None:
        async with self._lock:
            self._collection = None
            self._client = None
        logger.debug("ChromaDBVectorStore: Torn down (client/collection references released).")
