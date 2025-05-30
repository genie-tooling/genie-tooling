#src/genie_tooling/vector_stores/impl/chromadb_store.py
import asyncio
import functools
import logging
import uuid
from pathlib import Path
from typing import Any, AsyncIterable, Dict, List, Optional, Tuple, cast

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    ChromaCollection = Any
    ChromaAPIClient = Any
except ImportError:
    chromadb = None
    ChromaSettings = None
    ChromaCollection = None
    ChromaAPIClient = None

from genie_tooling.core.types import Chunk, EmbeddingVector, RetrievedChunk
from genie_tooling.vector_stores.abc import VectorStorePlugin


class _RetrievedChunkImpl(RetrievedChunk, Chunk):
    def __init__(self, content: str, metadata: Dict[str, Any], score: float, id: Optional[str] = None, rank: Optional[int] = None):
        self.content: str = content; self.metadata: Dict[str, Any] = metadata
        self.id: Optional[str] = id; self.score: float = score; self.rank: Optional[int] = rank

class ChromaDBVectorStore(VectorStorePlugin):
    plugin_id: str = "chromadb_vector_store_v1"
    description: str = "Vector store using ChromaDB."

    _client: Optional[ChromaAPIClient] = None
    _collection: Optional[ChromaCollection] = None
    _collection_name: str = "genie_default_collection"
    _path: Optional[str] = None
    _host: Optional[str] = None
    _port: Optional[int] = None
    _use_hnsw_indexing: bool = False
    _hnsw_space: str = "l2"
    _lock: asyncio.Lock

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._lock = asyncio.Lock()
        if not chromadb or not ChromaSettings:
            logger.error(f"{self.plugin_id} Error: 'chromadb-client' not installed.")
            return

        cfg = config or {}
        self._collection_name = cfg.get("collection_name", self._collection_name)
        self._host = cfg.get("host")
        self._port = cfg.get("port")
        if self._port and isinstance(self._port, str): self._port = int(self._port)

        client_type_determined: str
        if self._host and self._port:
            client_type_determined = "http"
            self._path = None
        elif "path" in cfg:
            self._path = cfg["path"]
            if self._path is None:
                client_type_determined = "ephemeral"
                logger.info(f"{self.plugin_id}: 'path' explicitly set to None in config, using ephemeral client.")
            else:
                client_type_determined = "persistent"
        else:
            default_base_path = Path("./.genie_data/chromadb")
            self._path = str(default_base_path / self._collection_name)
            logger.info(f"{self.plugin_id}: 'path' not provided in config, defaulting to persistent storage at '{self._path}'.")
            client_type_determined = "persistent"

        self._use_hnsw_indexing = bool(cfg.get("use_hnsw_indexing", self._use_hnsw_indexing))
        self._hnsw_space = cfg.get("hnsw_space", self._hnsw_space).lower()
        if self._hnsw_space not in ["l2", "ip", "cosine"]: self._hnsw_space = "l2"

        def _init_client_sync():
            try:
                settings = ChromaSettings(anonymized_telemetry=False)
                if client_type_determined == "http":
                    logger.info(f"{self.plugin_id}: Connecting to remote ChromaDB: {self._host}:{self._port}")
                    return chromadb.HttpClient(host=self._host, port=self._port, settings=settings) # type: ignore
                elif client_type_determined == "persistent" and self._path:
                    logger.info(f"{self.plugin_id}: Using persistent ChromaDB at: {self._path}")
                    Path(self._path).mkdir(parents=True, exist_ok=True)
                    return chromadb.PersistentClient(path=self._path, settings=settings) # type: ignore
                else: # Ephemeral
                    logger.info(f"{self.plugin_id}: Using ephemeral in-memory ChromaDB client.")
                    return chromadb.Client(settings=settings) # type: ignore
            except Exception as e:
                logger.error(f"{self.plugin_id}: Failed to init ChromaDB client (type: {client_type_determined}): {e}", exc_info=True)
                return None

        loop = asyncio.get_running_loop()
        self._client = await loop.run_in_executor(None, _init_client_sync)
        if not self._client: return

        def _get_coll_sync():
            if not self._client: return None
            try:
                coll_meta = {"hnsw:space": self._hnsw_space} if self._use_hnsw_indexing else None
                return self._client.get_or_create_collection(name=self._collection_name, metadata=coll_meta, embedding_function=None)
            except Exception as e:
                logger.error(f"{self.plugin_id}: Failed to get/create collection '{self._collection_name}': {e}", exc_info=True)
                return None
        self._collection = await loop.run_in_executor(None, _get_coll_sync)
        if self._collection: logger.info(f"{self.plugin_id}: Collection '{self._collection_name}' ensured. Count: {self._collection.count()}.")

    async def add(self, embeddings: AsyncIterable[Tuple[Chunk, EmbeddingVector]], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self._collection: return {"added_count": 0, "errors": ["ChromaDB collection not initialized."]}
        cfg = config or {}; batch_size = int(cfg.get("batch_size", 100))
        current_batch_ids, current_batch_vectors, current_batch_metadatas, current_batch_documents = [], [], [], []
        added_count, errors_list, processed_count = 0, [], 0
        async with self._lock:
            loop = asyncio.get_running_loop()
            async for chunk, vector_list in embeddings:
                processed_count += 1
                if not vector_list: errors_list.append(f"Skipping chunk ID '{chunk.id}' due to empty vector."); continue
                chunk_id_str = chunk.id or str(uuid.uuid4())
                current_batch_ids.append(chunk_id_str); current_batch_vectors.append(vector_list)
                current_batch_documents.append(chunk.content)
                sanitized_meta = {k: str(v) if not isinstance(v, (str, int, float, bool)) else v for k, v in chunk.metadata.items()}
                current_batch_metadatas.append(sanitized_meta)
                if len(current_batch_ids) >= batch_size:
                    try:
                        add_partial = functools.partial(self._collection.add, ids=current_batch_ids, embeddings=current_batch_vectors, metadatas=current_batch_metadatas, documents=current_batch_documents)
                        await loop.run_in_executor(None, add_partial)
                        added_count += len(current_batch_ids)
                    except Exception as e: errors_list.append(f"Error adding batch: {e}")
                    finally: current_batch_ids, current_batch_vectors, current_batch_metadatas, current_batch_documents = [], [], [], []
            if current_batch_ids:
                try:
                    add_partial_final = functools.partial(self._collection.add, ids=current_batch_ids, embeddings=current_batch_vectors, metadatas=current_batch_metadatas, documents=current_batch_documents)
                    await loop.run_in_executor(None, add_partial_final)
                    added_count += len(current_batch_ids)
                except Exception as e: errors_list.append(f"Error adding final batch: {e}")
        return {"added_count": added_count, "errors": errors_list}

    async def search(self, query_embedding: EmbeddingVector, top_k: int, filter_metadata: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None) -> List[RetrievedChunk]:
        if not self._collection or not query_embedding: return []
        def _sync_s():
            if not self._collection: return None
            try:
                count = self._collection.count() or 0
                if count == 0: return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
                n_res = min(top_k, count) if count > 0 else top_k
                return self._collection.query(query_embeddings=[query_embedding], n_results=n_res, where=filter_metadata, include=["metadatas", "documents", "distances"])
            except Exception as e: logger.error(f"{self.plugin_id}: Error sync search: {e}"); return None
        async with self._lock:
            loop = asyncio.get_running_loop(); chroma_results = await loop.run_in_executor(None, _sync_s)
        ret_chunks: List[RetrievedChunk] = []
        if chroma_results and chroma_results.get("ids") and chroma_results["ids"][0]:
            ids, docs, metas, dists = chroma_results["ids"][0], chroma_results.get("documents",[[]])[0], chroma_results.get("metadatas",[[]])[0], chroma_results.get("distances",[[]])[0]
            for i, cid in enumerate(ids):
                dist_val = dists[i] if dists and i < len(dists) else float("inf")
                score = (1.0 - dist_val) if self._hnsw_space == "cosine" else (1.0 / (1.0 + dist_val))
                ret_chunks.append(cast(RetrievedChunk, _RetrievedChunkImpl(id=cid, content=docs[i] if docs and i < len(docs) else "", metadata=cast(Dict,metas[i] if metas and i < len(metas) else {}), score=max(0.0,min(1.0,score)), rank=i+1)))
        return ret_chunks

    async def delete(self, ids: Optional[List[str]] = None, filter_metadata: Optional[Dict[str, Any]] = None, delete_all: bool = False, config: Optional[Dict[str, Any]] = None) -> bool:
        if not self._client and not self._collection: return False
        def _sync_d():
            if delete_all and self._client:
                try: self._client.delete_collection(name=self._collection_name); self._collection=None; return True
                except Exception as e: logger.error(f"Error deleting collection: {e}"); return False
            elif self._collection:
                if ids:
                    try: self._collection.delete(ids=ids); return True
                    except Exception as e: logger.error(f"Error deleting by IDs: {e}"); return False
                elif filter_metadata:
                    try: self._collection.delete(where=filter_metadata); return True
                    except Exception as e: logger.error(f"Error deleting by filter: {e}"); return False
            return False
        async with self._lock:
            loop = asyncio.get_running_loop(); return await loop.run_in_executor(None, _sync_d)

    async def teardown(self) -> None:
        async with self._lock: self._collection = None; self._client = None
        logger.debug(f"{self.plugin_id}: Torn down.")
