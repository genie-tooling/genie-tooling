### src/genie_tooling/vector_stores/impl/chromadb_store.py
import asyncio
import functools
import logging
import uuid
from pathlib import Path
from typing import Any, AsyncIterable, Dict, List, Optional, Tuple, cast

from genie_tooling.core.types import Chunk, EmbeddingVector, RetrievedChunk
from genie_tooling.vector_stores.abc import VectorStorePlugin

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    ChromaCollection = Any
    ChromaAPIClient = Any # Generic type for chromadb.api.API
    CHROMADB_AVAILABLE = True
except ImportError:
    chromadb = None
    ChromaSettings = None
    ChromaCollection = None
    ChromaAPIClient = None
    CHROMADB_AVAILABLE = False
    logger.warning(
        "ChromaDBVectorStore: 'chromadb' library not installed. "
        "This plugin will not be functional. Please install it: poetry add chromadb"
    )


class _RetrievedChunkImpl(RetrievedChunk, Chunk): # type: ignore
    def __init__(self, content: str, metadata: Dict[str, Any], score: float, id: Optional[str] = None, rank: Optional[int] = None):
        self.content: str = content
        self.metadata: Dict[str, Any] = metadata
        self.id: Optional[str] = id
        self.score: float = score
        self.rank: Optional[int] = rank

class ChromaDBVectorStore(VectorStorePlugin):
    plugin_id: str = "chromadb_vector_store_v1"
    description: str = "Vector store using ChromaDB for local or remote storage."

    _client: Optional[ChromaAPIClient] = None
    _collection: Optional[ChromaCollection] = None
    _collection_name: str = "genie_default_chroma_collection"
    _embedding_dim: Optional[int] = None
    _use_hnsw_indexing: bool = False
    _hnsw_space: str = "l2"
    _lock: asyncio.Lock

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._lock = asyncio.Lock()
        if not CHROMADB_AVAILABLE or not chromadb or not ChromaSettings:
            logger.error(f"{self.plugin_id} Error: 'chromadb' library not available or ChromaSettings not imported. Cannot initialize.")
            return

        cfg = config or {}
        self._collection_name = cfg.get("collection_name", self._collection_name)
        self._embedding_dim = cfg.get("embedding_dim")
        if self._embedding_dim is not None:
            try:
                self._embedding_dim = int(self._embedding_dim)
            except (ValueError, TypeError):
                logger.warning(f"{self.plugin_id}: Invalid embedding_dim value '{self._embedding_dim}'. Will try to infer.")
                self._embedding_dim = None

        self._use_hnsw_indexing = bool(cfg.get("use_hnsw_indexing", self._use_hnsw_indexing))
        self._hnsw_space = str(cfg.get("hnsw_space", self._hnsw_space)).lower()
        if self._hnsw_space not in ["l2", "ip", "cosine"]:
            logger.warning(f"{self.plugin_id}: Invalid 'hnsw_space' value '{self._hnsw_space}'. Defaulting to 'l2'.")
            self._hnsw_space = "l2"

        host = cfg.get("host")
        port_str = cfg.get("port")
        path_config = cfg.get("path")
        port: Optional[int] = None
        if port_str is not None:
            try:
                port = int(port_str)
            except (ValueError, TypeError):
                logger.error(f"{self.plugin_id}: Invalid port value '{port_str}'. Port must be an integer.")
                return

        try:
            if host and port is not None:
                logger.info(f"{self.plugin_id}: Configuring HttpClient for ChromaDB at {host}:{port}")
                settings = ChromaSettings(anonymized_telemetry=False)
                self._client = chromadb.HttpClient(host=host, port=port, settings=settings)
            elif path_config:
                logger.info(f"{self.plugin_id}: Configuring PersistentClient for ChromaDB at path: {path_config}")
                settings = ChromaSettings(anonymized_telemetry=False)
                self._client = chromadb.PersistentClient(path=str(path_config), settings=settings)
            else: # Default to ephemeral
                logger.info(f"{self.plugin_id}: Configuring EphemeralClient for ChromaDB (in-memory).")
                settings = ChromaSettings(anonymized_telemetry=False)
                self._client = chromadb.EphemeralClient(settings=settings)
        except Exception as e:
            logger.error(f"{self.plugin_id}: Failed to initialize ChromaDB client: {e}", exc_info=True)
            self._client = None
            return

        if not self._client:
            logger.error(f"{self.plugin_id}: ChromaDB client initialization failed.")
            return

        await self._ensure_collection_exists_async_internal()

    async def _ensure_collection_exists_async_internal(self) -> None:
        if self._collection:
            return
        if not self._client or not self._embedding_dim:
            logger.debug(f"{self.plugin_id}: Cannot ensure collection yet, client or embedding_dim missing.")
            return

        loop = asyncio.get_running_loop()
        def _get_or_create_collection_sync_internal():
            if not self._client: return None
            try:
                collection_metadata = {"hnsw:space": self._hnsw_space} if self._use_hnsw_indexing else None
                logger.info(f"{self.plugin_id}: Attempting to get/create collection '{self._collection_name}' with metadata: {collection_metadata}")
                return self._client.get_or_create_collection(
                    name=self._collection_name,
                    metadata=collection_metadata,
                    embedding_function=None # We provide embeddings directly
                )
            except Exception as e_coll:
                logger.error(f"{self.plugin_id}: Failed to get/create collection '{self._collection_name}' internally: {e_coll}", exc_info=True)
                return None
        async with self._lock: # Ensure only one task tries to create/get collection
            if not self._collection: # Re-check under lock
                self._collection = await loop.run_in_executor(None, _get_or_create_collection_sync_internal)
                if self._collection:
                    count = self._collection.count()
                    logger.info(f"{self.plugin_id}: Collection '{self._collection_name}' ensured/loaded. Current items: {count}. HNSW Indexing: {self._use_hnsw_indexing}, Space: '{self._hnsw_space}'.")
                else:
                    logger.error(f"{self.plugin_id}: Failed to obtain ChromaDB collection '{self._collection_name}' internally.")

    async def add(self, embeddings: AsyncIterable[Tuple[Chunk, EmbeddingVector]], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self._client:
            return {"added_count": 0, "errors": ["ChromaDB client not initialized."]}

        first_chunk_for_dim_inference: Optional[Chunk] = None
        first_vector_for_dim_inference: Optional[EmbeddingVector] = None
        processed_first_item_for_dim_inference = False


        # Temporarily store the stream to allow peeking for the first item
        # This is a common pattern for async iterables when you need the first item separately.
        embedding_items_list: List[Tuple[Chunk, EmbeddingVector]] = []
        try:
            first_item_tuple = await anext(embeddings)
            embedding_items_list.append(first_item_tuple)
            first_chunk_for_dim_inference, first_vector_for_dim_inference = first_item_tuple
            processed_first_item_for_dim_inference = True
        except StopAsyncIteration:
            logger.info(f"{self.plugin_id}: No embeddings provided to add().")
            return {"added_count": 0, "errors": []}
        except Exception as e_peek:
            logger.error(f"{self.plugin_id}: Error peeking first embedding for dimension inference: {e_peek}")
            return {"added_count": 0, "errors": [f"Error inferring dimension: {e_peek}"]}

        # Now, reconstruct the iterable if we consumed an item, or use the original
        async def _reconstructed_embeddings_stream() -> AsyncIterable[Tuple[Chunk, EmbeddingVector]]:
            for item in embedding_items_list: # Yield the peeked item first
                yield item
            async for item in embeddings: # Yield the rest of the original stream
                yield item

        current_embeddings_stream = _reconstructed_embeddings_stream()


        if self._embedding_dim is None and first_vector_for_dim_inference:
            if first_vector_for_dim_inference:
                self._embedding_dim = len(first_vector_for_dim_inference)
                logger.info(f"{self.plugin_id}: Inferred embedding dimension: {self._embedding_dim} from first vector in add().")
                await self._ensure_collection_exists_async_internal() # Ensure collection with new dim
            else: # Should not happen if first_item_tuple was successfully retrieved
                logger.warning(f"{self.plugin_id}: First vector in add() was empty after peeking, cannot infer dimension yet.")
        elif self._embedding_dim is None and not processed_first_item_for_dim_inference:
            # This case means the original stream was empty.
            logger.info(f"{self.plugin_id}: No embeddings provided to add (stream was empty from start).")
            return {"added_count": 0, "errors": []}


        if not self._collection and self._embedding_dim is not None: # If dim was just inferred
            await self._ensure_collection_exists_async_internal()

        if not self._collection:
            return {"added_count": 0, "errors": ["ChromaDB collection not initialized or could not be created."]}

        cfg = config or {}
        batch_size = int(cfg.get("batch_size", 100))
        current_batch_ids: List[str] = []
        current_batch_vectors: List[EmbeddingVector] = []
        current_batch_metadatas: List[Dict[str, Any]] = []
        current_batch_documents: List[str] = []
        added_count = 0
        errors_list: List[str] = []
        processed_count = 0
        loop = asyncio.get_running_loop()

        async def process_item_local(chunk: Chunk, vector_list: EmbeddingVector):
            nonlocal processed_count, errors_list # Use nonlocal for outer scope vars
            processed_count += 1
            if not vector_list:
                errors_list.append(f"Skipping chunk ID '{chunk.id}' due to empty vector.")
                return
            if self._embedding_dim and len(vector_list) != self._embedding_dim:
                errors_list.append(f"Dimension mismatch for chunk ID '{chunk.id}'. Expected {self._embedding_dim}, got {len(vector_list)}.")
                return

            chunk_id_str = chunk.id or str(uuid.uuid4())
            current_batch_ids.append(chunk_id_str)
            current_batch_vectors.append(vector_list)
            current_batch_documents.append(chunk.content)

            # Ensure metadata is a flat dictionary of supported types
            sanitized_meta: Dict[str, Any] = {"genie_chunk_id": chunk_id_str} # Ensure at least one metadata field
            if chunk.metadata:
                for k, v in chunk.metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        sanitized_meta[k] = v
                    # ChromaDB also supports lists of these primitives
                    elif isinstance(v, list) and all(isinstance(item, (str, int, float, bool)) for item in v):
                        sanitized_meta[k] = v
                    else:
                        logger.debug(f"Metadata key '{k}' for chunk '{chunk_id_str}' has unsupported type {type(v)}. Converting to string or skipping.")
                        try:
                            sanitized_meta[k] = str(v) # Attempt to stringify
                        except:
                            logger.warning(f"Could not stringify metadata key '{k}' for chunk '{chunk_id_str}'. Skipping this metadata field.")


            current_batch_metadatas.append(sanitized_meta)

        async with self._lock: # Lock for the entire add operation
            async for chunk, vector_list in current_embeddings_stream: # Use the reconstructed stream
                await process_item_local(chunk, vector_list)

                if len(current_batch_ids) >= batch_size:
                    try:
                        add_partial_func = functools.partial(self._collection.add, ids=current_batch_ids, embeddings=current_batch_vectors, metadatas=current_batch_metadatas, documents=current_batch_documents)
                        await loop.run_in_executor(None, add_partial_func)
                        added_count += len(current_batch_ids)
                    except Exception as e:
                        errors_list.append(f"Error adding batch to ChromaDB: {e}")
                        logger.error(f"{self.plugin_id}: Error during batch add: {e}", exc_info=True)
                    finally:
                        current_batch_ids, current_batch_vectors, current_batch_metadatas, current_batch_documents = [], [], [], []

            if current_batch_ids: # Process any remaining items
                try:
                    add_partial_final_func = functools.partial(self._collection.add, ids=current_batch_ids, embeddings=current_batch_vectors, metadatas=current_batch_metadatas, documents=current_batch_documents)
                    await loop.run_in_executor(None, add_partial_final_func)
                    added_count += len(current_batch_ids)
                except Exception as e:
                    errors_list.append(f"Error adding final batch to ChromaDB: {e}")
                    logger.error(f"{self.plugin_id}: Error during final batch add: {e}", exc_info=True)

        logger.info(f"{self.plugin_id}: Add operation complete. Processed: {processed_count}, Added: {added_count}, Errors: {len(errors_list)}")
        return {"added_count": added_count, "errors": errors_list}

    async def search(self, query_embedding: EmbeddingVector, top_k: int, filter_metadata: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None) -> List[RetrievedChunk]:
        if not self._collection or not query_embedding:
            if not self._collection: logger.debug(f"{self.plugin_id}: Search called but collection is not initialized.")
            if not query_embedding: logger.debug(f"{self.plugin_id}: Search called with empty query_embedding.")
            return []

        loop = asyncio.get_running_loop()
        def _sync_search():
            if not self._collection: return None # Should not happen if initial check passed
            try:
                current_count = self._collection.count() or 0
                if current_count == 0:
                    logger.debug(f"{self.plugin_id}: Search called on empty collection '{self._collection_name}'.")
                    # Return structure that matches expected ChromaDB query result for empty
                    return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

                n_results_param = min(top_k, current_count)
                logger.debug(f"{self.plugin_id}: Querying collection '{self._collection_name}' with top_k={n_results_param}. Filter: {filter_metadata is not None}")

                return self._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results_param,
                    where=filter_metadata,
                    include=["metadatas", "documents", "distances"] # Ensure all are included
                )
            except Exception as e:
                logger.error(f"{self.plugin_id}: Error during ChromaDB query: {e}", exc_info=True)
                return None

        async with self._lock: # Reading from collection, lock might be needed if collection object isn't thread/task safe
            chroma_results = await loop.run_in_executor(None, _sync_search)

        retrieved_chunks: List[RetrievedChunk] = []
        if chroma_results and chroma_results.get("ids") and isinstance(chroma_results["ids"], list) and chroma_results["ids"] and isinstance(chroma_results["ids"][0], list):
            ids_list = chroma_results["ids"][0]
            documents_list = chroma_results.get("documents", [[]])[0] if chroma_results.get("documents") else []
            metadatas_list = chroma_results.get("metadatas", [[]])[0] if chroma_results.get("metadatas") else []
            distances_list = chroma_results.get("distances", [[]])[0] if chroma_results.get("distances") else []

            for i, item_id_any in enumerate(ids_list):
                item_id = str(item_id_any)
                content = documents_list[i] if documents_list and i < len(documents_list) else ""
                metadata = metadatas_list[i] if metadatas_list and i < len(metadatas_list) else {}
                distance_val_any = distances_list[i] if distances_list and i < len(distances_list) else float("inf")

                if not isinstance(distance_val_any, (int, float)):
                    logger.warning(f"ChromaDB returned non-numeric distance for ID {item_id}: {distance_val_any}. Skipping item.")
                    continue

                distance_val = float(distance_val_any)
                score = 0.0
                # Score conversion based on distance metric
                if self._hnsw_space == "cosine": # Cosine distance: 0 is identical, 2 is opposite. Score: 1 - (dist/2)
                    score = max(0.0, 1.0 - (distance_val / 2.0))
                elif self._hnsw_space == "l2": # L2 distance: 0 is identical. Score: 1 / (1 + dist)
                    score = 1.0 / (1.0 + distance_val) if distance_val >= 0 else 0.0
                elif self._hnsw_space == "ip": # Inner product: higher is more similar. Max value depends on vector normalization.
                                             # Assuming vectors are normalized, IP is cosine similarity, range -1 to 1.
                                             # Convert to 0-1 range: (IP + 1) / 2
                    score = (distance_val + 1.0) / 2.0
                score = max(0.0, min(1.0, score)) # Clamp score to [0,1]

                retrieved_chunks.append(
                    cast(RetrievedChunk, _RetrievedChunkImpl(
                        id=item_id,
                        content=content,
                        metadata=cast(Dict, metadata),
                        score=score,
                        rank=i + 1
                    ))
                )
        logger.debug(f"{self.plugin_id}: Search returned {len(retrieved_chunks)} chunks.")
        return retrieved_chunks

    async def delete(self, ids: Optional[List[str]] = None, filter_metadata: Optional[Dict[str, Any]] = None, delete_all: bool = False, config: Optional[Dict[str, Any]] = None) -> bool:
        if not self._client:
            logger.warning(f"{self.plugin_id}: Client not initialized, cannot perform delete.")
            return False
        # If not delete_all and collection is None, it's effectively a success (nothing to delete from it)
        if not delete_all and not self._collection:
            logger.warning(f"{self.plugin_id}: Collection not available for specific delete. Assuming success as there's nothing to delete from this instance's perspective.")
            return True


        loop = asyncio.get_running_loop()
        def _sync_delete():
            try:
                if delete_all:
                    if not self._client:
                        logger.error(f"{self.plugin_id}: Client is None during delete_all sync operation.")
                        return False
                    collection_name_to_delete = self._collection_name
                    logger.info(f"{self.plugin_id}: Deleting entire collection '{collection_name_to_delete}'.")
                    try:
                        self._client.delete_collection(name=collection_name_to_delete)
                    except Exception as e_del_coll:
                        # If collection doesn't exist, Chroma might raise an error.
                        # We can consider this "successful" in the context of delete_all.
                        logger.warning(f"{self.plugin_id}: Error deleting collection '{collection_name_to_delete}' (it might not exist): {e_del_coll}")
                    self._collection = None # Mark collection as None so it's recreated on next use
                    return True
                elif self._collection: # If not delete_all, self._collection must exist
                    if ids:
                        self._collection.delete(ids=ids)
                        logger.info(f"{self.plugin_id}: Deleted {len(ids)} items by ID from '{self._collection_name}'.")
                        return True
                    elif filter_metadata:
                        self._collection.delete(where=filter_metadata)
                        logger.info(f"{self.plugin_id}: Attempted deletion by filter from '{self._collection_name}'.")
                        return True
                    else:
                        logger.warning(f"{self.plugin_id}: Delete called without IDs, filter, or delete_all=True. No action taken.")
                        return False # No action taken, but not an error
                else: # Specific delete but no collection
                    logger.warning(f"{self.plugin_id}: Delete called for specific items, but collection is not available.")
                    return True # Considered success as there's nothing to delete
            except Exception as e:
                logger.error(f"{self.plugin_id}: Error during ChromaDB delete operation: {e}", exc_info=True)
                return False

        async with self._lock: # Ensure atomicity for delete_all and collection reset
            return await loop.run_in_executor(None, _sync_delete)

    async def teardown(self) -> None:
        async with self._lock: # Ensure exclusive access during teardown
            self._collection = None # Release reference to collection
            # ChromaDB client does not have an explicit close method for Ephemeral/Persistent.
            # HttpClient might, but we are not using it directly in a way that requires manual close here.
            self._client = None
        logger.info(f"{self.plugin_id}: Teardown complete, client and collection references released.")