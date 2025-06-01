### src/genie_tooling/vector_stores/impl/qdrant_store.py
import asyncio
import logging
import uuid
from typing import Any, AsyncIterable, Dict, List, Optional, Tuple

from genie_tooling.core.types import Chunk, EmbeddingVector, RetrievedChunk
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.vector_stores.abc import VectorStorePlugin

logger = logging.getLogger(__name__)

try:
    from qdrant_client import AsyncQdrantClient, QdrantClient
    from qdrant_client.http import models as rest
    from qdrant_client.http.models import (
        Distance as QdrantDistance,
    )
    from qdrant_client.http.models import (
        FieldCondition,
        Filter,
        MatchValue,
        PointStruct,
        VectorParams,
    )
    QDRANT_CLIENT_AVAILABLE = True
except ImportError:
    QDRANT_CLIENT_AVAILABLE = False
    AsyncQdrantClient = None # type: ignore
    QdrantClient = None # type: ignore
    rest = None # type: ignore
    QdrantDistance = None # type: ignore
    Filter = None # type: ignore
    FieldCondition = None # type: ignore
    MatchValue = None # type: ignore
    PointStruct = None # type: ignore
    VectorParams = None # type: ignore
    logger.warning(
        "QdrantVectorStorePlugin: 'qdrant-client' library not installed. "
        "This plugin will not be functional. Please install it: poetry add qdrant-client"
    )

# Helper for RetrievedChunk, as it's a protocol
class _RetrievedChunkImpl(RetrievedChunk, Chunk): # type: ignore
    def __init__(self, content: str, metadata: Dict[str, Any], score: float, id: Optional[str] = None, rank: Optional[int] = None):
        self.content: str = content
        self.metadata: Dict[str, Any] = metadata
        self.id: Optional[str] = id
        self.score: float = score
        self.rank: Optional[int] = rank


class QdrantVectorStorePlugin(VectorStorePlugin):
    plugin_id: str = "qdrant_vector_store_v1"
    description: str = "Vector store using Qdrant."

    _client: Optional[AsyncQdrantClient] = None
    _collection_name: str
    _embedding_dim: Optional[int] = None
    _distance_metric: Any
    _lock: asyncio.Lock

    KEY_URL = "url"
    KEY_HOST = "host"
    KEY_PORT = "port"
    KEY_API_KEY_NAME = "api_key_name"
    KEY_COLLECTION_NAME = "collection_name"
    KEY_EMBEDDING_DIM = "embedding_dim"
    KEY_DISTANCE_METRIC = "distance_metric"
    KEY_PREFER_GRPC = "prefer_grpc"
    KEY_TIMEOUT = "timeout_seconds"
    KEY_PATH = "path"

    def __init__(self):
        if QDRANT_CLIENT_AVAILABLE and QdrantDistance is not None:
            self._distance_metric = QdrantDistance.COSINE
        else:
            self._distance_metric = "Cosine"

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        if not QDRANT_CLIENT_AVAILABLE or QdrantDistance is None:
            logger.error(f"{self.plugin_id}: qdrant-client library not available or Distance type not loaded. Cannot initialize.")
            return

        self._lock = asyncio.Lock()
        cfg = config or {}

        self._collection_name = cfg.get(self.KEY_COLLECTION_NAME, "genie_default_qdrant_collection")
        self._embedding_dim = cfg.get(self.KEY_EMBEDDING_DIM)
        if self._embedding_dim:
            self._embedding_dim = int(self._embedding_dim)

        distance_str = cfg.get(self.KEY_DISTANCE_METRIC, "Cosine").upper()
        if distance_str == "EUCLID": # Common alternative spelling
            distance_str = "EUCLIDEAN"

        self._distance_metric = getattr(QdrantDistance, distance_str, QdrantDistance.COSINE)


        api_key: Optional[str] = None
        api_key_name = cfg.get(self.KEY_API_KEY_NAME)
        key_provider: Optional[KeyProvider] = cfg.get("key_provider")

        if api_key_name and key_provider:
            api_key = await key_provider.get_key(api_key_name)
            if not api_key:
                logger.warning(f"{self.plugin_id}: API key '{api_key_name}' not found via KeyProvider.")

        client_args: Dict[str, Any] = {
            "prefer_grpc": bool(cfg.get(self.KEY_PREFER_GRPC, False)),
            "timeout": float(cfg.get(self.KEY_TIMEOUT, 10.0)),
        }
        if api_key:
            client_args["api_key"] = api_key

        url = cfg.get(self.KEY_URL)
        host = cfg.get(self.KEY_HOST)
        port_cfg = cfg.get(self.KEY_PORT)
        port = int(port_cfg) if port_cfg is not None else None

        path_in_config = cfg.get(self.KEY_PATH)
        path_explicitly_none = self.KEY_PATH in cfg and path_in_config is None

        client_mode_info = ""
        if url:
            client_args["url"] = url
            client_mode_info = f"remote URL: {url}"
        elif host and port is not None:
            client_args["host"] = host
            client_args["port"] = port
            client_mode_info = f"remote host: {host}:{port}"
        elif path_in_config is not None and not path_explicitly_none:
            client_args["path"] = str(path_in_config)
            client_mode_info = f"local path: {str(path_in_config)}"
        elif path_explicitly_none:
            client_args["location"] = ":memory:" # Use location for explicit in-memory as per qdrant-client for QdrantClient, path=None for AsyncQdrantClient
            client_args["path"] = None # For AsyncQdrantClient, path=None is in-memory.
            client_mode_info = "in-memory (path explicitly None)"
        else:
            # Default in-memory: No location, path, url, or host/port args
            client_mode_info = "in-memory (default)"
            # For AsyncQdrantClient, if path is not set and no URL/host/port, it defaults to in-memory

        try:
            self._client = AsyncQdrantClient(**client_args)
            logger.info(f"{self.plugin_id}: Initialized Qdrant client ({client_mode_info}). Collection: '{self._collection_name}'.")
            if self._embedding_dim:
                await self._ensure_collection_exists_async()
        except Exception as e:
            logger.error(f"{self.plugin_id}: Failed to initialize Qdrant client: {e}", exc_info=True)
            self._client = None

    async def _ensure_collection_exists_async(self) -> None:
        if not self._client or not self._embedding_dim or not QDRANT_CLIENT_AVAILABLE or not VectorParams:
            return
        try:
            async with self._lock:
                try:
                    await self._client.get_collection(collection_name=self._collection_name)
                    logger.debug(f"Collection '{self._collection_name}' already exists.")
                except Exception as e: # Catch a more general exception initially
                    # Check if it's a "not found" type error based on status or message
                    is_not_found_error = False
                    if hasattr(e, "status_code") and e.status_code == 404: # type: ignore
                        is_not_found_error = True
                    elif "not found" in str(e).lower() or "code=not_found" in str(e).lower():
                        is_not_found_error = True

                    if is_not_found_error:
                        logger.info(f"Collection '{self._collection_name}' not found. Creating with dim={self._embedding_dim}, distance={self._distance_metric}.")
                        await self._client.create_collection(
                            collection_name=self._collection_name,
                            vectors_config=VectorParams(size=self._embedding_dim, distance=self._distance_metric),
                        )
                    else:
                        # Re-raise if it's not a "not found" error we can handle by creating
                        raise
        except Exception as e:
            logger.error(f"{self.plugin_id}: Error ensuring collection '{self._collection_name}' exists: {e}", exc_info=True)

    async def add(self, embeddings: AsyncIterable[Tuple[Chunk, EmbeddingVector]], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self._client or not QDRANT_CLIENT_AVAILABLE or not PointStruct:
            return {"added_count": 0, "errors": ["Qdrant client not initialized or library not available."]}
        cfg = config or {}
        batch_size = int(cfg.get("batch_size", 100))
        points_batch: List[PointStruct] = []
        added_count = 0
        errors_list: List[str] = []
        first_vector_processed = False

        async for chunk, vector_list in embeddings:
            if not self._embedding_dim and not first_vector_processed and vector_list:
                self._embedding_dim = len(vector_list)
                logger.info(f"{self.plugin_id}: Inferred embedding dimension: {self._embedding_dim} from first vector.")
                await self._ensure_collection_exists_async()
                if not self._client: # Check again if client became None after ensure_collection failure
                     return {"added_count": 0, "errors": ["Qdrant client failed post-collection-ensure."]}
            first_vector_processed = True

            if not vector_list:
                errors_list.append(f"Skipping chunk ID '{chunk.id}' due to empty vector.")
                continue

            if self._embedding_dim and len(vector_list) != self._embedding_dim:
                errors_list.append(f"Dimension mismatch for chunk ID '{chunk.id}'. Expected {self._embedding_dim}, got {len(vector_list)}.")
                continue

            point_id = chunk.id or str(uuid.uuid4())
            payload = {"content": chunk.content, "metadata": chunk.metadata or {}}

            points_batch.append(PointStruct(id=point_id, vector=vector_list, payload=payload))

            if len(points_batch) >= batch_size:
                try:
                    await self._client.upsert(collection_name=self._collection_name, points=points_batch, wait=True)
                    added_count += len(points_batch)
                except Exception as e:
                    errors_list.append(f"Error upserting batch to Qdrant: {e}")
                    logger.error(f"{self.plugin_id}: Error upserting batch: {e}", exc_info=True)
                finally:
                    points_batch = []

        if points_batch:
            try:
                await self._client.upsert(collection_name=self._collection_name, points=points_batch, wait=True)
                added_count += len(points_batch)
            except Exception as e:
                errors_list.append(f"Error upserting final batch to Qdrant: {e}")
                logger.error(f"{self.plugin_id}: Error upserting final batch: {e}", exc_info=True)

        return {"added_count": added_count, "errors": errors_list}


    def _to_qdrant_filter(self, filter_metadata: Dict[str, Any]) -> Optional[Filter]:
        if not filter_metadata or not QDRANT_CLIENT_AVAILABLE or not Filter or not FieldCondition or not MatchValue: return None
        must_conditions: List[FieldCondition] = []
        for key, value in filter_metadata.items():
            must_conditions.append(FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value)))
        return Filter(must=must_conditions) if must_conditions else None

    async def search(self, query_embedding: EmbeddingVector, top_k: int, filter_metadata: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None) -> List[RetrievedChunk]:
        if not self._client or not QDRANT_CLIENT_AVAILABLE:
            return []
        if not self._embedding_dim:
            logger.warning(f"{self.plugin_id}: Embedding dimension unknown. Cannot search.")
            return []
        if len(query_embedding) != self._embedding_dim:
            logger.warning(f"{self.plugin_id}: Query embedding dim {len(query_embedding)} != index dim {self._embedding_dim}.")
            return []

        qdrant_filter = self._to_qdrant_filter(filter_metadata) if filter_metadata else None

        try:
            search_results = await self._client.search(
                collection_name=self._collection_name,
                query_vector=query_embedding,
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False
            )
            retrieved_chunks: List[RetrievedChunk] = []
            for i, hit in enumerate(search_results):
                payload = hit.payload or {}
                content = payload.get("content", "")
                metadata = payload.get("metadata", {})
                qdrant_score = hit.score
                final_score = qdrant_score
                if self._distance_metric != QdrantDistance.COSINE:
                    final_score = 1.0 / (1.0 + qdrant_score) if qdrant_score >= 0 else 0.0
                final_score = max(0.0, min(1.0, final_score))

                retrieved_chunks.append(
                    _RetrievedChunkImpl(
                        id=str(hit.id), content=content, metadata=metadata,
                        score=final_score, rank=i + 1
                    )
                )
            return retrieved_chunks
        except Exception as e:
            logger.error(f"{self.plugin_id}: Error searching Qdrant: {e}", exc_info=True)
            return []

    async def delete(self, ids: Optional[List[str]] = None, filter_metadata: Optional[Dict[str, Any]] = None, delete_all: bool = False, config: Optional[Dict[str, Any]] = None) -> bool:
        if not self._client or not QDRANT_CLIENT_AVAILABLE or not rest:
            return False
        try:
            if delete_all:
                await self._client.delete_collection(collection_name=self._collection_name)
                logger.info(f"{self.plugin_id}: Collection '{self._collection_name}' deleted for delete_all.")
                if self._embedding_dim: # Recreate if we know the dimension
                    await self._ensure_collection_exists_async()
                return True

            points_selector: Any = None
            if ids:
                points_selector = rest.PointIdsList(points=ids)
            elif filter_metadata:
                points_selector = self._to_qdrant_filter(filter_metadata)

            if points_selector:
                await self._client.delete(
                    collection_name=self._collection_name,
                    points_selector=points_selector,
                    wait=True
                )
                return True
            logger.warning(f"{self.plugin_id}: Delete called without specific IDs, filter, or delete_all=True.")
            return False
        except Exception as e:
            logger.error(f"{self.plugin_id}: Error deleting from Qdrant: {e}", exc_info=True)
            return False

    async def teardown(self) -> None:
        if self._client:
            try:
                await self._client.close()
            except Exception as e:
                logger.error(f"{self.plugin_id}: Error closing Qdrant client: {e}", exc_info=True)
            finally:
                self._client = None
        logger.info(f"{self.plugin_id}: Teardown complete.")
