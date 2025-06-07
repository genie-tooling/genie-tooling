import logging
from typing import Any, AsyncIterable, Dict, List, Optional, cast

from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import Chunk, EmbeddingVector, RetrievedChunk
from genie_tooling.embedding_generators.abc import EmbeddingGeneratorPlugin
from genie_tooling.retrievers.abc import RetrieverPlugin
from genie_tooling.vector_stores.abc import VectorStorePlugin

logger = logging.getLogger(__name__)

class _QueryChunkForEmbedding:
    def __init__(self, query_text: str):
        self.content: str = query_text
        self.metadata: Dict[str, Any] = {"source": "retriever_query"}
        self.id: Optional[str] = "retriever_query_chunk"

class BasicSimilarityRetriever(RetrieverPlugin):
    plugin_id: str = "basic_similarity_retriever_v1"
    description: str = "Retrieves chunks based on semantic similarity between the query embedding and chunk embeddings, using configured embedder and vector store plugins."

    _plugin_manager: Optional[PluginManager] = None
    _embedder: Optional[EmbeddingGeneratorPlugin] = None
    _vector_store: Optional[VectorStorePlugin] = None
    _default_embedder_id: str = "sentence_transformer_embedder_v1"
    _default_vector_store_id: str = "faiss_vector_store_v1"
    _embedder_id_used: str = ""
    _vector_store_id_used: str = ""

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._plugin_manager = cfg.get("plugin_manager")
        if not self._plugin_manager or not isinstance(self._plugin_manager, PluginManager):
            logger.error(f"{self.plugin_id} Error: PluginManager not provided or invalid in config. Cannot load sub-plugins.")
            return

        self._embedder_id_used = cfg.get("embedder_id", self._default_embedder_id)
        embedder_config_for_setup = cfg.get("embedder_config", {}).copy()
        embedder_config_for_setup.setdefault("plugin_manager", self._plugin_manager) # Ensure PM is passed

        self._vector_store_id_used = cfg.get("vector_store_id", self._default_vector_store_id)
        vector_store_config_for_setup = cfg.get("vector_store_config", {}).copy()
        vector_store_config_for_setup.setdefault("plugin_manager", self._plugin_manager) # Ensure PM is passed

        logger.info(f"{self.plugin_id}: Initializing with Embedder='{self._embedder_id_used}', VectorStore='{self._vector_store_id_used}'.")

        embedder_instance = await self._plugin_manager.get_plugin_instance(self._embedder_id_used, config=embedder_config_for_setup)
        if embedder_instance and isinstance(embedder_instance, EmbeddingGeneratorPlugin):
            self._embedder = embedder_instance
            logger.debug(f"{self.plugin_id}: Embedder '{self._embedder_id_used}' loaded.")
        else:
            logger.error(f"{self.plugin_id} Error: EmbeddingGeneratorPlugin '{self._embedder_id_used}' not found or invalid.")
            self._embedder = None

        vector_store_instance = await self._plugin_manager.get_plugin_instance(self._vector_store_id_used, config=vector_store_config_for_setup)
        if vector_store_instance and isinstance(vector_store_instance, VectorStorePlugin):
            self._vector_store = vector_store_instance
            logger.debug(f"{self.plugin_id}: VectorStore '{self._vector_store_id_used}' loaded.")
        else:
            logger.error(f"{self.plugin_id} Error: VectorStorePlugin '{self._vector_store_id_used}' not found or invalid.")
            self._vector_store = None

        if self._embedder and self._vector_store:
            logger.info(f"{self.plugin_id}: Setup complete and sub-plugins loaded.")
        else:
            logger.error(f"{self.plugin_id}: Setup failed due to missing sub-plugins.")

    async def retrieve(self, query: str, top_k: int, config: Optional[Dict[str, Any]] = None) -> List[RetrievedChunk]:
        if not self._embedder or not self._vector_store:
            logger.error(f"{self.plugin_id} Error: Embedder or VectorStore not initialized. Cannot retrieve.")
            return []
        if not query or not query.strip():
            logger.warning(f"{self.plugin_id}: Empty query provided. Returning no results.")
            return []

        cfg = config or {}
        filter_metadata = cfg.get("filter_metadata")
        embedder_runtime_config = cfg.get("embedder_config", {}).copy() # Use copy for runtime config
        embedder_runtime_config.setdefault("plugin_manager", self._plugin_manager) # Ensure PM for embedder
        vector_store_runtime_config = cfg.get("vector_store_config", {}).copy() # Use copy
        vector_store_runtime_config.setdefault("plugin_manager", self._plugin_manager) # Ensure PM for vector store

        logger.debug(f"{self.plugin_id}: Embedding query: '{query[:100]}...'")
        async def query_chunk_stream() -> AsyncIterable[Chunk]: yield cast(Chunk, _QueryChunkForEmbedding(query_text=query))
        query_embedding_vector: Optional[EmbeddingVector] = None
        try:
            async for _chunk, vector in self._embedder.embed(chunks=query_chunk_stream(), config=embedder_runtime_config):
                query_embedding_vector = vector; break
            if not query_embedding_vector:
                logger.error(f"{self.plugin_id}: Failed to generate embedding for query. Embedder returned no vector.")
                return []
            logger.debug(f"{self.plugin_id}: Query embedded successfully.")
        except Exception as e_embed:
            logger.error(f"{self.plugin_id}: Error embedding query '{query[:100]}...': {e_embed}", exc_info=True)
            return []

        logger.debug(f"{self.plugin_id}: Searching vector store with top_k={top_k}, filter={filter_metadata is not None}.")
        try:
            results = await self._vector_store.search(query_embedding=query_embedding_vector, top_k=top_k, filter_metadata=filter_metadata, config=vector_store_runtime_config)
            logger.info(f"{self.plugin_id}: Retrieval complete. Found {len(results)} chunks.")
            return results
        except Exception as e_search:
            logger.error(f"{self.plugin_id}: Error searching vector store: {e_search}", exc_info=True)
            return []

    async def teardown(self) -> None:
        self._embedder = None; self._vector_store = None; self._plugin_manager = None
        logger.debug(f"{self.plugin_id}: Teardown complete (references released).")
