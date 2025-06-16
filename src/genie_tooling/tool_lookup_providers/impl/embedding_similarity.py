import logging
from typing import Any, AsyncIterable, Dict, List, Optional, Tuple, cast

from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import Chunk, EmbeddingVector
from genie_tooling.core.types import RetrievedChunk as CoreRetrievedChunk
from genie_tooling.embedding_generators.abc import EmbeddingGeneratorPlugin
from genie_tooling.lookup.types import RankedToolResult
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tool_lookup_providers.abc import ToolLookupProvider
from genie_tooling.vector_stores.abc import (
    VectorStorePlugin,
)

try:
    import numpy as np
except ImportError:
    np = None # type: ignore
    print("**WARNING** - NumPy not found. EmbeddingSimilarityLookupProvider's in-memory mode may not function optimally or at all.")

logger = logging.getLogger(__name__)

class _LookupQueryChunk(Chunk):
    def __init__(self, query_text: str, id_prefix: str = "lookup_query_"):
        self.content: str = query_text
        self.metadata: Dict[str, Any] = {"source": "lookup_query"}
        self.id: Optional[str] = f"{id_prefix}{hash(query_text)}"

class EmbeddingSimilarityLookupProvider(ToolLookupProvider):
    plugin_id: str = "embedding_similarity_lookup_v1"
    description: str = "Finds tools by comparing query embedding with tool description embeddings using cosine similarity, with optional persistent vector store."

    _indexed_tool_embeddings_np: Optional[Any] = None
    _indexed_tool_data_list_np: List[Dict[str, Any]] = []
    _tool_vector_store: Optional[VectorStorePlugin] = None
    _tool_embedding_collection_name: str = "genie_tool_embeddings_v1"
    _embedder: Optional[EmbeddingGeneratorPlugin] = None
    _plugin_manager: Optional[PluginManager] = None
    _key_provider: Optional[KeyProvider] = None
    DEFAULT_EMBEDDER_ID = "sentence_transformer_embedder_v1"
    DEFAULT_VECTOR_STORE_ID_FOR_TOOL_LOOKUP = "chromadb_vector_store_v1"

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._plugin_manager = cfg.get("plugin_manager")
        if not self._plugin_manager or not isinstance(self._plugin_manager, PluginManager):
            logger.error(f"{self.plugin_id} Error: PluginManager not provided. Cannot load sub-plugins.")
            return

        self._key_provider = cfg.get("key_provider")
        embedder_id_to_load = cfg.get("embedder_id", self.DEFAULT_EMBEDDER_ID)
        embedder_setup_config = cfg.get("embedder_config", {}).copy()
        embedder_setup_config.setdefault("plugin_manager", self._plugin_manager) # Ensure PM
        if self._key_provider and "key_provider" not in embedder_setup_config:
            embedder_setup_config["key_provider"] = self._key_provider

        embedder_instance_any = await self._plugin_manager.get_plugin_instance(embedder_id_to_load, config=embedder_setup_config)
        if embedder_instance_any and isinstance(embedder_instance_any, EmbeddingGeneratorPlugin):
            self._embedder = cast(EmbeddingGeneratorPlugin, embedder_instance_any)
            logger.info(f"{self.plugin_id}: Embedder '{embedder_id_to_load}' loaded.")
        else:
            logger.error(f"{self.plugin_id} Error: Embedder '{embedder_id_to_load}' not found/invalid.")
            return

        vector_store_id = cfg.get("vector_store_id")
        if vector_store_id:
            vs_config_for_setup = cfg.get("vector_store_config", {}).copy()
            vs_config_for_setup.setdefault("plugin_manager", self._plugin_manager) # Ensure PM
            self._tool_embedding_collection_name = cfg.get("tool_embeddings_collection_name", vs_config_for_setup.get("collection_name", self._tool_embedding_collection_name))
            vs_config_for_setup["collection_name"] = self._tool_embedding_collection_name
            tool_embeddings_path = cfg.get("tool_embeddings_path")
            if tool_embeddings_path is not None:
                vs_config_for_setup["path"] = tool_embeddings_path
            if self._key_provider and "key_provider" not in vs_config_for_setup:
                vs_config_for_setup["key_provider"] = self._key_provider
            vs_instance_any = await self._plugin_manager.get_plugin_instance(vector_store_id, config=vs_config_for_setup)
            if vs_instance_any and isinstance(vs_instance_any, VectorStorePlugin):
                self._tool_vector_store = cast(VectorStorePlugin, vs_instance_any)
                logger.info(f"{self.plugin_id}: Vector Store '{vector_store_id}' loaded. Collection: '{self._tool_embedding_collection_name}'. Path used: '{vs_config_for_setup.get('path', 'VS Default')}'")
                self._indexed_tool_embeddings_np = None
                self._indexed_tool_data_list_np = []
            else:
                logger.error(f"{self.plugin_id} Error: Vector Store '{vector_store_id}' not found/invalid. Will attempt in-memory NumPy if available.")
                self._tool_vector_store = None
        else:
            self._tool_vector_store = None
            logger.info(f"{self.plugin_id}: No Vector Store ID provided. Using in-memory NumPy index for tool embeddings.")
        if not self._tool_vector_store and not np:
            logger.error(f"{self.plugin_id} Error: NumPy not available and no Vector Store configured. Cannot function.")

    async def index_tools(self, tools_data: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None) -> None:
        if not self._embedder:
            logger.error(f"{self.plugin_id}: Embedder not available for indexing.")
            return

        texts_to_embed: List[str] = []
        tool_info_for_indexing: List[Dict[str, Any]] = []
        for item_data in tools_data:
            text_repr = item_data.get("lookup_text_representation")
            identifier = item_data.get("identifier")
            if isinstance(text_repr, str) and text_repr.strip() and identifier:
                texts_to_embed.append(text_repr)
                tool_info_for_indexing.append(item_data)
            else:
                logger.warning(f"{self.plugin_id}: Skipping tool data for indexing due to missing text or identifier: {str(item_data)[:100]}")

        if not texts_to_embed:
            logger.info(f"{self.plugin_id}: No valid tool texts provided for indexing.")
            return

        all_embeddings: List[EmbeddingVector] = []
        embedding_to_tool_info_map: Dict[int, Dict[str, Any]] = {}

        async def _chunks_stream_for_embed(texts: List[str], infos: List[Dict[str,Any]]) -> AsyncIterable[Chunk]:
            for i, text_content in enumerate(texts):
                chunk_id_prefix = infos[i].get("identifier", f"tool_desc_{i}")
                yield _LookupQueryChunk(text_content, id_prefix=f"{chunk_id_prefix}_idx_")

        embed_runtime_cfg = (config or {}).get("embedder_config", {}).copy()
        embed_runtime_cfg.setdefault("plugin_manager", self._plugin_manager)
        if self._key_provider and "key_provider" not in embed_runtime_cfg:
            embed_runtime_cfg["key_provider"] = self._key_provider

        try:
            original_chunk_index = 0
            async for chunk_obj, vec_list in self._embedder.embed(chunks=_chunks_stream_for_embed(texts_to_embed, tool_info_for_indexing), config=embed_runtime_cfg):
                if vec_list and isinstance(vec_list, list) and all(isinstance(x, float) for x in vec_list):
                    all_embeddings.append(vec_list)
                    if original_chunk_index < len(tool_info_for_indexing):
                        embedding_to_tool_info_map[len(all_embeddings)-1] = tool_info_for_indexing[original_chunk_index]
                else:
                    logger.warning(f"{self.plugin_id}: Invalid/empty embedding for content of chunk ID '{getattr(chunk_obj, 'id', 'N/A')}' (original index {original_chunk_index}). Skipped.")
                original_chunk_index += 1

            if len(all_embeddings) != len(texts_to_embed):
                logger.error(f"{self.plugin_id}: Mismatch after embedding. Expected {len(texts_to_embed)} embeddings, got {len(all_embeddings)}. Index may be incomplete.")
                tool_info_for_indexing = [embedding_to_tool_info_map[i] for i in range(len(all_embeddings)) if i in embedding_to_tool_info_map]
        except Exception as e:
            logger.error(f"{self.plugin_id}: Error during embedding tool texts: {e}", exc_info=True)
            return

        if not all_embeddings:
            logger.info(f"{self.plugin_id}: No embeddings were generated for tool texts.")
            return

        if self._tool_vector_store:
            async def _vs_stream_for_add() -> AsyncIterable[Tuple[Chunk, EmbeddingVector]]:
                for i, vec in enumerate(all_embeddings):
                    tool_info_item = tool_info_for_indexing[i]
                    vs_chunk = _LookupQueryChunk(tool_info_item["lookup_text_representation"], f"{tool_info_item['identifier']}_vs_")
                    vs_chunk.id = tool_info_item["identifier"]
                    vs_chunk.metadata = tool_info_item.get("_raw_metadata_snapshot", tool_info_item)
                    yield vs_chunk, vec

            vs_add_cfg = (config or {}).get("vector_store_config", {}).copy()
            vs_add_cfg.setdefault("plugin_manager", self._plugin_manager)
            if "collection_name" not in vs_add_cfg:
                vs_add_cfg["collection_name"] = self._tool_embedding_collection_name
            try:
                add_result = await self._tool_vector_store.add(embeddings=_vs_stream_for_add(), config=vs_add_cfg)
                logger.info(f"{self.plugin_id}: Indexed {add_result.get('added_count',0)} tool embeddings to Vector Store '{self._tool_vector_store.plugin_id}'. Errors: {add_result.get('errors',[])}")
            except Exception as e_vs_add:
                logger.error(f"{self.plugin_id}: Error adding tool embeddings to Vector Store: {e_vs_add}", exc_info=True)
        elif np:
            try:
                self._indexed_tool_embeddings_np = np.array(all_embeddings, dtype=np.float32)
                self._indexed_tool_data_list_np = tool_info_for_indexing
                logger.info(f"{self.plugin_id}: Indexed {len(self._indexed_tool_data_list_np)} tools using in-memory NumPy array (Shape: {self._indexed_tool_embeddings_np.shape}).")
            except ValueError as e_np:
                logger.error(f"{self.plugin_id}: Failed to convert embeddings to NumPy array: {e_np}. In-memory indexing failed.", exc_info=True)
                self._indexed_tool_embeddings_np = None
                self._indexed_tool_data_list_np = []
        else:
            logger.error(f"{self.plugin_id}: Neither Vector Store nor NumPy is available for indexing tool embeddings.")

    async def add_tool(self, tool_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> bool:
        # For this provider, add and update are the same operation (upsert).
        return await self.update_tool(tool_data.get("identifier", ""), tool_data, config)

    async def update_tool(self, tool_id: str, tool_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> bool:
        if not tool_id:
            logger.warning(f"{self.plugin_id}: update_tool called with no tool_id.")
            return False
        # Re-use the batch indexing logic with a single item.
        await self.index_tools([tool_data], config)
        return True # Assume success if no exception. `index_tools` logs errors.

    async def remove_tool(self, tool_id: str, config: Optional[Dict[str, Any]] = None) -> bool:
        if not tool_id:
            return False
        if self._tool_vector_store:
            vs_delete_cfg = (config or {}).get("vector_store_config", {}).copy()
            vs_delete_cfg.setdefault("plugin_manager", self._plugin_manager)
            if "collection_name" not in vs_delete_cfg:
                vs_delete_cfg["collection_name"] = self._tool_embedding_collection_name
            return await self._tool_vector_store.delete(ids=[tool_id], config=vs_delete_cfg)
        elif np and self._indexed_tool_embeddings_np is not None:
            # In-memory removal is more complex; requires rebuilding the array.
            # For simplicity, we'll just log a warning that this is inefficient.
            logger.warning(f"{self.plugin_id}: In-memory index does not efficiently support single-tool removal. Consider full re-indexing for changes.")
            # A simple implementation could filter and rebuild, but it's costly.
            return False
        return True

    async def find_tools(self, natural_language_query: str, top_k: int = 5, config: Optional[Dict[str,Any]]=None) -> List[RankedToolResult]:
        if not natural_language_query or not natural_language_query.strip():
            return []
        if not self._embedder:
            logger.error(f"{self.plugin_id}: Embedder not available for query.")
            return []

        async def _query_chunk_stream_for_embed() -> AsyncIterable[Chunk]:
            yield _LookupQueryChunk(natural_language_query)

        query_embedding: Optional[EmbeddingVector] = None
        embed_runtime_cfg = (config or {}).get("embedder_config", {}).copy()
        embed_runtime_cfg.setdefault("plugin_manager", self._plugin_manager)
        if self._key_provider and "key_provider" not in embed_runtime_cfg:
            embed_runtime_cfg["key_provider"] = self._key_provider

        try:
            async for _chunk, vec_list in self._embedder.embed(chunks=_query_chunk_stream_for_embed(), config=embed_runtime_cfg):
                if vec_list and isinstance(vec_list, list) and all(isinstance(x, float) for x in vec_list):
                    query_embedding = vec_list
                    break
        except Exception as e_q_embed:
            logger.error(f"{self.plugin_id}: Error embedding query '{natural_language_query[:50]}...': {e_q_embed}", exc_info=True)
            return []

        if not query_embedding:
            logger.error(f"{self.plugin_id}: Failed to generate embedding for query.")
            return []

        if self._tool_vector_store:
            vs_search_cfg = (config or {}).get("vector_store_config", {}).copy()
            vs_search_cfg.setdefault("plugin_manager", self._plugin_manager)
            if "collection_name" not in vs_search_cfg:
                vs_search_cfg["collection_name"] = self._tool_embedding_collection_name
            try:
                retrieved_vs_chunks: List[CoreRetrievedChunk] = await self._tool_vector_store.search(query_embedding, top_k, config=vs_search_cfg)
                return [
                    RankedToolResult(
                        tool_identifier=rc.id or f"vs_retrieved_{i}",
                        score=rc.score,
                        matched_tool_data=rc.metadata or {"id": rc.id, "text_content": rc.content},
                        description_snippet=rc.content[:200],
                        similarity_score_details={"score": rc.score}
                    ) for i, rc in enumerate(retrieved_vs_chunks)
                ]
            except Exception as e_vs_search:
                logger.error(f"{self.plugin_id}: Error searching Vector Store: {e_vs_search}", exc_info=True)
                return []
        elif np and self._indexed_tool_embeddings_np is not None and hasattr(self._indexed_tool_embeddings_np, "shape") and self._indexed_tool_embeddings_np.shape and self._indexed_tool_embeddings_np.size > 0:
            query_np_array = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            index_embeddings_np = self._indexed_tool_embeddings_np
            if index_embeddings_np.ndim == 1:
                if index_embeddings_np.shape[0] == query_np_array.shape[1]:
                    index_embeddings_np = index_embeddings_np.reshape(1, -1)
                else:
                    logger.error(f"{self.plugin_id}: In-memory index has unexpected shape for a single item: {index_embeddings_np.shape}")
                    return []
            if index_embeddings_np.size == 0 or index_embeddings_np.shape[1] != query_np_array.shape[1]:
                logger.error(f"{self.plugin_id}: In-memory index/query dimension mismatch ({index_embeddings_np.shape} vs {query_np_array.shape}). Cannot compute similarity.")
                return []

            dot_products = np.dot(index_embeddings_np, query_np_array.T).flatten()
            query_norm = np.linalg.norm(query_np_array)
            index_norms = np.linalg.norm(index_embeddings_np, axis=1)
            similarities = dot_products / ((index_norms * query_norm) + 1e-9)
            similarities = np.clip(similarities, -1.0, 1.0)
            similarities = np.nan_to_num(similarities, nan=-1.0)

            num_tools_in_index = len(self._indexed_tool_data_list_np)
            actual_top_k = min(top_k, num_tools_in_index)
            if actual_top_k == 0:
                return []

            top_k_indices = np.argsort(-similarities)[:actual_top_k]
            results: List[RankedToolResult] = []
            for i in top_k_indices:
                if 0 <= i < num_tools_in_index:
                    tool_data = self._indexed_tool_data_list_np[i]
                    score = (float(similarities[i]) + 1.0) / 2.0
                    results.append(RankedToolResult(
                        tool_identifier=tool_data.get("identifier", f"unknown_tool_{i}"),
                        score=score,
                        matched_tool_data=tool_data,
                        description_snippet=tool_data.get("lookup_text_representation","")[:200],
                        similarity_score_details={"cosine_similarity": float(similarities[i])}
                    ))
            return results
        else:
            logger.warning(f"{self.plugin_id}: Tool index not built or NumPy not available for search.")
            return []

    async def teardown(self) -> None:
        self._embedder = None
        self._tool_vector_store = None
        self._plugin_manager = None
        self._key_provider = None
        self._indexed_tool_embeddings_np = None
        self._indexed_tool_data_list_np = []
        logger.debug(f"{self.plugin_id}: Teardown complete, resources released.")
