import logging
from typing import Any, AsyncIterable, Dict, List, Optional, cast

logger = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:
    np = None
    logger.warning("NumPy not found. EmbeddingSimilarityLookupProvider's in-memory mode may not function optimally or at all.")

from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import Chunk, EmbeddingVector
from genie_tooling.core.types import RetrievedChunk as CoreRetrievedChunk
from genie_tooling.embedding_generators.abc import EmbeddingGeneratorPlugin
from genie_tooling.lookup.types import RankedToolResult
from genie_tooling.tool_lookup_providers.abc import ToolLookupProvider
from genie_tooling.vector_stores.abc import (
    VectorStorePlugin,  # Import VectorStorePlugin
)


class _LookupQueryChunk(Chunk):
    def __init__(self, query_text: str, id_prefix: str = "lookup_query_"):
        self.content: str = query_text
        self.metadata: Dict[str, Any] = {"source": "lookup_query"}
        # Ensure ID is somewhat unique for potential internal use by embedders/vector stores
        self.id: Optional[str] = f"{id_prefix}{hash(query_text)}"

class EmbeddingSimilarityLookupProvider(ToolLookupProvider):
    plugin_id: str = "embedding_similarity_lookup_v1"
    description: str = "Finds tools by comparing query embedding with tool description embeddings using cosine similarity, with optional persistent vector store."

    # For in-memory (NumPy) mode
    _indexed_tool_embeddings_np: Optional[Any] = None # np.ndarray
    _indexed_tool_data_list_np: List[Dict[str, Any]] = []

    # For VectorStore mode
    _tool_vector_store: Optional[VectorStorePlugin] = None
    _tool_embedding_collection_name: str = "genie_tool_embeddings_v1" # Default collection name

    _embedder: Optional[EmbeddingGeneratorPlugin] = None
    _plugin_manager: Optional[PluginManager] = None

    DEFAULT_EMBEDDER_ID = "sentence_transformer_embedder_v1"

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._plugin_manager = cfg.get("plugin_manager")
        if not self._plugin_manager or not isinstance(self._plugin_manager, PluginManager):
            logger.error(f"{self.plugin_id} Error: PluginManager not provided in config. Cannot load sub-plugins.")
            return

        # 1. Load Embedder (common for both modes)
        embedder_id_to_load = cfg.get("embedder_id", self.DEFAULT_EMBEDDER_ID)
        embedder_setup_config = cfg.get("embedder_config", {})
        if "key_provider" not in embedder_setup_config and hasattr(self._plugin_manager, "_key_provider"): # Hacky access, better if KP passed explicitly
             # This assumes PluginManager might have a KeyProvider, which is not standard.
             # KeyProvider should ideally be passed down from the service that configures this lookup provider.
             # For now, let's assume embedder_config will contain it if needed.
             pass

        logger.info(f"{self.plugin_id}: Setting up with embedder '{embedder_id_to_load}'.")
        embedder_instance_any = await self._plugin_manager.get_plugin_instance(embedder_id_to_load, config=embedder_setup_config)
        if embedder_instance_any and isinstance(embedder_instance_any, EmbeddingGeneratorPlugin):
            self._embedder = cast(EmbeddingGeneratorPlugin, embedder_instance_any)
            logger.debug(f"{self.plugin_id}: Embedder '{embedder_id_to_load}' loaded.")
        else:
            logger.error(f"{self.plugin_id} Error: Embedder plugin '{embedder_id_to_load}' not found or invalid.")
            self._embedder = None
            return # Cannot function without an embedder

        # 2. Optionally Load VectorStore
        vector_store_id = cfg.get("vector_store_id")
        if vector_store_id:
            logger.info(f"{self.plugin_id}: Configuring to use VectorStore plugin '{vector_store_id}'.")
            vs_config_for_setup = cfg.get("vector_store_config", {})
            # Ensure collection name for tool embeddings is passed to the vector store's setup if it uses it.
            # Some vector stores might take collection_name at setup, others at add/search time.
            self._tool_embedding_collection_name = cfg.get("tool_embedding_collection_name", self._tool_embedding_collection_name)
            if "collection_name" not in vs_config_for_setup: # Prioritize specific, then default
                vs_config_for_setup["collection_name"] = self._tool_embedding_collection_name

            vs_instance_any = await self._plugin_manager.get_plugin_instance(vector_store_id, config=vs_config_for_setup)
            if vs_instance_any and isinstance(vs_instance_any, VectorStorePlugin):
                self._tool_vector_store = cast(VectorStorePlugin, vs_instance_any)
                logger.info(f"{self.plugin_id}: VectorStore '{vector_store_id}' loaded for tool embeddings (collection: '{self._tool_embedding_collection_name}'). In-memory NumPy index will NOT be used.")
                self._indexed_tool_embeddings_np = None # Ensure NumPy mode is off
                self._indexed_tool_data_list_np = []
            else:
                logger.error(f"{self.plugin_id} Error: VectorStore plugin '{vector_store_id}' configured but not found or invalid. Falling back to in-memory NumPy index if NumPy is available.")
                self._tool_vector_store = None
        else:
            logger.info(f"{self.plugin_id}: No VectorStore plugin configured. Will use in-memory NumPy index (if NumPy is available).")
            self._tool_vector_store = None

        if not self._tool_vector_store and not np:
             logger.error(f"{self.plugin_id} Error: NumPy not available and no VectorStore configured. This provider cannot function.")


    async def index_tools(self, tools_data: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None) -> None:
        if not self._embedder:
            logger.error(f"{self.plugin_id}: Embedder not available. Cannot index tools.")
            return

        texts_to_embed: List[str] = []
        # This list will store data needed for either NumPy index or for VectorStore Chunks
        tool_info_for_indexing: List[Dict[str, Any]] = []

        for tool_item_data in tools_data:
            lookup_text = tool_item_data.get("lookup_text_representation")
            identifier = tool_item_data.get("identifier")

            if isinstance(lookup_text, str) and lookup_text.strip() and identifier:
                texts_to_embed.append(lookup_text)
                tool_info_for_indexing.append(tool_item_data) # Store the whole formatted item
            else:
                logger.warning(f"{self.plugin_id}: Tool data for ID '{identifier}' missing 'lookup_text_representation' or identifier. Skipping from index.")

        if not texts_to_embed:
            logger.info(f"{self.plugin_id}: No valid tool texts provided to embed for indexing.")
            if self._tool_vector_store: pass # Vector store handles empty adds
            elif np: self._indexed_tool_embeddings_np = np.array([])
            self._indexed_tool_data_list_np = []
            return

        async def _tool_texts_as_chunks_for_embedding(infos: List[Dict[str,Any]]) -> AsyncIterable[Chunk]:
            for i, info_item in enumerate(infos):
                # Use a unique ID for the chunk, e.g., based on tool identifier for this indexing run
                chunk_id = info_item.get("identifier", f"tool_index_item_{i}")
                yield _LookupQueryChunk(query_text=info_item["lookup_text_representation"], id_prefix=f"{chunk_id}_text_")

        all_embeddings_list: List[EmbeddingVector] = []
        # Map index in all_embeddings_list back to the original tool_info_for_indexing item
        embedding_to_tool_info_map: Dict[int, Dict[str, Any]] = {}
        current_embedding_idx = 0

        embedder_runtime_config = (config or {}).get("embedder_config", (config or {}).get("embedder_runtime_config", {}))

        try:
            logger.debug(f"{self.plugin_id}: Embedding {len(texts_to_embed)} tool texts for index...")
            # We need to associate the resulting embedding with the correct tool_info_for_indexing item
            # The embedder yields (Chunk, EmbeddingVector). The Chunk here is our _LookupQueryChunk.
            # We need to reconstruct which original tool_info_for_indexing item it came from.
            # The _LookupQueryChunk's content IS texts_to_embed[i] if we build it carefully.

            # Create a stream of (original_tool_info, text_to_embed)
            # This is getting complex. Simpler: assume embedder preserves order.
            async for chunk_obj, vector_as_list in self._embedder.embed(
                chunks=_tool_texts_as_chunks_for_embedding(tool_info_for_indexing),
                config=embedder_runtime_config):
                if vector_as_list and isinstance(vector_as_list, list) and all(isinstance(x, float) for x in vector_as_list):
                    all_embeddings_list.append(vector_as_list)
                    # This assumes the order of embeddings matches the order of tool_info_for_indexing
                    if current_embedding_idx < len(tool_info_for_indexing):
                        embedding_to_tool_info_map[len(all_embeddings_list) - 1] = tool_info_for_indexing[current_embedding_idx]
                    current_embedding_idx += 1
                else:
                    logger.warning(f"{self.plugin_id}: Received invalid/empty embedding for tool text '{chunk_obj.content}'. It will be skipped.")
                    # If an embedding is skipped, current_embedding_idx should still advance to keep map correct for subsequent valid ones
                    current_embedding_idx +=1


        except Exception as e_embed_idx:
            logger.error(f"{self.plugin_id}: Error during tool text embedding for index: {e_embed_idx}", exc_info=True)
            if self._tool_vector_store: pass # Let vector store handle empty list if it occurs
            elif np: self._indexed_tool_embeddings_np = np.array([])
            self._indexed_tool_data_list_np = []
            return

        if not all_embeddings_list:
            logger.info(f"{self.plugin_id}: No embeddings generated. Index will be empty.")
            if self._tool_vector_store: pass
            elif np: self._indexed_tool_embeddings_np = np.array([])
            self._indexed_tool_data_list_np = []
            return

        # Mode 1: Using a VectorStore
        if self._tool_vector_store:
            async def _vector_store_chunk_stream() -> AsyncIterable[Tuple[Chunk, EmbeddingVector]]:
                for i, vector in enumerate(all_embeddings_list):
                    tool_info = embedding_to_tool_info_map.get(i)
                    if not tool_info: # Should not happen if logic above is correct
                        logger.warning(f"Could not find tool_info for embedding index {i}. Skipping.")
                        continue

                    # Create a Chunk to store in the vector store
                    # The ID must be the tool's actual identifier for later retrieval.
                    # Content is the text that was embedded.
                    # Metadata can be the raw tool metadata or the formatted data.
                    vs_chunk = _LookupQueryChunk(
                        query_text=tool_info["lookup_text_representation"],
                        id_prefix=tool_info["identifier"] + "_vecstore_"
                    )
                    # Override ID to be the tool identifier
                    vs_chunk.id = tool_info["identifier"]
                    vs_chunk.metadata = tool_info.get("_raw_metadata_snapshot", tool_info) # Store rich metadata

                    yield vs_chunk, vector

            vs_add_config = (config or {}).get("vector_store_config", {})
            if "collection_name" not in vs_add_config: # Ensure collection name for this specific operation
                vs_add_config["collection_name"] = self._tool_embedding_collection_name

            try:
                add_result = await self._tool_vector_store.add(embeddings=_vector_store_chunk_stream(), config=vs_add_config)
                logger.info(f"{self.plugin_id}: Indexed {add_result.get('added_count', 0)} tool embeddings into VectorStore '{self._tool_vector_store.plugin_id}' collection '{vs_add_config['collection_name']}'. Errors: {add_result.get('errors', [])}")
            except Exception as e_vs_add:
                logger.error(f"{self.plugin_id}: Error adding tool embeddings to VectorStore: {e_vs_add}", exc_info=True)

        # Mode 2: In-memory NumPy (only if VectorStore is not used and NumPy is available)
        elif np:
            try:
                self._indexed_tool_embeddings_np = np.array(all_embeddings_list, dtype=np.float32)
                # _indexed_tool_data_list_np needs to correspond to the embeddings in _indexed_tool_embeddings_np
                self._indexed_tool_data_list_np = [embedding_to_tool_info_map[i] for i in range(len(all_embeddings_list)) if i in embedding_to_tool_info_map]
                if len(self._indexed_tool_embeddings_np) != len(self._indexed_tool_data_list_np):
                    logger.error(f"CRITICAL: Mismatch between NumPy embeddings ({len(self._indexed_tool_embeddings_np)}) and tool data ({len(self._indexed_tool_data_list_np)}) after processing. Index may be corrupt.")
                    # Potentially clear them to avoid issues
                    self._indexed_tool_embeddings_np = np.array([])
                    self._indexed_tool_data_list_np = []


                logger.info(f"{self.plugin_id}: Successfully indexed {len(self._indexed_tool_data_list_np)} tools into in-memory NumPy array.")
            except ValueError as e_np_array:
                logger.error(f"{self.plugin_id}: Failed to convert embeddings to NumPy array: {e_np_array}. Likely inconsistent embedding dimensions.", exc_info=True)
                self._indexed_tool_embeddings_np = np.array([])
                self._indexed_tool_data_list_np = []
        else:
            logger.error(f"{self.plugin_id}: Neither VectorStore nor NumPy is available for indexing.")


    async def find_tools(
        self,
        natural_language_query: str,
        top_k: int = 5,
        config: Optional[Dict[str, Any]] = None
    ) -> List[RankedToolResult]:
        if not natural_language_query or not natural_language_query.strip():
            logger.debug(f"{self.plugin_id}: Empty query provided. Returning no results.")
            return []
        if not self._embedder:
            logger.error(f"{self.plugin_id}: Embedder not available. Cannot find tools.")
            return []

        async def _query_as_chunk_stream_for_find() -> AsyncIterable[Chunk]:
            yield _LookupQueryChunk(query_text=natural_language_query)

        query_embedding_list: Optional[List[float]] = None
        embedder_runtime_config = (config or {}).get("embedder_config", (config or {}).get("embedder_runtime_config", {}))
        try:
            logger.debug(f"{self.plugin_id}: Embedding lookup query: '{natural_language_query[:100]}...'")
            async for _chunk_obj, vector_as_list in self._embedder.embed(chunks=_query_as_chunk_stream_for_find(), config=embedder_runtime_config):
                if vector_as_list and isinstance(vector_as_list, list) and all(isinstance(x, float) for x in vector_as_list) :
                    query_embedding_list = vector_as_list
                break
        except Exception as e_embed_q:
            logger.error(f"{self.plugin_id}: Error embedding lookup query: {e_embed_q}", exc_info=True)
            return []

        if not query_embedding_list:
            logger.error(f"{self.plugin_id}: Failed to generate embedding for the lookup query.")
            return []

        # Mode 1: Search using VectorStore
        if self._tool_vector_store:
            vs_search_config = (config or {}).get("vector_store_config", {})
            if "collection_name" not in vs_search_config:
                vs_search_config["collection_name"] = self._tool_embedding_collection_name

            try:
                retrieved_vs_chunks: List[CoreRetrievedChunk] = await self._tool_vector_store.search(
                    query_embedding=query_embedding_list,
                    top_k=top_k,
                    config=vs_search_config
                    # filter_metadata can be part of vs_search_config if store supports it
                )
                results: List[RankedToolResult] = []
                for r_chunk in retrieved_vs_chunks:
                    # r_chunk.id should be the tool_identifier
                    # r_chunk.metadata should be the tool_info (or _raw_metadata_snapshot)
                    tool_data_from_vs = r_chunk.metadata or {"identifier": r_chunk.id, "lookup_text_representation": r_chunk.content}
                    if not tool_data_from_vs.get("identifier") and r_chunk.id: # Ensure identifier is present
                        tool_data_from_vs["identifier"] = r_chunk.id

                    results.append(RankedToolResult(
                        tool_identifier=r_chunk.id or "unknown_tool_from_vs",
                        score=r_chunk.score, # VectorStore score is already 0-1
                        matched_tool_data=tool_data_from_vs,
                        description_snippet=r_chunk.content[:200] + ("..." if len(r_chunk.content) > 200 else "")
                    ))
                logger.info(f"{self.plugin_id}: Found {len(results)} tools via VectorStore. Top score: {results[0].score if results else 'N/A'}.")
                return results
            except Exception as e_vs_search:
                logger.error(f"{self.plugin_id}: Error searching tool embeddings in VectorStore: {e_vs_search}", exc_info=True)
                return []

        # Mode 2: In-memory NumPy search (fallback if no VectorStore or if NumPy not available for VS mode)
        elif np and self._indexed_tool_embeddings_np is not None and hasattr(self._indexed_tool_embeddings_np, "size") and self._indexed_tool_embeddings_np.size > 0:
            query_vector_np = np.array(query_embedding_list, dtype=np.float32).reshape(1, -1)

            if self._indexed_tool_embeddings_np.ndim == 1:
                if self._indexed_tool_embeddings_np.shape[0] == query_vector_np.shape[1]:
                     indexed_embeddings_for_calc = self._indexed_tool_embeddings_np.reshape(1, -1)
                else:
                     logger.error(f"{self.plugin_id}: In-memory NumPy embeddings array has unexpected shape {self._indexed_tool_embeddings_np.shape}.")
                     return []
            else:
                indexed_embeddings_for_calc = self._indexed_tool_embeddings_np

            if indexed_embeddings_for_calc.shape[1] != query_vector_np.shape[1]:
                logger.error(f"{self.plugin_id}: Query embedding dimension ({query_vector_np.shape[1]}) does not match indexed tool embedding dimension ({indexed_embeddings_for_calc.shape[1]}).")
                return []

            dot_products = np.dot(indexed_embeddings_for_calc, query_vector_np.T).flatten()
            query_norm = np.linalg.norm(query_vector_np)
            index_norms = np.linalg.norm(indexed_embeddings_for_calc, axis=1)
            epsilon = 1e-9
            denominator = (index_norms * query_norm) + epsilon
            similarities = dot_products / denominator
            similarities = np.clip(similarities, -1.0, 1.0)
            similarities = np.nan_to_num(similarities, nan=-1.0)

            num_indexed_tools = len(self._indexed_tool_data_list_np)
            actual_top_k = min(top_k, num_indexed_tools)
            if actual_top_k == 0: return []

            top_indices = np.argsort(-similarities)[:actual_top_k]

            results: List[RankedToolResult] = []
            for i in top_indices:
                if not (0 <= i < num_indexed_tools): continue
                tool_data = self._indexed_tool_data_list_np[i]
                relevance_score = (float(similarities[i]) + 1.0) / 2.0
                snippet_text = tool_data.get("lookup_text_representation", tool_data.get("description_llm", tool_data.get("name", "")))
                results.append(RankedToolResult(
                    tool_identifier=tool_data.get("identifier", f"unknown_tool_at_index_{i}"),
                    score=relevance_score,
                    matched_tool_data=tool_data,
                    description_snippet=snippet_text[:200] + ("..." if len(snippet_text) > 200 else "")
                ))
            logger.info(f"{self.plugin_id}: Found {len(results)} tools via in-memory NumPy. Top score: {results[0].score if results else 'N/A'}.")
            return results
        else:
            logger.warning(f"{self.plugin_id}: Index not built or NumPy not available. Cannot find tools using in-memory method.")
            return []

    async def teardown(self) -> None:
        # Teardown for embedder and vector_store is managed by PluginManager if they were loaded via it.
        # This provider only needs to release its direct references.
        self._embedder = None
        self._tool_vector_store = None
        self._plugin_manager = None
        self._indexed_tool_embeddings_np = None
        self._indexed_tool_data_list_np = []
        logger.debug(f"{self.plugin_id}: Torn down.")

