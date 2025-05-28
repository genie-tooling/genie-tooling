### src/genie_tooling/lookup/providers/impl/embedding_similarity.py
"""EmbeddingSimilarityLookupProvider: Finds tools using semantic similarity of embeddings."""
import logging
from typing import Any, AsyncIterable, Dict, List, Optional, cast  # Added Type

logger = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:
    np = None
    logger.warning("NumPy not found. EmbeddingSimilarityLookupProvider may not function optimally or at all.")


from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import Chunk
# Updated import path for EmbeddingGeneratorPlugin
from genie_tooling.embedding_generators.abc import EmbeddingGeneratorPlugin
# Updated import path for ToolLookupProvider
from genie_tooling.tool_lookup_providers.abc import ToolLookupProvider
# RankedToolResult is expected to remain in genie_tooling.lookup.types for now
from genie_tooling.lookup.types import RankedToolResult


class _LookupQueryChunk: # Simple ad-hoc class for query text
    def __init__(self, query_text: str):
        self.content: str = query_text
        self.metadata: Dict[str, Any] = {"source": "lookup_query"}
        self.id: Optional[str] = f"lookup_query_{hash(query_text)}" # Simple ID

class EmbeddingSimilarityLookupProvider(ToolLookupProvider):
    plugin_id: str = "embedding_similarity_lookup_v1"
    description: str = "Finds tools by comparing query embedding with tool description embeddings using cosine similarity."

    _indexed_tool_embeddings: Optional[Any] = None # np.ndarray
    _indexed_tool_data_list: List[Dict[str, Any]] = []

    _embedder: Optional[EmbeddingGeneratorPlugin] = None
    _plugin_manager: Optional[PluginManager] = None

    DEFAULT_EMBEDDER_ID = "sentence_transformer_embedder_v1" # Ensure this plugin exists

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        if not np:
            logger.error(f"{self.plugin_id} Error: 'numpy' library not installed. This provider cannot function.")
            return

        cfg = config or {}
        self._plugin_manager = cfg.get("plugin_manager")
        if not self._plugin_manager or not isinstance(self._plugin_manager, PluginManager):
            logger.error(f"{self.plugin_id} Error: PluginManager not provided in config. Cannot load embedder.")
            return

        embedder_id_to_load = cfg.get("embedder_id", self.DEFAULT_EMBEDDER_ID)
        embedder_setup_config = cfg.get("embedder_config", {})

        logger.info(f"{self.plugin_id}: Setting up with embedder '{embedder_id_to_load}'.")

        embedder_instance_any = await self._plugin_manager.get_plugin_instance(embedder_id_to_load, config=embedder_setup_config)
        if embedder_instance_any and isinstance(embedder_instance_any, EmbeddingGeneratorPlugin):
            self._embedder = cast(EmbeddingGeneratorPlugin, embedder_instance_any)
            logger.debug(f"{self.plugin_id}: Embedder '{embedder_id_to_load}' loaded and set up.")
        else:
            logger.error(f"{self.plugin_id} Error: Embedder plugin '{embedder_id_to_load}' not found, invalid, or failed setup.")
            self._embedder = None

    async def index_tools(self, tools_data: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None) -> None:
        if not self._embedder:
            logger.error(f"{self.plugin_id}: Embedder not available. Cannot index tools.")
            self._indexed_tool_embeddings = None
            self._indexed_tool_data_list = []
            return
        if not np:
            logger.error(f"{self.plugin_id}: NumPy not available. Cannot process embeddings for indexing.")
            return

        texts_to_embed: List[str] = []
        valid_tools_data_for_index: List[Dict[str, Any]] = []

        for tool_item_data in tools_data:
            lookup_text = tool_item_data.get("lookup_text_representation") # Expected from formatter
            identifier = tool_item_data.get("identifier")

            if isinstance(lookup_text, str) and lookup_text.strip() and identifier:
                texts_to_embed.append(lookup_text)
                valid_tools_data_for_index.append(tool_item_data)
            else:
                logger.warning(f"{self.plugin_id}: Tool data for ID '{identifier}' missing 'lookup_text_representation' or identifier. Skipping from index.")

        if not texts_to_embed:
            logger.info(f"{self.plugin_id}: No valid tool texts provided to embed for indexing. Index will be empty.")
            self._indexed_tool_embeddings = np.array([])
            self._indexed_tool_data_list = []
            return

        async def _texts_as_temp_chunks_for_indexing() -> AsyncIterable[Chunk]:
            for _i, text_content in enumerate(texts_to_embed):
                yield cast(Chunk, _LookupQueryChunk(query_text=text_content)) # Reuse _LookupQueryChunk for simplicity

        all_embeddings_list: List[List[float]] = []
        # embedder_runtime_config = (config or {}).get("embedder_config", {}) # Config for embed method
        embedder_runtime_config = (config or {}).get("embedder_config", (config or {}).get("embedder_runtime_config", {}))


        try:
            logger.debug(f"{self.plugin_id}: Embedding {len(texts_to_embed)} tool texts for index...")
            async for _chunk_obj, vector_as_list in self._embedder.embed(chunks=_texts_as_temp_chunks_for_indexing(), config=embedder_runtime_config):
                if vector_as_list and isinstance(vector_as_list, list) and all(isinstance(x, float) for x in vector_as_list):
                    all_embeddings_list.append(vector_as_list)
                else:
                    # This implies an error or an empty embedding for one of the texts.
                    # To maintain alignment, we must either skip the corresponding tool_data or insert a placeholder.
                    # For V1, let's log and effectively skip by not adding to all_embeddings_list, which will cause a mismatch handled below.
                    logger.warning(f"{self.plugin_id}: Received invalid or empty embedding for a tool text during indexing. This tool text may be excluded or cause misalignment.")

            # Crucial alignment check
            if len(all_embeddings_list) != len(valid_tools_data_for_index):
                 # This is problematic. For simplicity, if there's a mismatch because some embeddings failed (returned empty),
                 # we cannot reliably map embeddings back to original tool_data.
                 # A more robust embedder should yield (Chunk, Optional[EmbeddingVector]) to allow tracking.
                 # For now, if counts mismatch, the index is considered faulty.
                 logger.error(f"{self.plugin_id}: Mismatch between number of generated embeddings ({len(all_embeddings_list)}) "
                                f"and valid tool data items ({len(valid_tools_data_for_index)}). Indexing aborted to prevent misalignment.")
                 self._indexed_tool_embeddings = np.array([])
                 self._indexed_tool_data_list = []
                 return

        except Exception as e_embed_idx:
            logger.error(f"{self.plugin_id}: Error during tool text embedding for index: {e_embed_idx}") # Removed exc_info=True
            self._indexed_tool_embeddings = np.array([])
            self._indexed_tool_data_list = []
            return

        if not all_embeddings_list:
            self._indexed_tool_embeddings = np.array([])
        else:
            try:
                self._indexed_tool_embeddings = np.array(all_embeddings_list, dtype=np.float32)
            except ValueError as e_np:
                logger.error(f"{self.plugin_id}: Failed to convert embeddings to NumPy array: {e_np}. Likely inconsistent embedding dimensions.", exc_info=True)
                self._indexed_tool_embeddings = np.array([])
                valid_tools_data_for_index = [] # Cannot use data if embeddings are bad

        self._indexed_tool_data_list = valid_tools_data_for_index
        logger.info(f"{self.plugin_id}: Successfully indexed {len(self._indexed_tool_data_list)} tools.")

    async def find_tools(
        self,
        natural_language_query: str,
        top_k: int = 5,
        config: Optional[Dict[str, Any]] = None
    ) -> List[RankedToolResult]:
        if not natural_language_query or not natural_language_query.strip():
            logger.debug(f"{self.plugin_id}: Empty query provided. Returning no results.")
            return []

        if self._indexed_tool_embeddings is None or not hasattr(self._indexed_tool_embeddings, "size") or self._indexed_tool_embeddings.size == 0 or not self._embedder or not np:
            logger.warning(f"{self.plugin_id}: Index not built, embedder unavailable, or NumPy missing. Cannot find tools.")
            return []

        async def _query_as_chunk_stream_for_find() -> AsyncIterable[Chunk]:
            yield cast(Chunk, _LookupQueryChunk(query_text=natural_language_query))

        query_embedding_list: Optional[List[float]] = None
        embedder_runtime_config = (config or {}).get("embedder_config", (config or {}).get("embedder_runtime_config", {}))
        try:
            logger.debug(f"{self.plugin_id}: Embedding lookup query: '{natural_language_query[:100]}...'")
            async for _chunk_obj, vector_as_list in self._embedder.embed(chunks=_query_as_chunk_stream_for_find(), config=embedder_runtime_config):
                if vector_as_list and isinstance(vector_as_list, list) and all(isinstance(x, float) for x in vector_as_list) :
                    query_embedding_list = vector_as_list
                break
        except Exception as e_embed_q:
            logger.error(f"{self.plugin_id}: Error embedding lookup query: {e_embed_q}") # Removed exc_info=True
            return []

        if not query_embedding_list:
            logger.error(f"{self.plugin_id}: Failed to generate embedding for the lookup query.")
            return []

        query_vector_np = np.array(query_embedding_list, dtype=np.float32).reshape(1, -1)

        # Cosine Similarity: (A . B) / (||A|| * ||B||)
        # Ensure indexed embeddings are 2D array for dot product with 2D query_vector_np
        if self._indexed_tool_embeddings.ndim == 1: # Should be 2D (n_tools, dim)
            # This case implies only one tool was indexed or an issue with array creation.
            # For a single indexed vector, reshape it to (1, dim)
            if self._indexed_tool_embeddings.shape[0] == query_vector_np.shape[1]: # Check if it's a single vector of correct dim
                 indexed_embeddings_for_calc = self._indexed_tool_embeddings.reshape(1, -1)
            else:
                 logger.error(f"{self.plugin_id}: Indexed embeddings array has unexpected shape {self._indexed_tool_embeddings.shape}.")
                 return []
        else:
            indexed_embeddings_for_calc = self._indexed_tool_embeddings


        dot_products = np.dot(indexed_embeddings_for_calc, query_vector_np.T).flatten()
        query_norm = np.linalg.norm(query_vector_np)
        index_norms = np.linalg.norm(indexed_embeddings_for_calc, axis=1)

        epsilon = 1e-9 # To avoid division by zero
        denominator = (index_norms * query_norm) + epsilon

        similarities = dot_products / denominator
        # Clip to handle potential floating point inaccuracies leading to values slightly outside [-1, 1]
        similarities = np.clip(similarities, -1.0, 1.0)
        similarities = np.nan_to_num(similarities, nan=-1.0) # Replace NaN with a low score

        num_indexed_tools = len(self._indexed_tool_data_list)
        actual_top_k = min(top_k, num_indexed_tools)

        if actual_top_k == 0: return []

        top_indices = np.argsort(-similarities)[:actual_top_k] # Negative for descending sort

        results: List[RankedToolResult] = []
        for i in top_indices:
            if not (0 <= i < num_indexed_tools): continue

            tool_data = self._indexed_tool_data_list[i]
            cosine_score = float(similarities[i])
            # Map cosine similarity [-1, 1] to relevance score [0, 1]
            relevance_score = (cosine_score + 1.0) / 2.0

            # Snippet from the text that was embedded for this tool
            snippet_text = tool_data.get("lookup_text_representation", tool_data.get("description_llm", tool_data.get("name", "")))

            results.append(
                RankedToolResult(
                    tool_identifier=tool_data.get("identifier", f"unknown_tool_at_index_{i}"),
                    score=relevance_score,
                    matched_tool_data=tool_data,
                    description_snippet=snippet_text[:200] + ("..." if len(snippet_text) > 200 else "")
                )
            )
        logger.info(f"{self.plugin_id}: Found {len(results)} tools for query. Top score: {results[0].score if results else 'N/A'}.")
        return results

    async def teardown(self) -> None:
        if self._embedder and hasattr(self._embedder, "teardown"):
            try:
                await self._embedder.teardown()
            except Exception as e_td:
                logger.error(f"{self.plugin_id}: Error tearing down embedder: {e_td}", exc_info=True)
        self._indexed_tool_embeddings = None
        self._indexed_tool_data_list = []
        self._embedder = None
        self._plugin_manager = None
        logger.debug(f"{self.plugin_id}: Torn down.")
