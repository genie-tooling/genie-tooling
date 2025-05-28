"""RAGManager: Orchestrates RAG pipelines using pluggable components."""
import logging
from typing import (  # Added TypeVar
    Any,
    AsyncIterable,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    cast,
)

from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import (  # Removed direct PluginType import
    Chunk,
    Document,
    EmbeddingVector,
    Plugin,
    RetrievedChunk,
)

from .plugins.abc import (
    DocumentLoaderPlugin,
    EmbeddingGeneratorPlugin,
    RetrieverPlugin,
    TextSplitterPlugin,
    VectorStorePlugin,
)

logger = logging.getLogger(__name__)

# Define PT bound to the Plugin protocol for use in _get_configured_plugin
PT = TypeVar("PT", bound=Plugin)

class RAGManager:
    """
    Manages RAG pipelines including data ingestion (load, split, embed, store)
    and retrieval. It uses the PluginManager to load and instantiate necessary
    RAG component plugins.
    """
    def __init__(self, plugin_manager: PluginManager):
        self._plugin_manager = plugin_manager
        logger.info("RAGManager initialized.")

    async def _get_configured_plugin(
        self,
        plugin_id: str,
        plugin_protocol: Type[PT], # Use PT here
        component_name: str,
        plugin_config: Optional[Dict[str, Any]] = None,
    ) -> Optional[PT]: # Return Optional[PT]
        """Helper to get, instantiate, and setup a RAG plugin component."""

        instance_any = await self._plugin_manager.get_plugin_instance(plugin_id, config=plugin_config)

        if instance_any and isinstance(instance_any, plugin_protocol):
            logger.debug(f"RAGManager: Successfully loaded {component_name} plugin '{plugin_id}' matching expected type.")
            return cast(PT, instance_any) # Cast to PT
        elif instance_any:
            logger.error(f"RAGManager: Plugin '{plugin_id}' loaded but is not a valid {component_name} "
                         f"(expected type compatible with {plugin_protocol.__name__}, got {type(instance_any).__name__}).")
        else:
            logger.error(f"RAGManager: {component_name} plugin '{plugin_id}' could not be loaded.")
        return None

    async def index_data_source(
        self,
        loader_id: str,
        loader_source_uri: str,
        splitter_id: str,
        embedder_id: str,
        vector_store_id: str,
        loader_config: Optional[Dict[str, Any]] = None,
        splitter_config: Optional[Dict[str, Any]] = None,
        embedder_config: Optional[Dict[str, Any]] = None,
        vector_store_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        logger.info(f"Starting RAG indexing for source '{loader_source_uri}' using pipeline: "
                    f"Loader='{loader_id}', Splitter='{splitter_id}', Embedder='{embedder_id}', Store='{vector_store_id}'.")

        doc_loader = await self._get_configured_plugin(loader_id, DocumentLoaderPlugin, "DocumentLoader", loader_config)
        text_splitter = await self._get_configured_plugin(splitter_id, TextSplitterPlugin, "TextSplitter", splitter_config)
        embed_generator = await self._get_configured_plugin(embedder_id, EmbeddingGeneratorPlugin, "EmbeddingGenerator", embedder_config)
        vec_store = await self._get_configured_plugin(vector_store_id, VectorStorePlugin, "VectorStore", vector_store_config)

        if not all([doc_loader, text_splitter, embed_generator, vec_store]):
            missing = [name for inst, name in [(doc_loader, "Loader"), (text_splitter, "Splitter"), (embed_generator, "Embedder"), (vec_store, "Store")] if not inst]
            msg = f"One or more RAG components failed to load: {', '.join(missing)}."
            logger.error(msg)
            return {"status": "error", "message": msg}

        doc_loader = cast(DocumentLoaderPlugin, doc_loader)
        text_splitter = cast(TextSplitterPlugin, text_splitter)
        embed_generator = cast(EmbeddingGeneratorPlugin, embed_generator)
        vec_store = cast(VectorStorePlugin, vec_store)

        try:
            logger.debug(f"Loading documents from '{loader_source_uri}' with loader '{loader_id}'.")
            documents: AsyncIterable[Document] = doc_loader.load(source_uri=loader_source_uri, config=loader_config)

            logger.debug(f"Splitting documents with splitter '{splitter_id}'.")
            chunks: AsyncIterable[Chunk] = text_splitter.split(documents=documents, config=splitter_config)

            logger.debug(f"Generating embeddings with embedder '{embedder_id}'.")
            chunk_embeddings: AsyncIterable[tuple[Chunk, EmbeddingVector]] = embed_generator.embed(chunks=chunks, config=embedder_config)

            logger.debug(f"Storing embeddings in vector store '{vector_store_id}'.")
            add_result = await vec_store.add(embeddings=chunk_embeddings, config=vector_store_config)

            added_count_from_store = add_result.get("added_count", "unknown (store did not report)")
            store_errors = add_result.get("errors", [])

            if store_errors: logger.warning(f"Errors encountered during vector store add: {store_errors}")

            msg = f"Data source '{loader_source_uri}' indexed into '{vector_store_id}'. Added count from store: {added_count_from_store}."
            logger.info(f"Successfully indexed data. {msg}")
            return {"status": "success", "message": msg, "added_count": added_count_from_store, "store_errors": store_errors}

        except Exception as e:
            logger.error(f"Error during RAG indexing pipeline for source '{loader_source_uri}': {e}", exc_info=True)
            return {"status": "error", "message": f"Indexing failed: {str(e)}"}

    async def retrieve_from_query(
        self,
        query_text: str,
        retriever_id: str,
        retriever_config: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[RetrievedChunk]:
        logger.info(f"Attempting retrieval for query: '{query_text[:100]}...' using retriever '{retriever_id}'.")

        final_retriever_config = {"plugin_manager": self._plugin_manager, **(retriever_config or {})}
        retriever_plugin = await self._get_configured_plugin(retriever_id, RetrieverPlugin, "Retriever", final_retriever_config)

        if not retriever_plugin:
            logger.error(f"Retriever plugin '{retriever_id}' not found or failed to load.")
            return []

        retriever_plugin = cast(RetrieverPlugin, retriever_plugin)

        try:
            results = await retriever_plugin.retrieve(query=query_text, top_k=top_k, config=final_retriever_config)
            logger.info(f"Retrieved {len(results)} chunks for query using retriever '{retriever_id}'.")
            return results
        except Exception as e:
            logger.error(f"Error during RAG retrieval with retriever '{retriever_id}': {e}", exc_info=True)
            return []
