import logging
from typing import (
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
from genie_tooling.core.types import (
    Chunk,
    Document,
    EmbeddingVector,
    Plugin,
    RetrievedChunk,
)
from genie_tooling.document_loaders.abc import DocumentLoaderPlugin
from genie_tooling.embedding_generators.abc import EmbeddingGeneratorPlugin
from genie_tooling.retrievers.abc import RetrieverPlugin
from genie_tooling.text_splitters.abc import TextSplitterPlugin
from genie_tooling.vector_stores.abc import VectorStorePlugin

logger = logging.getLogger(__name__)

PT = TypeVar("PT", bound=Plugin)

class RAGManager:
    def __init__(self, plugin_manager: PluginManager):
        self._plugin_manager = plugin_manager
        logger.info("RAGManager initialized.")

    async def _get_plugin_instance_for_rag(
        self,
        plugin_id: str,
        expected_protocol: Type[PT],
        component_name: str,
        plugin_setup_config: Optional[Dict[str, Any]] = None,
    ) -> Optional[PT]:
        # REFACTORED: Ensure PluginManager is always passed to sub-plugins
        final_plugin_setup_config = (plugin_setup_config or {}).copy()
        final_plugin_setup_config.setdefault("plugin_manager", self._plugin_manager)

        instance_any = await self._plugin_manager.get_plugin_instance(plugin_id, config=final_plugin_setup_config)

        if instance_any and isinstance(instance_any, expected_protocol):
            logger.debug(f"RAGManager: Successfully instantiated and set up {component_name} plugin '{plugin_id}'.")
            return cast(PT, instance_any)

        if instance_any:
            logger.error(f"RAGManager: Instantiated plugin '{plugin_id}' is not a valid {component_name}. Type: {type(instance_any)}")
        else:
            logger.error(f"RAGManager: {component_name} plugin '{plugin_id}' not found or failed to load.")
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
        component_names = {"doc_loader": "DocumentLoader", "text_splitter": "TextSplitter", "embed_generator": "EmbeddingGenerator", "vec_store": "VectorStore"}
        doc_loader = await self._get_plugin_instance_for_rag(loader_id, DocumentLoaderPlugin, component_names["doc_loader"], loader_config)
        text_splitter = await self._get_plugin_instance_for_rag(splitter_id, TextSplitterPlugin, component_names["text_splitter"], splitter_config)
        embed_generator = await self._get_plugin_instance_for_rag(embedder_id, EmbeddingGeneratorPlugin, component_names["embed_generator"], embedder_config)
        vec_store = await self._get_plugin_instance_for_rag(vector_store_id, VectorStorePlugin, component_names["vec_store"], vector_store_config)
        loaded_components = {component_names["doc_loader"]: doc_loader, component_names["text_splitter"]: text_splitter, component_names["embed_generator"]: embed_generator, component_names["vec_store"]: vec_store}
        if not all(loaded_components.values()):
            missing = [name for name, inst in loaded_components.items() if not inst]; msg = f"One or more RAG components failed to load: {', '.join(missing)}."; logger.error(msg)
            return {"status": "error", "message": msg}
        doc_loader = cast(DocumentLoaderPlugin, doc_loader); text_splitter = cast(TextSplitterPlugin, text_splitter); embed_generator = cast(EmbeddingGeneratorPlugin, embed_generator); vec_store = cast(VectorStorePlugin, vec_store)
        try:
            documents: AsyncIterable[Document] = doc_loader.load(source_uri=loader_source_uri, config=loader_config)
            chunks: AsyncIterable[Chunk] = text_splitter.split(documents=documents, config=splitter_config)
            chunk_embeddings: AsyncIterable[tuple[Chunk, EmbeddingVector]] = embed_generator.embed(chunks=chunks, config=embedder_config)
            add_result = await vec_store.add(embeddings=chunk_embeddings, config=vector_store_config)
            added_count_from_store = add_result.get("added_count", "unknown (store did not report)"); store_errors = add_result.get("errors", [])
            if store_errors: logger.warning(f"Errors encountered during vector store add: {store_errors}")
            msg = f"Data source '{loader_source_uri}' indexed into '{vector_store_id}'. Added count from store: {added_count_from_store}."
            logger.info(f"Successfully indexed data. {msg}")
            return {"status": "success", "message": msg, "added_count": added_count_from_store, "store_errors": store_errors}
        except Exception as e:
            logger.error(f"Error during RAG indexing pipeline for source '{loader_source_uri}': {e}", exc_info=True)
            return {"status": "error", "message": f"Indexing failed: {str(e)}"}

    async def retrieve_from_query(self, query_text: str, retriever_id: str, retriever_config: Optional[Dict[str, Any]] = None, top_k: int = 5) -> List[RetrievedChunk]:
        logger.info(f"Attempting retrieval for query: '{query_text[:100]}...' using retriever '{retriever_id}'.")
        # REFACTORED: The _get_plugin_instance_for_rag helper now handles injecting the PluginManager
        retriever_plugin = await self._get_plugin_instance_for_rag(retriever_id, RetrieverPlugin, "Retriever", retriever_config)
        if not retriever_plugin: logger.error(f"Retriever plugin '{retriever_id}' not found or failed to load."); return []
        retriever_plugin = cast(RetrieverPlugin, retriever_plugin)
        try:
            runtime_retrieve_config = retriever_config or {}
            results = await retriever_plugin.retrieve(query=query_text, top_k=top_k, config=runtime_retrieve_config)
            logger.info(f"Retrieved {len(results)} chunks for query using retriever '{retriever_id}'.")
            return results
        except Exception as e:
            logger.error(f"Error during RAG retrieval with retriever '{retriever_id}': {e}", exc_info=True)
            return []
