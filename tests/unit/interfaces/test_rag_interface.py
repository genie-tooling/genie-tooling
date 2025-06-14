### tests/unit/interfaces/test_rag_interface.py
from typing import List
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest
from genie_tooling.config.models import MiddlewareConfig
from genie_tooling.core.types import RetrievedChunk
from genie_tooling.interfaces import RAGInterface
from genie_tooling.observability.manager import InteractionTracingManager
from genie_tooling.rag.manager import RAGManager
from genie_tooling.security.key_provider import KeyProvider


@pytest.fixture()
def mock_rag_manager() -> MagicMock:
    """Mocks the RAGManager."""
    mgr = AsyncMock(spec=RAGManager)
    mgr.index_data_source = AsyncMock(return_value={"status": "success", "added_count": 0})
    mgr.retrieve_from_query = AsyncMock(return_value=[])
    return mgr

@pytest.fixture()
def mock_middleware_config_for_rag() -> MagicMock:
    """Mocks MiddlewareConfig, focusing on RAG-related default IDs."""
    cfg = MagicMock(spec=MiddlewareConfig)
    cfg.default_rag_loader_id = "default_loader_id"
    cfg.default_rag_splitter_id = "default_splitter_id"
    cfg.default_rag_embedder_id = "default_embedder_id"
    cfg.default_rag_vector_store_id = "default_vector_store_id"
    cfg.default_rag_retriever_id = "default_retriever_id"
    # Initialize configuration dictionaries
    cfg.document_loader_configurations = {}
    cfg.text_splitter_configurations = {}
    cfg.embedding_generator_configurations = {}
    cfg.vector_store_configurations = {}
    cfg.retriever_configurations = {}
    return cfg

@pytest.fixture()
def mock_key_provider_for_rag() -> MagicMock:
    """Mocks the KeyProvider."""
    return MagicMock(spec=KeyProvider)

@pytest.fixture()
def mock_tracing_manager_for_rag_if() -> AsyncMock: # Changed to AsyncMock
    """Mocks the InteractionTracingManager for RAGInterface tests."""
    mgr = AsyncMock(spec=InteractionTracingManager) # Use AsyncMock for the manager
    mgr.trace_event = AsyncMock()
    return mgr

@pytest.fixture()
def rag_interface(
    mock_rag_manager: MagicMock,
    mock_middleware_config_for_rag: MagicMock,
    mock_key_provider_for_rag: MagicMock,
    mock_tracing_manager_for_rag_if: AsyncMock, # Updated type hint
) -> RAGInterface:
    """Provides a RAGInterface instance with mocked dependencies."""
    return RAGInterface(
        rag_manager=mock_rag_manager,
        config=mock_middleware_config_for_rag,
        key_provider=mock_key_provider_for_rag,
        tracing_manager=mock_tracing_manager_for_rag_if,
    )

@pytest.mark.asyncio()
class TestRAGInterfaceIndexDirectory:
    """Tests for RAGInterface.index_directory() method."""

    async def test_index_directory_success_with_defaults(
        self, rag_interface: RAGInterface, mock_rag_manager: MagicMock, mock_middleware_config_for_rag: MagicMock, mock_key_provider_for_rag: MagicMock
    ):
        """Test successful directory indexing using default component IDs."""
        path = "./test_docs"
        collection = "my_collection"
        mock_rag_manager.index_data_source = AsyncMock(return_value={"status": "success", "added_count": 10})

        result = await rag_interface.index_directory(path, collection_name=collection)

        assert result["status"] == "success"
        mock_rag_manager.index_data_source.assert_awaited_once_with(
            loader_id=mock_middleware_config_for_rag.default_rag_loader_id,
            loader_source_uri=path,
            splitter_id=mock_middleware_config_for_rag.default_rag_splitter_id,
            embedder_id=mock_middleware_config_for_rag.default_rag_embedder_id,
            vector_store_id=mock_middleware_config_for_rag.default_rag_vector_store_id,
            loader_config={},
            splitter_config={},
            embedder_config={"key_provider": mock_key_provider_for_rag},
            vector_store_config={"collection_name": collection}
        )
        rag_interface._tracing_manager.trace_event.assert_any_call("rag.index_directory.start", ANY, "RAGInterface", ANY) # type: ignore
        rag_interface._tracing_manager.trace_event.assert_any_call("rag.index_directory.end", ANY, "RAGInterface", ANY) # type: ignore

    async def test_index_directory_with_overrides(
        self, rag_interface: RAGInterface, mock_rag_manager: MagicMock, mock_key_provider_for_rag: MagicMock
    ):
        """Test directory indexing with overridden component IDs and configs."""
        mock_rag_manager.index_data_source = AsyncMock(return_value={"status": "success"})
        await rag_interface.index_directory(
            path="./data", collection_name="override_coll",
            loader_id="custom_loader", splitter_id="custom_splitter",
            embedder_id="custom_embedder", vector_store_id="custom_vs",
            loader_config={"lc_key": "lc_val"}, splitter_config={"sc_key": "sc_val"},
            embedder_config={"ec_key": "ec_val"}, vector_store_config={"vsc_key": "vsc_val"}
        )
        mock_rag_manager.index_data_source.assert_awaited_once_with(
            loader_id="custom_loader", loader_source_uri="./data",
            splitter_id="custom_splitter", embedder_id="custom_embedder",
            vector_store_id="custom_vs",
            loader_config={"lc_key": "lc_val"},
            splitter_config={"sc_key": "sc_val"},
            embedder_config={"ec_key": "ec_val", "key_provider": mock_key_provider_for_rag},
            vector_store_config={"vsc_key": "vsc_val", "collection_name": "override_coll"}
        )

    async def test_index_directory_missing_embedder_id_raises_error(self, rag_interface: RAGInterface, mock_middleware_config_for_rag: MagicMock):
        """Test error if embedder ID cannot be resolved."""
        mock_middleware_config_for_rag.default_rag_embedder_id = None
        with pytest.raises(ValueError, match="RAG embedder ID not resolved for index_directory."):
            await rag_interface.index_directory("./docs")

    async def test_index_directory_missing_vector_store_id_raises_error(self, rag_interface: RAGInterface, mock_middleware_config_for_rag: MagicMock):
        """Test error if vector store ID cannot be resolved."""
        mock_middleware_config_for_rag.default_rag_vector_store_id = None
        with pytest.raises(ValueError, match="RAG vector store ID not resolved for index_directory."):
            await rag_interface.index_directory("./docs")

    async def test_index_directory_rag_manager_failure(
        self, rag_interface: RAGInterface, mock_rag_manager: MagicMock
    ):
        """Test handling of failure from the RAGManager during indexing."""
        mock_rag_manager.index_data_source = AsyncMock(side_effect=RuntimeError("Indexing pipeline failed"))
        with pytest.raises(RuntimeError, match="Indexing pipeline failed"):
            await rag_interface.index_directory("./data")


@pytest.mark.asyncio()
class TestRAGInterfaceIndexWebPage:
    """Tests for RAGInterface.index_web_page() method."""
    async def test_index_web_page_success_with_defaults(
        self, rag_interface: RAGInterface, mock_rag_manager: MagicMock, mock_middleware_config_for_rag: MagicMock, mock_key_provider_for_rag: MagicMock
    ):
        url = "http://example.com"
        collection = "web_collection"
        mock_rag_manager.index_data_source = AsyncMock(return_value={"status": "success", "added_count": 1})

        result = await rag_interface.index_web_page(url, collection_name=collection)
        assert result["status"] == "success"
        mock_rag_manager.index_data_source.assert_awaited_once_with(
            loader_id="web_page_loader_v1", # Corrected: Uses the hardcoded default for web pages
            loader_source_uri=url,
            splitter_id=mock_middleware_config_for_rag.default_rag_splitter_id,
            embedder_id=mock_middleware_config_for_rag.default_rag_embedder_id,
            vector_store_id=mock_middleware_config_for_rag.default_rag_vector_store_id,
            loader_config={},
            splitter_config={},
            embedder_config={"key_provider": mock_key_provider_for_rag},
            vector_store_config={"collection_name": collection}
        )
        rag_interface._tracing_manager.trace_event.assert_any_call("rag.index_web_page.start", ANY, "RAGInterface", ANY) # type: ignore
        rag_interface._tracing_manager.trace_event.assert_any_call("rag.index_web_page.end", ANY, "RAGInterface", ANY) # type: ignore

@pytest.mark.asyncio()
class TestRAGInterfaceSearch:
    """Tests for RAGInterface.search() method."""

    async def test_search_success_with_defaults(
        self, rag_interface: RAGInterface, mock_rag_manager: MagicMock, mock_middleware_config_for_rag: MagicMock, mock_key_provider_for_rag: MagicMock
    ):
        """Test successful search using default retriever and its components."""
        query = "What is Genie?"
        collection = "search_collection"
        mock_chunks: List[RetrievedChunk] = [{"id":"res1", "content":"Genie is cool.", "score":0.9, "metadata":{}}] # type: ignore
        mock_rag_manager.retrieve_from_query = AsyncMock(return_value=mock_chunks)

        results = await rag_interface.search(query, collection_name=collection, top_k=3)

        assert results == mock_chunks
        # FIX: The test must assert the retriever_config that the RAGInterface *actually* builds.
        mock_rag_manager.retrieve_from_query.assert_awaited_once_with(
            query_text=query,
            retriever_id=mock_middleware_config_for_rag.default_rag_retriever_id,
            retriever_config={
                "embedder_id": mock_middleware_config_for_rag.default_rag_embedder_id,
                "vector_store_id": mock_middleware_config_for_rag.default_rag_vector_store_id,
                "embedder_config": {"key_provider": mock_key_provider_for_rag},
                "vector_store_config": {"collection_name": collection}
            },
            top_k=3
        )
        rag_interface._tracing_manager.trace_event.assert_any_call("rag.search.start", ANY, "RAGInterface", ANY) # type: ignore
        rag_interface._tracing_manager.trace_event.assert_any_call("rag.search.end", ANY, "RAGInterface", ANY) # type: ignore

    async def test_search_with_overrides(
        self, rag_interface: RAGInterface, mock_rag_manager: MagicMock, mock_key_provider_for_rag: MagicMock
    ):
        """Test search with overridden retriever ID and config."""
        mock_rag_manager.retrieve_from_query = AsyncMock(return_value=[])
        await rag_interface.search(
            "query", collection_name="coll", top_k=10,
            retriever_id="custom_retriever",
            retriever_config={"rc_key": "rc_val", "embedder_config": {"ec_key": "ec_val"}}
        )
        # FIX: The test must assert the retriever_config that the RAGInterface *actually* builds.
        mock_rag_manager.retrieve_from_query.assert_awaited_once_with(
            query_text="query",
            retriever_id="custom_retriever",
            retriever_config={
                "rc_key": "rc_val",
                "embedder_id": rag_interface._config.default_rag_embedder_id,
                "vector_store_id": rag_interface._config.default_rag_vector_store_id,
                "embedder_config": {"ec_key": "ec_val", "key_provider": mock_key_provider_for_rag},
                "vector_store_config": {"collection_name": "coll"}
            },
            top_k=10
        )