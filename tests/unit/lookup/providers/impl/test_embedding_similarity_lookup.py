### tests/unit/lookup/providers/impl/test_embedding_similarity_lookup.py
import logging
from typing import Any, AsyncIterable, AsyncGenerator, Dict, List, Optional, Tuple, cast
from unittest.mock import AsyncMock, patch

import pytest

try:
    import numpy as np
except ImportError:
    np = None

from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import Chunk, EmbeddingVector, Plugin
from genie_tooling.core.types import RetrievedChunk as CoreRetrievedChunk
from genie_tooling.embedding_generators.abc import EmbeddingGeneratorPlugin
from genie_tooling.lookup.types import RankedToolResult
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tool_lookup_providers.impl.embedding_similarity import (
    EmbeddingSimilarityLookupProvider,
)
from genie_tooling.vector_stores.abc import (
    VectorStorePlugin,
)

PROVIDER_LOGGER_NAME = "genie_tooling.tool_lookup_providers.impl.embedding_similarity"

# Concrete implementation for testing
class _TestRetrievedChunk(CoreRetrievedChunk):
    def __init__(self, id: str, content: str, score: float, metadata: Optional[Dict[str, Any]] = None, rank: Optional[int] = None):
        self.id = id
        self.content = content
        self.score = score
        self.metadata = metadata or {}
        self.rank = rank


class MockEmbedderForLookup(EmbeddingGeneratorPlugin, Plugin):
    _plugin_id_value: str
    _fixed_embedding: Optional[List[float]] = None
    _embeddings_map: Optional[Dict[str, List[float]]] = None
    _fail_on_embed: bool = False
    teardown_called: bool = False
    setup_config_received: Optional[Dict[str, Any]] = None
    last_embed_config_received: Optional[Dict[str, Any]] = None

    def __init__(self, plugin_id_val: str = "mock_embedder_for_lookup_v1"):
        self._plugin_id_value = plugin_id_val
        self.description: str = "Mock embedder"
    @property
    def plugin_id(self) -> str:
        return self._plugin_id_value
    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.setup_config_received = config
        self.teardown_called = False
        self.last_embed_config_received = None
    def set_fixed_embedding(self, e: List[float]):
        self._fixed_embedding=e
        self._embeddings_map=None
    def set_embeddings_map(self, m: Dict[str,List[float]]):
        self._embeddings_map=m
        self._fixed_embedding=None
    def set_fail_on_embed(self, f: bool):
        self._fail_on_embed = f
    async def embed(self, chunks: AsyncIterable[Chunk], config: Optional[Dict[str,Any]]=None) -> AsyncIterable[tuple[Chunk,EmbeddingVector]]:
        self.last_embed_config_received = config
        if self._fail_on_embed:
            raise RuntimeError("Simulated embedder failure")
        async for chunk_item in chunks:
            if self._embeddings_map and chunk_item.content in self._embeddings_map:
                yield chunk_item, self._embeddings_map[chunk_item.content]
            elif self._fixed_embedding:
                yield chunk_item, self._fixed_embedding
            else:
                yield chunk_item, [0.1] * (config or {}).get("expected_dim",3)
    async def teardown(self) -> None:
        self.teardown_called = True

class MockVectorStoreForLookup(VectorStorePlugin, Plugin):
    plugin_id: str = "mock_vs_for_lookup_v1"
    description: str = "Mock VS"
    setup_config_received: Optional[Dict[str, Any]] = None
    add_called_with_config: Optional[Dict[str, Any]] = None
    search_called_with_config: Optional[Dict[str, Any]] = None
    items_added: List[Tuple[Chunk, EmbeddingVector]]
    search_results_to_return: List[CoreRetrievedChunk]
    add_should_fail: bool
    search_should_fail: bool

    def __init__(self): # Initialize instance variables here
        self.items_added = []
        self.search_results_to_return = []
        self.add_should_fail = False
        self.search_should_fail = False

    async def setup(self, config: Optional[Dict[str, Any]] = None):
        self.setup_config_received = config
        # DO NOT RESET add_should_fail or search_should_fail or search_results_to_return here.
        # Let tests control these flags and data.
        # self.items_added = [] # Only reset if test logic requires it per setup
        # self.search_results_to_return = [] # Only reset if test logic requires it per setup
        # self.add_should_fail = False # REMOVED
        # self.search_should_fail = False # REMOVED

    async def add(self, embeddings: AsyncIterable[Tuple[Chunk, EmbeddingVector]], config: Optional[Dict[str, Any]] = None):
        self.add_called_with_config = config
        if self.add_should_fail:
            raise RuntimeError("Simulated VS add failure")
        count = 0
        current_items_in_batch: List[Tuple[Chunk, EmbeddingVector]] = []
        async for item in embeddings:
            current_items_in_batch.append(item)
            count+=1
        self.items_added.extend(current_items_in_batch) # Add them at the end of processing the iterable
        return {"added_count":count, "errors": []}

    async def search(self, qe: EmbeddingVector, tk: int, fm: Optional[Dict[str,Any]]=None, config: Optional[Dict[str,Any]]=None):
        self.search_called_with_config = config
        if self.search_should_fail:
            raise RuntimeError("Simulated VS search failure")
        return self.search_results_to_return[:tk]

    def set_search_results(self, results: List[CoreRetrievedChunk]):
        self.search_results_to_return = results

    async def delete(self, ids=None, fm=None, da=False, config=None):
        return True
    async def teardown(self):
        pass


@pytest.fixture
def mock_plugin_manager_for_es_lookup(mocker) -> PluginManager:
    pm = mocker.MagicMock(spec=PluginManager)
    pm.get_plugin_instance = AsyncMock()
    pm.list_discovered_plugin_classes = mocker.MagicMock(return_value={})
    return pm

@pytest.fixture
def mock_key_provider_for_es_lookup(mocker) -> KeyProvider:
    kp = mocker.AsyncMock(spec=KeyProvider)
    kp.get_key = AsyncMock(return_value="mock_api_key_for_es_lookup")
    return kp

@pytest.fixture
async def es_lookup_provider(mock_plugin_manager_for_es_lookup: PluginManager, mock_key_provider_for_es_lookup: KeyProvider) -> AsyncGenerator[EmbeddingSimilarityLookupProvider, None]:
    provider_instance = EmbeddingSimilarityLookupProvider()
    mock_embedder_instance = MockEmbedderForLookup(plugin_id_val=EmbeddingSimilarityLookupProvider.DEFAULT_EMBEDDER_ID)

    async def get_instance_side_effect(plugin_id_req, config=None, **kwargs):
        if plugin_id_req == EmbeddingSimilarityLookupProvider.DEFAULT_EMBEDDER_ID:
            await mock_embedder_instance.setup(config)
            return mock_embedder_instance
        if plugin_id_req == "mock_vs_for_lookup_v1":
            # Create a new instance of MockVectorStoreForLookup each time it's requested by this ID
            # This ensures that flags like add_should_fail are not persisted across different test setups
            # if the same plugin manager mock is reused.
            # However, for tests where we pre-configure the mock_vs, we need to ensure *that* instance is returned.
            # The current structure of the failing tests defines mock_vs locally, so this generic path
            # might not be hit for "mock_vs_for_lookup_v1" if the test's local side_effect takes precedence.
            # For safety, let's assume tests will set a more specific side_effect if they need a pre-configured mock_vs.
            mock_vs = MockVectorStoreForLookup()
            await mock_vs.setup(config)
            return mock_vs
        generic_mock_plugin = AsyncMock(spec=Plugin)
        generic_mock_plugin.plugin_id = plugin_id_req # type: ignore
        if hasattr(generic_mock_plugin, "setup") and callable(generic_mock_plugin.setup):
             await generic_mock_plugin.setup(config)
        return generic_mock_plugin

    mock_plugin_manager_for_es_lookup.get_plugin_instance.side_effect = get_instance_side_effect

    await provider_instance.setup(config={
        "plugin_manager": mock_plugin_manager_for_es_lookup,
        "key_provider": mock_key_provider_for_es_lookup
    })
    yield provider_instance
    await provider_instance.teardown()

# --- Existing Tests (some might be slightly adjusted) ---

@pytest.mark.asyncio
async def test_es_setup_direct_vs_config_params(
    mock_plugin_manager_for_es_lookup: PluginManager,
    mock_key_provider_for_es_lookup: KeyProvider
):
    provider = EmbeddingSimilarityLookupProvider()
    mock_embedder = MockEmbedderForLookup(plugin_id_val="custom_embed_id")
    mock_vs = MockVectorStoreForLookup()

    async def get_plugin_side_effect_vs(plugin_id_req, config=None, **kwargs):
        if plugin_id_req == "custom_embed_id":
            await mock_embedder.setup(config)
            return mock_embedder
        if plugin_id_req == "mock_vs_for_lookup_v1":
            await mock_vs.setup(config)
            return mock_vs
        return None
    mock_plugin_manager_for_es_lookup.get_plugin_instance.side_effect = get_plugin_side_effect_vs

    await provider.setup(config={
        "plugin_manager": mock_plugin_manager_for_es_lookup,
        "key_provider": mock_key_provider_for_es_lookup,
        "embedder_id": "custom_embed_id",
        "embedder_config": {"model": "embed_model"},
        "vector_store_id": "mock_vs_for_lookup_v1",
        "tool_embeddings_collection_name": "my_tools_collection",
        "tool_embeddings_path": "/custom/tools/db_path",
        "vector_store_config": {"specific_vs_param": "value", "path": "ignored_path_due_to_direct"}
    })

    assert provider._embedder is mock_embedder
    assert mock_embedder.setup_config_received == {"model": "embed_model", "key_provider": mock_key_provider_for_es_lookup}
    assert provider._tool_vector_store is mock_vs
    assert mock_vs.setup_config_received is not None
    assert mock_vs.setup_config_received.get("collection_name") == "my_tools_collection"
    assert mock_vs.setup_config_received.get("path") == "/custom/tools/db_path"
    assert mock_vs.setup_config_received.get("specific_vs_param") == "value"
    assert mock_vs.setup_config_received.get("key_provider") is mock_key_provider_for_es_lookup
    assert provider._key_provider is mock_key_provider_for_es_lookup
    await provider.teardown()


@pytest.mark.asyncio
async def test_es_index_and_find_pass_kp_to_embedder(
    es_lookup_provider: AsyncGenerator[EmbeddingSimilarityLookupProvider, None],
    mock_key_provider_for_es_lookup: KeyProvider
):
    provider = await anext(es_lookup_provider)
    mock_embedder_instance = cast(MockEmbedderForLookup, provider._embedder)
    assert mock_embedder_instance is not None
    assert mock_embedder_instance.setup_config_received is not None
    assert mock_embedder_instance.setup_config_received.get("key_provider") is mock_key_provider_for_es_lookup

    await provider.index_tools(
        tools_data=[{"identifier":"t1", "lookup_text_representation":"text", "_raw_metadata_snapshot": {}}],
        config={"embedder_config": {"runtime_param_index": True}}
    )
    assert mock_embedder_instance.last_embed_config_received is not None
    assert mock_embedder_instance.last_embed_config_received.get("runtime_param_index") is True
    assert mock_embedder_instance.last_embed_config_received.get("key_provider") is mock_key_provider_for_es_lookup

    if np and provider._tool_vector_store is None and provider._indexed_tool_embeddings_np is None:
        mock_embedder_instance.set_fixed_embedding([0.1,0.2,0.3])
        await provider.index_tools([{"identifier":"t_find", "lookup_text_representation":"find text for np", "_raw_metadata_snapshot": {}}])

    await provider.find_tools(
        "find me a tool",
        config={"embedder_config": {"runtime_param_find": True}}
    )
    assert mock_embedder_instance.last_embed_config_received is not None
    assert mock_embedder_instance.last_embed_config_received.get("runtime_param_find") is True
    assert mock_embedder_instance.last_embed_config_received.get("key_provider") is mock_key_provider_for_es_lookup


@pytest.mark.asyncio
async def test_es_setup_success_default_embedder(
    es_lookup_provider: AsyncGenerator[EmbeddingSimilarityLookupProvider, None],
):
    provider = await anext(es_lookup_provider)
    assert isinstance(provider._embedder, MockEmbedderForLookup)
    embedder_setup_config = cast(MockEmbedderForLookup, provider._embedder).setup_config_received
    assert embedder_setup_config is not None
    assert embedder_setup_config.get("key_provider") is provider._key_provider


@pytest.mark.skipif(np is None, reason="NumPy not available")
@pytest.mark.asyncio
async def test_es_index_tools_successful_numpy(
    es_lookup_provider: AsyncGenerator[EmbeddingSimilarityLookupProvider, None]
):
    provider = await anext(es_lookup_provider)
    provider._tool_vector_store = None
    provider._indexed_tool_embeddings_np = None
    provider._indexed_tool_data_list_np = []

    mock_embedder_instance = cast(MockEmbedderForLookup, provider._embedder)
    mock_embedder_instance.set_embeddings_map({
        "Tool A desc": [1.0, 0.0, 0.0],
        "Tool B desc": [0.0, 1.0, 0.0]
    })

    tools_data = [
        {"identifier": "tool_a", "lookup_text_representation": "Tool A desc", "_raw_metadata_snapshot": {"name": "Tool A"}},
        {"identifier": "tool_b", "lookup_text_representation": "Tool B desc", "_raw_metadata_snapshot": {"name": "Tool B"}}
    ]
    await provider.index_tools(tools_data)

    assert provider._indexed_tool_embeddings_np is not None
    assert provider._indexed_tool_embeddings_np.shape == (2, 3)
    assert len(provider._indexed_tool_data_list_np) == 2
    assert provider._indexed_tool_data_list_np[0]["identifier"] == "tool_a"

# --- New Tests for EmbeddingSimilarityLookupProvider ---

@pytest.mark.asyncio
async def test_es_setup_plugin_manager_missing(mock_key_provider_for_es_lookup: KeyProvider, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
    provider = EmbeddingSimilarityLookupProvider()
    await provider.setup(config={"key_provider": mock_key_provider_for_es_lookup})
    assert provider._embedder is None
    assert provider._tool_vector_store is None
    assert any(f"{provider.plugin_id} Error: PluginManager not provided. Cannot load sub-plugins." in record.message for record in caplog.records)

@pytest.mark.asyncio
async def test_es_setup_embedder_not_found(mock_plugin_manager_for_es_lookup: PluginManager, mock_key_provider_for_es_lookup: KeyProvider, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
    provider = EmbeddingSimilarityLookupProvider()
    mock_plugin_manager_for_es_lookup.get_plugin_instance.return_value = None

    await provider.setup(config={
        "plugin_manager": mock_plugin_manager_for_es_lookup,
        "key_provider": mock_key_provider_for_es_lookup,
        "embedder_id": "non_existent_embedder"
    })
    assert provider._embedder is None
    assert any(f"{provider.plugin_id} Error: Embedder 'non_existent_embedder' not found/invalid." in record.message for record in caplog.records)

@pytest.mark.asyncio
async def test_es_setup_vector_store_not_found(mock_plugin_manager_for_es_lookup: PluginManager, mock_key_provider_for_es_lookup: KeyProvider, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
    provider = EmbeddingSimilarityLookupProvider()
    mock_embedder_instance = MockEmbedderForLookup()

    async def get_instance_side_effect(plugin_id_req, config=None, **kwargs):
        if plugin_id_req == EmbeddingSimilarityLookupProvider.DEFAULT_EMBEDDER_ID:
            await mock_embedder_instance.setup(config)
            return mock_embedder_instance
        if plugin_id_req == "non_existent_vs":
            return None
        return AsyncMock()
    mock_plugin_manager_for_es_lookup.get_plugin_instance.side_effect = get_instance_side_effect

    await provider.setup(config={
        "plugin_manager": mock_plugin_manager_for_es_lookup,
        "key_provider": mock_key_provider_for_es_lookup,
        "vector_store_id": "non_existent_vs"
    })
    assert provider._embedder is mock_embedder_instance
    assert provider._tool_vector_store is None
    assert any(f"{provider.plugin_id} Error: Vector Store 'non_existent_vs' not found/invalid." in record.message for record in caplog.records)

@pytest.mark.asyncio
async def test_es_setup_no_numpy_no_vs(mock_plugin_manager_for_es_lookup: PluginManager, mock_key_provider_for_es_lookup: KeyProvider, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
    provider = EmbeddingSimilarityLookupProvider()
    mock_embedder_instance = MockEmbedderForLookup()
    mock_plugin_manager_for_es_lookup.get_plugin_instance.return_value = mock_embedder_instance

    with patch("genie_tooling.tool_lookup_providers.impl.embedding_similarity.np", None):
        await provider.setup(config={
            "plugin_manager": mock_plugin_manager_for_es_lookup,
            "key_provider": mock_key_provider_for_es_lookup,
            "vector_store_id": None
        })
    assert provider._embedder is mock_embedder_instance
    assert provider._tool_vector_store is None
    assert any(f"{provider.plugin_id} Error: NumPy not available and no Vector Store configured. Cannot function." in record.message for record in caplog.records)

@pytest.mark.asyncio
async def test_es_index_tools_embedder_fails(es_lookup_provider: AsyncGenerator[EmbeddingSimilarityLookupProvider, None], caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
    provider = await anext(es_lookup_provider)
    cast(MockEmbedderForLookup, provider._embedder).set_fail_on_embed(True)
    tools_data = [{"identifier": "t1", "lookup_text_representation": "text", "_raw_metadata_snapshot": {}}]
    await provider.index_tools(tools_data)
    assert provider._indexed_tool_embeddings_np is None
    assert any(f"{provider.plugin_id}: Error during embedding tool texts: Simulated embedder failure" in record.message for record in caplog.records)

@pytest.mark.asyncio
async def test_es_index_tools_vs_add_fails(mock_plugin_manager_for_es_lookup: PluginManager, mock_key_provider_for_es_lookup: KeyProvider, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
    provider = EmbeddingSimilarityLookupProvider()
    mock_embedder = MockEmbedderForLookup()
    mock_vs = MockVectorStoreForLookup() # Create the instance that will be configured to fail
    mock_vs.add_should_fail = True       # Configure it to fail

    async def get_instance_side_effect(plugin_id_req, config=None, **kwargs):
        if plugin_id_req == EmbeddingSimilarityLookupProvider.DEFAULT_EMBEDDER_ID:
            await mock_embedder.setup(config)
            return mock_embedder
        if plugin_id_req == "mock_vs_for_lookup_v1":
            # Return the pre-configured mock_vs instance
            await mock_vs.setup(config) # Its setup will be called by provider
            return mock_vs
        return None
    mock_plugin_manager_for_es_lookup.get_plugin_instance.side_effect = get_instance_side_effect

    await provider.setup(config={
        "plugin_manager": mock_plugin_manager_for_es_lookup,
        "key_provider": mock_key_provider_for_es_lookup,
        "vector_store_id": "mock_vs_for_lookup_v1"
    })
    tools_data = [{"identifier": "t1", "lookup_text_representation": "text", "_raw_metadata_snapshot": {}}]
    await provider.index_tools(tools_data)

    assert any(
        f"{provider.plugin_id}: Error adding tool embeddings to Vector Store: Simulated VS add failure" in record.message
        and record.name == PROVIDER_LOGGER_NAME
        for record in caplog.records
    ), "Expected error log for VS add failure not found."
    await provider.teardown()

@pytest.mark.asyncio
async def test_es_find_tools_query_embed_fails(es_lookup_provider: AsyncGenerator[EmbeddingSimilarityLookupProvider, None], caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
    provider = await anext(es_lookup_provider)
    cast(MockEmbedderForLookup, provider._embedder).set_fail_on_embed(True)
    results = await provider.find_tools("query")
    assert results == []
    assert any(f"{provider.plugin_id}: Error embedding query 'query...': Simulated embedder failure" in record.message for record in caplog.records)

@pytest.mark.asyncio
async def test_es_find_tools_vs_search_fails(mock_plugin_manager_for_es_lookup: PluginManager, mock_key_provider_for_es_lookup: KeyProvider, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
    provider = EmbeddingSimilarityLookupProvider()
    mock_embedder = MockEmbedderForLookup()
    mock_vs = MockVectorStoreForLookup() # Create the instance
    mock_vs.search_should_fail = True    # Configure it to fail

    async def get_instance_side_effect(plugin_id_req, config=None, **kwargs):
        if plugin_id_req == EmbeddingSimilarityLookupProvider.DEFAULT_EMBEDDER_ID:
            await mock_embedder.setup(config)
            return mock_embedder
        if plugin_id_req == "mock_vs_for_lookup_v1":
            await mock_vs.setup(config) # Its setup will be called
            return mock_vs               # Return the pre-configured instance
        return None
    mock_plugin_manager_for_es_lookup.get_plugin_instance.side_effect = get_instance_side_effect

    await provider.setup(config={
        "plugin_manager": mock_plugin_manager_for_es_lookup,
        "key_provider": mock_key_provider_for_es_lookup,
        "vector_store_id": "mock_vs_for_lookup_v1"
    })
    results = await provider.find_tools("query")

    assert results == []
    assert any(
        f"{provider.plugin_id}: Error searching Vector Store: Simulated VS search failure" in record.message
        and record.name == PROVIDER_LOGGER_NAME
        for record in caplog.records
    ), "Expected error log for VS search failure not found."
    await provider.teardown()

@pytest.mark.asyncio
async def test_es_find_tools_empty_query(es_lookup_provider: AsyncGenerator[EmbeddingSimilarityLookupProvider, None]):
    provider = await anext(es_lookup_provider)
    results = await provider.find_tools("   ")
    assert results == []
    results_none = await provider.find_tools("")
    assert results_none == []

@pytest.mark.skipif(np is None, reason="NumPy not available")
@pytest.mark.asyncio
async def test_es_find_tools_no_tools_indexed_numpy(es_lookup_provider: AsyncGenerator[EmbeddingSimilarityLookupProvider, None], caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING, logger=PROVIDER_LOGGER_NAME)
    provider = await anext(es_lookup_provider)
    provider._tool_vector_store = None
    provider._indexed_tool_embeddings_np = None
    provider._indexed_tool_data_list_np = []

    results = await provider.find_tools("query")
    assert results == []
    assert any(f"{provider.plugin_id}: Tool index not built or NumPy not available for search." in record.message for record in caplog.records)

@pytest.mark.asyncio
async def test_es_find_tools_no_tools_indexed_vs(mock_plugin_manager_for_es_lookup: PluginManager, mock_key_provider_for_es_lookup: KeyProvider):
    provider = EmbeddingSimilarityLookupProvider()
    mock_embedder = MockEmbedderForLookup()
    mock_vs = MockVectorStoreForLookup()
    mock_vs.set_search_results([])

    async def get_instance_side_effect(plugin_id_req, config=None, **kwargs):
        if plugin_id_req == EmbeddingSimilarityLookupProvider.DEFAULT_EMBEDDER_ID:
            await mock_embedder.setup(config)
            return mock_embedder
        if plugin_id_req == "mock_vs_for_lookup_v1":
            await mock_vs.setup(config)
            return mock_vs
        return None
    mock_plugin_manager_for_es_lookup.get_plugin_instance.side_effect = get_instance_side_effect

    await provider.setup(config={
        "plugin_manager": mock_plugin_manager_for_es_lookup,
        "key_provider": mock_key_provider_for_es_lookup,
        "vector_store_id": "mock_vs_for_lookup_v1"
    })
    results = await provider.find_tools("query")
    assert results == []
    await provider.teardown()

@pytest.mark.skipif(np is None, reason="NumPy not available")
@pytest.mark.asyncio
async def test_es_find_tools_with_numpy_index(es_lookup_provider: AsyncGenerator[EmbeddingSimilarityLookupProvider, None]):
    provider = await anext(es_lookup_provider)
    provider._tool_vector_store = None

    mock_embedder_instance = cast(MockEmbedderForLookup, provider._embedder)
    mock_embedder_instance.set_embeddings_map({
        "Tool A desc": [1.0, 0.0, 0.0],
        "Tool B desc": [0.0, 1.0, 0.0],
        "Query text":  [0.9, 0.1, 0.0]
    })
    tools_data = [
        {"identifier": "tool_a", "lookup_text_representation": "Tool A desc", "_raw_metadata_snapshot": {}},
        {"identifier": "tool_b", "lookup_text_representation": "Tool B desc", "_raw_metadata_snapshot": {}}
    ]
    await provider.index_tools(tools_data)
    assert provider._indexed_tool_embeddings_np is not None

    results = await provider.find_tools("Query text", top_k=1)
    assert len(results) == 1
    assert results[0].tool_identifier == "tool_a"

@pytest.mark.asyncio
async def test_es_find_tools_with_vector_store(mock_plugin_manager_for_es_lookup: PluginManager, mock_key_provider_for_es_lookup: KeyProvider):
    provider = EmbeddingSimilarityLookupProvider()
    mock_embedder = MockEmbedderForLookup()
    mock_vs_instance = MockVectorStoreForLookup()

    retrieved_from_vs = [
        _TestRetrievedChunk(
            id="tool_c_vs",
            content="VS result C",
            score=0.95,
            metadata={
                "identifier": "tool_c_vs",
                "lookup_text_representation": "Description for Tool C",
                "_raw_metadata_snapshot": {"name": "Tool C", "description_llm": "Desc C"}
            }
        ),
        _TestRetrievedChunk(
            id="tool_d_vs",
            content="VS result D",
            score=0.85,
            metadata={
                "identifier": "tool_d_vs",
                "lookup_text_representation": "Description for Tool D",
                "_raw_metadata_snapshot": {"name": "Tool D", "description_llm": "Desc D"}
            }
        )
    ]
    mock_embedder.set_fixed_embedding([0.1,0.2,0.3])

    async def get_instance_side_effect(plugin_id_req, config=None, **kwargs):
        if plugin_id_req == EmbeddingSimilarityLookupProvider.DEFAULT_EMBEDDER_ID:
            await mock_embedder.setup(config)
            return mock_embedder
        if plugin_id_req == "mock_vs_for_lookup_v1":
            await mock_vs_instance.setup(config)
            return mock_vs_instance
        return None
    mock_plugin_manager_for_es_lookup.get_plugin_instance.side_effect = get_instance_side_effect

    await provider.setup(config={
        "plugin_manager": mock_plugin_manager_for_es_lookup,
        "key_provider": mock_key_provider_for_es_lookup,
        "vector_store_id": "mock_vs_for_lookup_v1"
    })

    assert provider._tool_vector_store is mock_vs_instance
    cast(MockVectorStoreForLookup, provider._tool_vector_store).set_search_results(retrieved_from_vs)


    results = await provider.find_tools("query for vs", top_k=2)
    assert len(results) == 2, f"Expected 2 results, got {len(results)}. Results: {results}"
    assert results[0].tool_identifier == "tool_c_vs"
    assert results[0].score == 0.95
    assert results[1].tool_identifier == "tool_d_vs"
    await provider.teardown()

@pytest.mark.asyncio
async def test_es_key_provider_to_vector_store(mock_plugin_manager_for_es_lookup: PluginManager, mock_key_provider_for_es_lookup: KeyProvider):
    provider = EmbeddingSimilarityLookupProvider()
    mock_embedder = MockEmbedderForLookup()
    mock_vs = MockVectorStoreForLookup()

    async def get_instance_side_effect(plugin_id_req, config=None, **kwargs):
        if plugin_id_req == EmbeddingSimilarityLookupProvider.DEFAULT_EMBEDDER_ID:
            await mock_embedder.setup(config)
            return mock_embedder
        if plugin_id_req == "mock_vs_for_lookup_v1":
            await mock_vs.setup(config)
            return mock_vs
        return None
    mock_plugin_manager_for_es_lookup.get_plugin_instance.side_effect = get_instance_side_effect

    await provider.setup(config={
        "plugin_manager": mock_plugin_manager_for_es_lookup,
        "key_provider": mock_key_provider_for_es_lookup,
        "vector_store_id": "mock_vs_for_lookup_v1",
        "vector_store_config": {"some_vs_param": "val"}
    })

    assert provider._tool_vector_store is mock_vs
    assert mock_vs.setup_config_received is not None
    assert mock_vs.setup_config_received.get("key_provider") is mock_key_provider_for_es_lookup
    assert mock_vs.setup_config_received.get("some_vs_param") == "val"
    await provider.teardown()

@pytest.mark.asyncio
async def test_es_index_tools_embedder_returns_empty_list_for_some(
    es_lookup_provider: AsyncGenerator[EmbeddingSimilarityLookupProvider, None], caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.WARNING, logger=PROVIDER_LOGGER_NAME)
    provider = await anext(es_lookup_provider)
    provider._tool_vector_store = None

    mock_embedder_instance = cast(MockEmbedderForLookup, provider._embedder)
    mock_embedder_instance.set_embeddings_map({
        "Tool A desc": [1.0, 0.0, 0.0],
        "Tool B desc": []
    })

    tools_data = [
        {"identifier": "tool_a", "lookup_text_representation": "Tool A desc", "_raw_metadata_snapshot": {}},
        {"identifier": "tool_b", "lookup_text_representation": "Tool B desc", "_raw_metadata_snapshot": {}}
    ]
    await provider.index_tools(tools_data)

    assert any(f"{provider.plugin_id}: Invalid/empty embedding for content of chunk ID 'tool_b_idx_" in rec.message for rec in caplog.records)
    if np and provider._indexed_tool_embeddings_np is not None:
        assert provider._indexed_tool_embeddings_np.shape == (1, 3)
        assert len(provider._indexed_tool_data_list_np) == 1
        assert provider._indexed_tool_data_list_np[0]["identifier"] == "tool_a"
    elif provider._tool_vector_store:
        assert len(cast(MockVectorStoreForLookup, provider._tool_vector_store).items_added) == 1

@pytest.mark.asyncio
async def test_es_index_tools_mismatch_embeddings_and_texts(
    es_lookup_provider: AsyncGenerator[EmbeddingSimilarityLookupProvider, None], caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
    provider = await anext(es_lookup_provider)
    provider._tool_vector_store = None

    mock_embedder_instance = cast(MockEmbedderForLookup, provider._embedder)
    async def mock_embed_mismatch(chunks: AsyncIterable[Chunk], config=None):
        count = 0
        async for chunk in chunks:
            if count == 0:
                yield chunk, [0.5, 0.5, 0.5]
            count += 1
            if count >= 1: # Only yield one embedding for multiple chunks
                break
    mock_embedder_instance.embed = mock_embed_mismatch # type: ignore

    tools_data = [
        {"identifier": "tool_x", "lookup_text_representation": "Text X", "_raw_metadata_snapshot": {}},
        {"identifier": "tool_y", "lookup_text_representation": "Text Y", "_raw_metadata_snapshot": {}}
    ]
    await provider.index_tools(tools_data)

    assert any(f"{provider.plugin_id}: Mismatch after embedding. Expected 2 embeddings, got 1. Index may be incomplete." in record.message for record in caplog.records)
    if np and provider._indexed_tool_embeddings_np is not None:
        assert provider._indexed_tool_embeddings_np.shape == (1, 3)
        assert len(provider._indexed_tool_data_list_np) == 1
        assert provider._indexed_tool_data_list_np[0]["identifier"] == "tool_x"

@pytest.mark.skipif(np is None, reason="NumPy not available")
@pytest.mark.asyncio
async def test_es_find_tools_numpy_dimension_mismatch(es_lookup_provider: AsyncGenerator[EmbeddingSimilarityLookupProvider, None], caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
    provider = await anext(es_lookup_provider)
    provider._tool_vector_store = None

    mock_embedder_instance = cast(MockEmbedderForLookup, provider._embedder)
    mock_embedder_instance.set_embeddings_map({"Tool 3D": [1.0, 0.0, 0.0]})
    await provider.index_tools([{"identifier": "tool_3d", "lookup_text_representation": "Tool 3D", "_raw_metadata_snapshot": {}}])

    mock_embedder_instance.set_fixed_embedding([0.9, 0.1]) # Query embedding has different dimension
    results = await provider.find_tools("Query for 2D")

    assert results == []
    assert any(f"{provider.plugin_id}: In-memory index/query dimension mismatch ((1, 3) vs (1, 2)). Cannot compute similarity." in record.message for record in caplog.records)

@pytest.mark.skipif(np is None, reason="NumPy not available")
@pytest.mark.asyncio
async def test_es_find_tools_numpy_single_item_index_correct_dim(es_lookup_provider: AsyncGenerator[EmbeddingSimilarityLookupProvider, None]):
    provider = await anext(es_lookup_provider)
    provider._tool_vector_store = None

    mock_embedder_instance = cast(MockEmbedderForLookup, provider._embedder)
    mock_embedder_instance.set_embeddings_map({"Single Tool": [0.1, 0.2, 0.3]})
    await provider.index_tools([{"identifier": "single_tool", "lookup_text_representation": "Single Tool", "_raw_metadata_snapshot": {}}])

    mock_embedder_instance.set_fixed_embedding([0.11, 0.22, 0.33]) # Query embedding
    results = await provider.find_tools("Query for single")
    assert len(results) == 1
    assert results[0].tool_identifier == "single_tool"

@pytest.mark.asyncio
async def test_es_teardown_clears_resources(es_lookup_provider: AsyncGenerator[EmbeddingSimilarityLookupProvider, None]):
    provider = await anext(es_lookup_provider)
    # Setup some dummy state to ensure teardown clears them
    provider._embedder = MockEmbedderForLookup() # type: ignore
    provider._tool_vector_store = MockVectorStoreForLookup() # type: ignore
    if np:
        provider._indexed_tool_embeddings_np = np.array([[1.0]])
    provider._indexed_tool_data_list_np = [{"id":"dummy"}]
    provider._plugin_manager = AsyncMock(spec=PluginManager)
    provider._key_provider = AsyncMock(spec=KeyProvider)

    await provider.teardown()

    assert provider._embedder is None
    assert provider._tool_vector_store is None
    assert provider._plugin_manager is None
    assert provider._key_provider is None
    assert provider._indexed_tool_embeddings_np is None
    assert provider._indexed_tool_data_list_np == []