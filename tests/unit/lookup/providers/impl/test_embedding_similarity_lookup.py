from typing import Any, AsyncIterable, Dict, List, Optional, Tuple, cast
from unittest.mock import AsyncMock

import pytest

try:
    import numpy as np
except ImportError:
    np = None

from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.core.types import Chunk, EmbeddingVector, Plugin
from genie_tooling.embedding_generators.abc import EmbeddingGeneratorPlugin
from genie_tooling.security.key_provider import KeyProvider
from genie_tooling.tool_lookup_providers.impl.embedding_similarity import (
    EmbeddingSimilarityLookupProvider,
)
from genie_tooling.vector_stores.abc import VectorStorePlugin


class MockEmbedderForLookup(EmbeddingGeneratorPlugin, Plugin):
    _plugin_id_value: str
    _fixed_embedding: Optional[List[float]] = None; _embeddings_map: Optional[Dict[str, List[float]]] = None
    _fail_on_embed: bool = False; teardown_called: bool = False
    setup_config_received: Optional[Dict[str, Any]] = None
    last_embed_config_received: Optional[Dict[str, Any]] = None

    def __init__(self, plugin_id_val: str = "mock_embedder_for_lookup_v1"):
        self._plugin_id_value = plugin_id_val; self.description: str = "Mock embedder"
    @property
    def plugin_id(self) -> str: return self._plugin_id_value
    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.setup_config_received = config; self.teardown_called = False; self.last_embed_config_received = None
    def set_fixed_embedding(self, e: List[float]): self._fixed_embedding=e; self._embeddings_map=None
    def set_embeddings_map(self, m: Dict[str,List[float]]): self._embeddings_map=m; self._fixed_embedding=None
    def set_fail_on_embed(self, f: bool): self._fail_on_embed = f
    async def embed(self, chunks: AsyncIterable[Chunk], config: Optional[Dict[str,Any]]=None) -> AsyncIterable[tuple[Chunk,EmbeddingVector]]:
        self.last_embed_config_received = config
        if self._fail_on_embed: raise RuntimeError("Simulated embedder failure")
        async for chunk_item in chunks:
            if self._embeddings_map and chunk_item.content in self._embeddings_map:
                yield chunk_item, self._embeddings_map[chunk_item.content]
            elif self._fixed_embedding: yield chunk_item, self._fixed_embedding
            else: yield chunk_item, [0.1] * (config or {}).get("expected_dim",3)
    async def teardown(self) -> None: self.teardown_called = True

class MockVectorStoreForLookup(VectorStorePlugin, Plugin):
    plugin_id: str = "mock_vs_for_lookup_v1"
    description: str = "Mock VS"
    setup_config_received: Optional[Dict[str, Any]] = None
    add_called_with_config: Optional[Dict[str, Any]] = None
    search_called_with_config: Optional[Dict[str, Any]] = None
    items_added: List[Tuple[Chunk, EmbeddingVector]] = []

    async def setup(self, config: Optional[Dict[str, Any]] = None): self.setup_config_received = config; self.items_added = []
    async def add(self, embeddings: AsyncIterable[Tuple[Chunk, EmbeddingVector]], config: Optional[Dict[str, Any]] = None):
        self.add_called_with_config = config; count = 0
        async for item in embeddings: self.items_added.append(item); count+=1
        return {"added_count":count}
    async def search(self, qe: EmbeddingVector, tk: int, fm: Optional[Dict[str,Any]]=None, config: Optional[Dict[str,Any]]=None):
        self.search_called_with_config = config; return []
    async def delete(self, ids=None, fm=None, da=False, config=None): return True
    async def teardown(self): pass


@pytest.fixture
def mock_plugin_manager_for_es_lookup(mocker) -> PluginManager:
    pm = mocker.MagicMock(spec=PluginManager)
    pm.get_plugin_instance = AsyncMock()
    pm.list_discovered_plugin_classes = mocker.MagicMock(return_value={})
    return pm

@pytest.fixture
def mock_key_provider_for_es_lookup(mocker) -> KeyProvider:
    return mocker.AsyncMock(spec=KeyProvider)

@pytest.fixture
async def es_lookup_provider(mock_plugin_manager_for_es_lookup: PluginManager, mock_key_provider_for_es_lookup: KeyProvider) -> EmbeddingSimilarityLookupProvider:
    provider_instance = EmbeddingSimilarityLookupProvider()
    mock_embedder_instance = MockEmbedderForLookup()

    async def get_instance_side_effect(plugin_id_req, config=None, **kwargs):
        if plugin_id_req == EmbeddingSimilarityLookupProvider.DEFAULT_EMBEDDER_ID:
            await mock_embedder_instance.setup(config)
            return mock_embedder_instance
        generic_mock_plugin = AsyncMock(name=f"mock_plugin_{plugin_id_req}")
        if hasattr(generic_mock_plugin, "setup") and callable(generic_mock_plugin.setup):
             await generic_mock_plugin.setup(config)
        return generic_mock_plugin

    mock_plugin_manager_for_es_lookup.get_plugin_instance.side_effect = get_instance_side_effect

    await provider_instance.setup(config={
        "plugin_manager": mock_plugin_manager_for_es_lookup,
        "key_provider": mock_key_provider_for_es_lookup
    })
    return provider_instance

@pytest.mark.asyncio
async def test_es_setup_direct_vs_config_params(
    mock_plugin_manager_for_es_lookup: PluginManager,
    mock_key_provider_for_es_lookup: KeyProvider
):
    provider = EmbeddingSimilarityLookupProvider()
    mock_embedder = MockEmbedderForLookup(plugin_id_val="custom_embed_id")
    mock_vs = MockVectorStoreForLookup()

    async def get_plugin_side_effect_vs(plugin_id_req, config=None, **kwargs):
        if plugin_id_req == "custom_embed_id": await mock_embedder.setup(config); return mock_embedder
        if plugin_id_req == "custom_vs_id": await mock_vs.setup(config); return mock_vs
        return None
    mock_plugin_manager_for_es_lookup.get_plugin_instance.side_effect = get_plugin_side_effect_vs

    await provider.setup(config={
        "plugin_manager": mock_plugin_manager_for_es_lookup,
        "key_provider": mock_key_provider_for_es_lookup,
        "embedder_id": "custom_embed_id",
        "embedder_config": {"model": "embed_model", "key_provider": mock_key_provider_for_es_lookup},
        "vector_store_id": "custom_vs_id",
        "tool_embeddings_collection_name": "my_tools_collection",
        "tool_embeddings_path": "/custom/tools/db_path",
        "vector_store_config": {"collection_name": "ignored_collection", "path": "ignored_path"}
    })

    assert provider._embedder is mock_embedder
    assert mock_embedder.setup_config_received == {"model": "embed_model", "key_provider": mock_key_provider_for_es_lookup}
    assert provider._tool_vector_store is mock_vs
    assert mock_vs.setup_config_received is not None
    assert mock_vs.setup_config_received.get("collection_name") == "my_tools_collection"
    assert mock_vs.setup_config_received.get("path") == "/custom/tools/db_path"
    assert provider._key_provider is mock_key_provider_for_es_lookup

@pytest.mark.asyncio
async def test_es_index_and_find_pass_kp_to_embedder(
    es_lookup_provider: EmbeddingSimilarityLookupProvider,
    mock_key_provider_for_es_lookup: KeyProvider
):
    provider = await es_lookup_provider
    mock_embedder_instance = cast(MockEmbedderForLookup, provider._embedder)
    assert mock_embedder_instance is not None
    assert mock_embedder_instance.setup_config_received is not None
    assert mock_embedder_instance.setup_config_received.get("key_provider") is mock_key_provider_for_es_lookup

    original_embed_method = mock_embedder_instance.embed
    embed_call_log = []
    async def spy_embed_wrapper(*args, **kwargs):
        embed_call_log.append({"args": args, "kwargs": kwargs})
        async for item in original_embed_method(*args, **kwargs):
            yield item
    mock_embedder_instance.embed = spy_embed_wrapper # type: ignore

    await provider.index_tools(
        tools_data=[{"identifier":"t1", "lookup_text_representation":"text", "_raw_metadata_snapshot": {}}],
        config={"embedder_config": {"runtime_param_index": True}}
    )
    assert len(embed_call_log) > 0, "Embedder's embed method was not called during index_tools"
    index_embed_call_kwargs = embed_call_log[0]["kwargs"].get("config", {})
    assert index_embed_call_kwargs.get("runtime_param_index") is True
    assert index_embed_call_kwargs.get("key_provider") is mock_key_provider_for_es_lookup

    embed_call_log.clear()
    mock_embedder_instance.embed = original_embed_method

    if np and provider._tool_vector_store is None:
        mock_embedder_instance.set_fixed_embedding([0.1,0.2,0.3])
        await provider.index_tools([{"identifier":"t_find", "lookup_text_representation":"find text for np", "_raw_metadata_snapshot": {}}])

    mock_embedder_instance.embed = spy_embed_wrapper # type: ignore

    await provider.find_tools(
        "find me a tool", # Positional argument for 'query'
        config={"embedder_config": {"runtime_param_find": True}}
    )
    assert len(embed_call_log) > 0, "Embedder's embed method was not called during find_tools"
    find_embed_call_kwargs = embed_call_log[0]["kwargs"].get("config", {})
    assert find_embed_call_kwargs.get("runtime_param_find") is True
    assert find_embed_call_kwargs.get("key_provider") is mock_key_provider_for_es_lookup

    mock_embedder_instance.embed = original_embed_method


@pytest.mark.asyncio
async def test_es_setup_success_default_embedder(
    es_lookup_provider: EmbeddingSimilarityLookupProvider,
    mock_plugin_manager_for_es_lookup: PluginManager
):
    provider = await es_lookup_provider
    assert isinstance(provider._embedder, MockEmbedderForLookup)
    embedder_setup_config = cast(MockEmbedderForLookup, provider._embedder).setup_config_received
    assert embedder_setup_config is not None
    assert embedder_setup_config.get("key_provider") is provider._key_provider


@pytest.mark.skipif(np is None, reason="NumPy not available")
@pytest.mark.asyncio
async def test_es_index_tools_successful(
    es_lookup_provider: EmbeddingSimilarityLookupProvider
):
    provider = await es_lookup_provider
    assert provider._tool_vector_store is None
    assert provider._indexed_tool_embeddings_np is None

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

