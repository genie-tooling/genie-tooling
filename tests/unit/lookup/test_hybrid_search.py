import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.lookup.types import RankedToolResult
from genie_tooling.tool_lookup_providers.abc import ToolLookupProvider
from genie_tooling.tool_lookup_providers.impl.hybrid_search import (
    HybridSearchLookupProvider,
)

PROVIDER_LOGGER_NAME = "genie_tooling.tool_lookup_providers.impl.hybrid_search"

@pytest.fixture
def mock_dense_provider() -> MagicMock:
    provider = AsyncMock(spec=ToolLookupProvider)
    provider.plugin_id = "mock_dense_provider_v1"
    provider.find_tools = AsyncMock(return_value=[])
    provider.index_tools = AsyncMock()
    provider.add_tool = AsyncMock(return_value=True)
    provider.update_tool = AsyncMock(return_value=True)
    provider.remove_tool = AsyncMock(return_value=True)
    provider.teardown = AsyncMock()
    return provider

@pytest.fixture
def mock_sparse_provider() -> MagicMock:
    provider = AsyncMock(spec=ToolLookupProvider)
    provider.plugin_id = "mock_sparse_provider_v1"
    provider.find_tools = AsyncMock(return_value=[])
    provider.index_tools = AsyncMock()
    provider.add_tool = AsyncMock(return_value=True)
    provider.update_tool = AsyncMock(return_value=True)
    provider.remove_tool = AsyncMock(return_value=True)
    provider.teardown = AsyncMock()
    return provider

@pytest.fixture
def mock_plugin_manager_for_hybrid(mock_dense_provider, mock_sparse_provider) -> MagicMock:
    pm = MagicMock(spec=PluginManager)
    async def get_instance_side_effect(plugin_id, config=None):
        if plugin_id == "embedding_similarity_lookup_v1":
            return mock_dense_provider
        if plugin_id == "keyword_match_lookup_v1":
            return mock_sparse_provider
        return None
    pm.get_plugin_instance = AsyncMock(side_effect=get_instance_side_effect)
    return pm

@pytest.fixture
async def hybrid_search_provider(mock_plugin_manager_for_hybrid) -> HybridSearchLookupProvider:
    provider = HybridSearchLookupProvider()
    await provider.setup({"plugin_manager": mock_plugin_manager_for_hybrid})
    return provider

@pytest.mark.asyncio
class TestHybridSearchProvider:
    async def test_setup_success(self, hybrid_search_provider, mock_dense_provider, mock_sparse_provider):
        provider = await hybrid_search_provider
        assert provider._dense_provider is mock_dense_provider
        assert provider._sparse_provider is mock_sparse_provider

    async def test_setup_dense_provider_fails(self, mock_plugin_manager_for_hybrid, mock_sparse_provider, caplog):
        caplog.set_level(logging.ERROR, logger=PROVIDER_LOGGER_NAME)
        async def get_instance_fail_dense(plugin_id, config=None):
            if plugin_id == "embedding_similarity_lookup_v1": return None
            if plugin_id == "keyword_match_lookup_v1": return mock_sparse_provider
            return None
        mock_plugin_manager_for_hybrid.get_plugin_instance.side_effect = get_instance_fail_dense

        provider = HybridSearchLookupProvider()
        await provider.setup({"plugin_manager": mock_plugin_manager_for_hybrid})

        assert provider._dense_provider is None
        assert provider._sparse_provider is not None
        assert "Dense provider 'embedding_similarity_lookup_v1' not found or invalid." in caplog.text

    async def test_index_tools_delegates_to_both(self, hybrid_search_provider, mock_dense_provider, mock_sparse_provider):
        provider = await hybrid_search_provider
        tools_data = [{"id": "1"}]
        await provider.index_tools(tools_data)
        mock_dense_provider.index_tools.assert_awaited_once_with(tools_data, None)
        mock_sparse_provider.index_tools.assert_awaited_once_with(tools_data, None)

    async def test_incremental_methods_delegate_to_both(self, hybrid_search_provider, mock_dense_provider, mock_sparse_provider):
        provider = await hybrid_search_provider
        tool_data = {"identifier": "t1"}

        await provider.add_tool(tool_data)
        mock_dense_provider.add_tool.assert_awaited_once_with(tool_data, None)
        mock_sparse_provider.add_tool.assert_awaited_once_with(tool_data, None)

        await provider.update_tool("t1", tool_data)
        mock_dense_provider.update_tool.assert_awaited_once_with("t1", tool_data, None)
        mock_sparse_provider.update_tool.assert_awaited_once_with("t1", tool_data, None)

        await provider.remove_tool("t1")
        mock_dense_provider.remove_tool.assert_awaited_once_with("t1", None)
        mock_sparse_provider.remove_tool.assert_awaited_once_with("t1", None)

    async def test_find_tools_rrf_fusion(self, hybrid_search_provider, mock_dense_provider, mock_sparse_provider):
        provider = await hybrid_search_provider
        mock_dense_provider.find_tools.return_value = [
            RankedToolResult("tool_A", 0.9, similarity_score_details={"cosine": 0.9}),
            RankedToolResult("tool_B", 0.8, similarity_score_details={"cosine": 0.8}),
        ]
        mock_sparse_provider.find_tools.return_value = [
            RankedToolResult("tool_B", 2.0, matched_keywords=["b"]),
            RankedToolResult("tool_A", 1.0, matched_keywords=["a"]),
            RankedToolResult("tool_C", 1.0, matched_keywords=["c"]),
        ]

        results = await provider.find_tools("query", top_k=3)

        assert len(results) == 3
        # RRF scores (k=60):
        # A: 1/(61) + 1/(62) = 0.01639 + 0.01612 = 0.03251
        # B: 1/(62) + 1/(61) = 0.01612 + 0.01639 = 0.03251
        # C: 1/(63) = 0.01587
        assert results[0].tool_identifier in ["tool_A", "tool_B"]
        assert results[1].tool_identifier in ["tool_A", "tool_B"]
        assert results[2].tool_identifier == "tool_C"
        assert results[0].score == pytest.approx(results[1].score)
        assert results[1].score > results[2].score
        # Check that diagnostic info is preserved
        tool_a_result = next(r for r in results if r.tool_identifier == "tool_A")
        assert tool_a_result.similarity_score_details == {"cosine": 0.9}
        assert tool_a_result.matched_keywords == ["a"]

    async def test_find_tools_one_provider_empty(self, hybrid_search_provider, mock_dense_provider, mock_sparse_provider):
        provider = await hybrid_search_provider
        mock_dense_provider.find_tools.return_value = []
        mock_sparse_provider.find_tools.return_value = [
            RankedToolResult("tool_C", 1.0),
            RankedToolResult("tool_D", 1.0),
        ]

        results = await provider.find_tools("query", top_k=2)

        assert len(results) == 2
        assert results[0].tool_identifier in ["tool_C", "tool_D"]
        assert results[1].tool_identifier in ["tool_C", "tool_D"]

    async def test_find_tools_both_providers_empty(self, hybrid_search_provider, mock_dense_provider, mock_sparse_provider):
        provider = await hybrid_search_provider
        mock_dense_provider.find_tools.return_value = []
        mock_sparse_provider.find_tools.return_value = []

        results = await provider.find_tools("query")
        assert results == []

    async def test_teardown_delegates_to_both(self, hybrid_search_provider, mock_dense_provider, mock_sparse_provider):
        provider = await hybrid_search_provider
        await provider.teardown()
        mock_dense_provider.teardown.assert_awaited_once()
        mock_sparse_provider.teardown.assert_awaited_once()
        assert provider._dense_provider is None
        assert provider._sparse_provider is None
