### tests/unit/tools/impl/test_intelligent_search_aggregator_tool.py
from typing import Any, AsyncIterable, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genie_tooling.core.types import Chunk, EmbeddingVector
from genie_tooling.embedding_generators.abc import EmbeddingGeneratorPlugin
from genie_tooling.tools.impl.intelligent_search_aggregator_tool import (
    IntelligentSearchAggregatorTool,
)

# Constants for patching and logging
AGGREGATOR_MODULE_PATH = "genie_tooling.tools.impl.intelligent_search_aggregator_tool"
AGGREGATOR_LOGGER_NAME = AGGREGATOR_MODULE_PATH
NUMPY_AVAILABLE_PATH = f"{AGGREGATOR_MODULE_PATH}.NUMPY_AVAILABLE"
RANK_BM25_AVAILABLE_PATH = f"{AGGREGATOR_MODULE_PATH}.RANK_BM25_AVAILABLE"


# Mock Embedder for testing
class MockEmbedderForAggregator(EmbeddingGeneratorPlugin):
    plugin_id: str = "mock_embedder_for_aggregator"
    description: str = "A mock embedder"
    _embeddings_map: Dict[str, List[float]]

    def __init__(self):
        self._embeddings_map = {}

    def set_embeddings(self, new_map: Dict[str, List[float]]):
        self._embeddings_map = new_map

    async def embed(
        self, chunks: AsyncIterable[Chunk], config: Optional[Dict[str, Any]] = None
    ) -> AsyncIterable[Tuple[Chunk, EmbeddingVector]]:
        async for chunk in chunks:
            # Return pre-defined embedding or a default based on content length
            yield chunk, self._embeddings_map.get(
                chunk.content, [0.1 * len(chunk.content)] * 3
            )

    async def setup(self, config=None):
        pass

    async def teardown(self):
        pass


@pytest.fixture()
def mock_genie_for_aggregator() -> MagicMock:
    """Provides a mock of the Genie facade for the aggregator tool."""
    genie = MagicMock(name="MockGenieFacadeForAggregator")
    genie.execute_tool = AsyncMock()
    return genie


@pytest.fixture()
def mock_plugin_manager_for_aggregator() -> AsyncMock:
    """Provides a mock PluginManager."""
    pm = AsyncMock(name="MockPluginManagerForAggregator")
    pm.get_plugin_instance = AsyncMock()
    return pm


@pytest.fixture()
async def aggregator_tool(
    mock_plugin_manager_for_aggregator: AsyncMock,
) -> IntelligentSearchAggregatorTool:
    """Provides an initialized IntelligentSearchAggregatorTool."""
    mock_embedder = MockEmbedderForAggregator()
    mock_plugin_manager_for_aggregator.get_plugin_instance.return_value = mock_embedder
    tool = IntelligentSearchAggregatorTool(
        plugin_manager=mock_plugin_manager_for_aggregator
    )
    await tool.setup()
    tool._test_mock_embedder = mock_embedder  # type: ignore # Attach for test access
    return tool


@pytest.mark.asyncio()
class TestIntelligentSearchAggregatorExecution:
    """Tests the execute method and its internal logic."""

    @patch(NUMPY_AVAILABLE_PATH, True)
    @patch(RANK_BM25_AVAILABLE_PATH, True)
    async def test_execute_success_and_reranking(
        self,
        aggregator_tool: IntelligentSearchAggregatorTool,
        mock_genie_for_aggregator: MagicMock,
    ):
        """Test a full successful run with all scoring mechanisms enabled."""
        tool = await aggregator_tool
        mock_google_results = {
            "results": [
                {
                    "title": "Doc A",
                    "url": "url_a",
                    "description": "About python programming",
                },
                {"title": "Doc B", "url": "url_b", "description": "About data science"},
            ]
        }
        mock_arxiv_results = {
            "results": [
                {
                    "title": "Paper C",
                    "entry_id": "id_c",
                    "pdf_url": "url_c",
                    "summary": "A python paper",
                },
            ]
        }
        mock_genie_for_aggregator.execute_tool.side_effect = [
            mock_google_results,
            mock_arxiv_results,
        ]


        tool._test_mock_embedder.set_embeddings(  # type: ignore
            {
                "Paper C A python paper": [0.9, 0.1, 0.1],
                "Doc A About python programming": [0.8, 0.2, 0.2],
                "Doc B About data science": [0.1, 0.9, 0.1],
                "python research paper": [0.95, 0.05, 0.05],  # Query
            }
        )

        query = "python research paper"
        result = await tool.execute(
            {"query": query},
            key_provider=AsyncMock(),
            context={"genie_framework_instance": mock_genie_for_aggregator},
        )

        assert result["error"] is None
        assert len(result["results"]) == 3
        assert result["results"][0]["title"] == "Paper C"
        assert result["results"][1]["title"] == "Doc A"
        assert result["results"][2]["title"] == "Doc B"
        assert result["results"][0]["scores"]["combined_weighted"] > 0

        assert result["results"][0]["scores"]["semantic"] > 0.9
        assert result["results"][0]["scores"]["bm25"] is not None

    async def test_one_source_fails(
        self,
        aggregator_tool: IntelligentSearchAggregatorTool,
        mock_genie_for_aggregator: MagicMock,
    ):
        """Test that the tool continues if one of the search tools raises an exception."""
        tool = await aggregator_tool
        mock_google_results = {
            "results": [{"title": "Google Result", "url": "url_g", "description": "desc g"}]
        }
        mock_genie_for_aggregator.execute_tool.side_effect = [
            mock_google_results,
            RuntimeError("ArXiv is down"),
        ]
        result = await tool.execute(
            {"query": "test"},
            key_provider=AsyncMock(),
            context={"genie_framework_instance": mock_genie_for_aggregator},
        )
        assert len(result["results"]) == 1
        assert result["results"][0]["title"] == "Google Result"

    async def test_no_results_from_any_source(
        self,
        aggregator_tool: IntelligentSearchAggregatorTool,
        mock_genie_for_aggregator: MagicMock,
    ):
        """Test behavior when all sources return empty results."""
        tool = await aggregator_tool
        mock_genie_for_aggregator.execute_tool.return_value = {"results": []}
        result = await tool.execute(
            {"query": "obscure query"},
            key_provider=AsyncMock(),
            context={"genie_framework_instance": mock_genie_for_aggregator},
        )
        assert result["error"] == "No search results from any source."
        assert result["results"] == []

    @patch(RANK_BM25_AVAILABLE_PATH, False)
    async def test_bm25_unavailable_fallback(
        self,
        aggregator_tool: IntelligentSearchAggregatorTool,
        mock_genie_for_aggregator: MagicMock,
    ):
        """Test that BM25 scoring is skipped if the library is not available."""
        tool = await aggregator_tool
        mock_genie_for_aggregator.execute_tool.return_value = {
            "results": [{"title": "T", "url": "U", "description": "D"}]
        }
        result = await tool.execute(
            {"query": "test"},
            key_provider=AsyncMock(),
            context={"genie_framework_instance": mock_genie_for_aggregator},
        )
        assert result["results"][0]["scores"]["bm25"] == 0.0
