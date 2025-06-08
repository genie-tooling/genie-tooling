import asyncio
import logging
from typing import Any, Dict, List, Optional, cast

from genie_tooling.core.plugin_manager import PluginManager
from genie_tooling.lookup.types import RankedToolResult
from genie_tooling.tool_lookup_providers.abc import ToolLookupProvider

logger = logging.getLogger(__name__)

class HybridSearchLookupProvider(ToolLookupProvider):
    plugin_id: str = "hybrid_search_lookup_v1"
    description: str = "Combines results from dense (embedding) and sparse (keyword) search providers using Reciprocal Rank Fusion (RRF)."

    _dense_provider: Optional[ToolLookupProvider] = None
    _sparse_provider: Optional[ToolLookupProvider] = None
    _plugin_manager: Optional[PluginManager] = None

    async def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._plugin_manager = cfg.get("plugin_manager")
        if not self._plugin_manager:
            logger.error(f"{self.plugin_id}: PluginManager not provided. Cannot load sub-providers.")
            return

        dense_provider_id = cfg.get("dense_provider_id", "embedding_similarity_lookup_v1")
        sparse_provider_id = cfg.get("sparse_provider_id", "keyword_match_lookup_v1")

        dense_instance = await self._plugin_manager.get_plugin_instance(dense_provider_id, config=cfg.get("dense_provider_config", {}))
        if dense_instance and isinstance(dense_instance, ToolLookupProvider):
            self._dense_provider = cast(ToolLookupProvider, dense_instance)
        else:
            logger.error(f"{self.plugin_id}: Dense provider '{dense_provider_id}' not found or invalid.")

        sparse_instance = await self._plugin_manager.get_plugin_instance(sparse_provider_id, config=cfg.get("sparse_provider_config", {}))
        if sparse_instance and isinstance(sparse_instance, ToolLookupProvider):
            self._sparse_provider = cast(ToolLookupProvider, sparse_instance)
        else:
            logger.error(f"{self.plugin_id}: Sparse provider '{sparse_provider_id}' not found or invalid.")

    async def index_tools(self, tools_data: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None) -> None:
        if self._dense_provider:
            await self._dense_provider.index_tools(tools_data, config)
        if self._sparse_provider:
            await self._sparse_provider.index_tools(tools_data, config)

    async def add_tool(self, tool_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> bool:
        results = []
        if self._dense_provider:
            results.append(await self._dense_provider.add_tool(tool_data, config))
        if self._sparse_provider:
            results.append(await self._sparse_provider.add_tool(tool_data, config))
        return all(results)

    async def update_tool(self, tool_id: str, tool_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> bool:
        results = []
        if self._dense_provider:
            results.append(await self._dense_provider.update_tool(tool_id, tool_data, config))
        if self._sparse_provider:
            results.append(await self._sparse_provider.update_tool(tool_id, tool_data, config))
        return all(results)

    async def remove_tool(self, tool_id: str, config: Optional[Dict[str, Any]] = None) -> bool:
        results = []
        if self._dense_provider:
            results.append(await self._dense_provider.remove_tool(tool_id, config))
        if self._sparse_provider:
            results.append(await self._sparse_provider.remove_tool(tool_id, config))
        return all(results)

    async def find_tools(self, natural_language_query: str, top_k: int = 5, config: Optional[Dict[str, Any]] = None) -> List[RankedToolResult]:
        if not self._dense_provider or not self._sparse_provider:
            logger.error(f"{self.plugin_id}: Both dense and sparse providers must be configured to perform hybrid search.")
            return []

        cfg = config or {}
        dense_top_k = cfg.get("dense_top_k", top_k * 2)
        sparse_top_k = cfg.get("sparse_top_k", top_k * 2)
        k_constant = cfg.get("rrf_k_constant", 60)

        dense_results_task = self._dense_provider.find_tools(natural_language_query, dense_top_k, config)
        sparse_results_task = self._sparse_provider.find_tools(natural_language_query, sparse_top_k, config)

        dense_results, sparse_results = await asyncio.gather(dense_results_task, sparse_results_task)

        # Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        for rank, result in enumerate(dense_results):
            tool_id = result.tool_identifier
            rrf_scores[tool_id] = rrf_scores.get(tool_id, 0) + 1.0 / (k_constant + rank + 1)

        for rank, result in enumerate(sparse_results):
            tool_id = result.tool_identifier
            rrf_scores[tool_id] = rrf_scores.get(tool_id, 0) + 1.0 / (k_constant + rank + 1)

        if not rrf_scores:
            return []

        # CORRECTED: Intelligently merge results to preserve all diagnostic data
        combined_results: Dict[str, RankedToolResult] = {}
        for res in dense_results:
            if res.tool_identifier not in combined_results:
                combined_results[res.tool_identifier] = res

        for res in sparse_results:
            if res.tool_identifier in combined_results:
                # Merge diagnostic info into the existing result from the dense search
                existing_res = combined_results[res.tool_identifier]
                if res.matched_keywords:
                    existing_res.matched_keywords = res.matched_keywords
            else:
                # This tool was only found by sparse search
                combined_results[res.tool_identifier] = res


        sorted_tool_ids = sorted(rrf_scores.keys(), key=lambda tid: rrf_scores[tid], reverse=True)

        final_results: List[RankedToolResult] = []
        for tool_id in sorted_tool_ids[:top_k]:
            if tool_id in combined_results:
                result_to_add = combined_results[tool_id]
                result_to_add.score = rrf_scores[tool_id] # Update score to RRF score
                final_results.append(result_to_add)

        return final_results

    async def teardown(self) -> None:
        if self._dense_provider:
            await self._dense_provider.teardown()
        if self._sparse_provider:
            await self._sparse_provider.teardown()
        self._dense_provider = None
        self._sparse_provider = None
        self._plugin_manager = None
        logger.debug(f"{self.plugin_id}: Teardown complete.")
